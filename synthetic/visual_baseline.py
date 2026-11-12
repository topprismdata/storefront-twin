#!/usr/bin/env python3
"""
E1/E3 视觉表征基线 — 左邻右舍门头识别（主力路线，非 OCR）
==========================================================
设计文档 §7.1 B1/B2/B3 / §9.1 E1/E3：CLIP / SigLIP / DINOv2 冻结编码器 + FAISS 检索。
文献依据：品牌/logo 识别属视觉检索（LogoDet-3K），DINOv2 局部特征擅长细粒度结构；
OCR 仅辅助消歧（见 E0_REPORT.md §4 架构纠偏）。

评估两任务：
  L0 品牌鉴别：query 图 → 参考集 kNN 投票（left_right vs other_brand）。
  L1 候选簇检索：留一店交叉验证（LOSO），query 店从其余店参考集检索，Recall@1/@5 + 簇分类。

隔离（§9.2）：按 store_id 分组；无 store_id 的正例各自独立组；负例独立组。
治理：真实图权利 pending_review，G0/G1 未过 → 探索性探针，非正式精度宣称。

模型权重：本地 models/ 目录（requests 直下，绕过 hf_hub 离线限制）。
用法：
  python3 visual_baseline.py --embed     # 计算并缓存嵌入
  python3 visual_baseline.py --eval      # 评估（需先 embed）
  python3 visual_baseline.py --all
"""
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
MODEL_CACHE = ROOT.parent / "models"
CACHE = ROOT / "vis_cache"
BASELINE_VERSION = "e1e3-visual-v1.0"
MODELS = {
    "dinov2": "facebook__dinov2-base",
    "clip": "openai__clip-vit-base-patch32",
    "siglip": "google__siglip-base-patch16-224",
}
DEVICE = "mps"


# ---------------------------------------------------------------------------
# 数据加载 + 招牌裁剪
# ---------------------------------------------------------------------------
def load_registry():
    reg = json.load(open(DATA / "registry_v3.json"))
    sha_idx = json.load(open(DATA / "sha_file_index.json"))
    rows = []
    for r in reg:
        if not r.get("is_representative"):
            continue
        p = sha_idx.get(r["sha256"])
        if not p:
            continue
        fp = DATA / p
        if not fp.exists():
            continue
        # 分组键：store_id 优先，否则 image_id 独立组
        group = r.get("store_id") or r["image_id"]
        rows.append({
            "image_id": r["image_id"], "path": str(fp), "label": r["label"],
            "neg_type": r.get("neg_type", ""), "group": group,
            "candidate_style": (r.get("candidate_style") or "").split("_")[0],
            "brand_state": r.get("brand_state", ""),
        })
    return rows


def sign_crop_box(img):
    """复用 E0 投影检测器取招牌框；失败则中心裁剪。返回 (x0,y0,x1,y1)。"""
    import cv2
    sys.path.insert(0, str(ROOT))
    import e0_baseline as e0
    cands = e0.detect_sign_candidates(img)
    H, W = img.shape[:2]
    if cands:
        return cands[0]["box"]
    # 中心裁剪（上中部，招牌常见位置）
    return [int(W*0.15), int(H*0.20), int(W*0.85), int(H*0.60)]


def sign_crop_candidates(img, max_n=3):
    """返回多个招牌候选裁剪框（检测器候选 + 上中部回退），供多裁剪取最优相似度。
    修复 FN：单一检测失败回退中心裁剪会嵌入错误区域。"""
    import e0_baseline as e0
    H, W = img.shape[:2]
    cands = e0.detect_sign_candidates(img)
    boxes = [c["box"] for c in cands[:max_n]]
    # 回退/补充：上中部宽裁剪（招牌常见位置，比单中心框覆盖更全）
    boxes.append([int(W*0.08), int(H*0.12), int(W*0.92), int(H*0.62)])
    # 去重过近框
    out = []
    for b in boxes:
        if b[2]-b[0] < 16 or b[3]-b[1] < 16:
            continue
        if all(abs(b[0]-o[0])+abs(b[1]-o[1]) > 0.05*W for o in out):
            out.append(b)
    return out or [[int(W*0.15), int(H*0.20), int(W*0.85), int(H*0.60)]]

# ---------------------------------------------------------------------------
# 嵌入模型
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, name):
        import torch
        from transformers import AutoModel, AutoProcessor, CLIPModel, SiglipModel
        self.name = name
        self.torch = torch
        path = str(MODEL_CACHE / MODELS[name])
        if name == "dinov2":
            self.model = AutoModel.from_pretrained(path, local_files_only=True)
            self.proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        elif name == "clip":
            self.model = CLIPModel.from_pretrained(path, local_files_only=True)
            self.proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        else:
            self.model = SiglipModel.from_pretrained(path, local_files_only=True)
            self.proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model.eval().to(DEVICE)

    def embed_pil(self, imgs):
        """imgs: list[PIL.Image] → L2 归一化嵌入 (N, D)。"""
        torch = self.torch
        inputs = self.proc(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            if self.name == "dinov2":
                out = self.model(**inputs).last_hidden_state[:, 0]  # CLS
            elif self.name == "clip":
                out = self.model.get_image_features(**inputs)
            else:
                out = self.model.get_image_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True)
        return out.cpu().numpy().astype(np.float32)


def compute_embeddings(rows, views=("full", "crop"), models=None):
    from PIL import Image
    import cv2
    CACHE.mkdir(exist_ok=True)
    for name in (models or list(MODELS)):
        emb = Embedder(name)
        for view in views:
            cache_f = CACHE / f"{name}_{view}.npz"
            if cache_f.exists():
                print(f"skip {name}/{view} (cached)")
                continue
            vecs, ids = [], []
            batch_imgs, batch_ids = [], []
            def flush():
                if batch_imgs:
                    v = emb.embed_pil(batch_imgs)
                    vecs.append(v); ids.extend(batch_ids)
                    batch_imgs.clear(); batch_ids.clear()
            for i, r in enumerate(rows):
                img = cv2.imread(r["path"])
                if img is None:
                    continue
                if view == "crop":
                    x0, y0, x1, y1 = sign_crop_box(img)
                    img = img[y0:y1, x0:x1]
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                batch_imgs.append(pil); batch_ids.append(r["image_id"])
                if len(batch_imgs) >= 16:
                    flush()
                if (i+1) % 100 == 0:
                    print(f"  {name}/{view} {i+1}/{len(rows)}")
            flush()
            V = np.concatenate(vecs, 0)
            np.savez(cache_f, vecs=V, ids=np.array(ids))
            print(f"saved {name}/{view} {V.shape}")
        del emb
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# FAISS 检索评估
# ---------------------------------------------------------------------------
def load_emb(name, view):
    z = np.load(CACHE / f"{name}_{view}.npz", allow_pickle=True)
    return z["vecs"], list(z["ids"])


def faiss_index(vecs):
    import faiss
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return idx


def eval_brand(rows, name, view, k=5):
    """L0 品牌鉴别：每个正/负例 query，从其余样本参考集 kNN 投票。"""
    vecs, ids = load_emb(name, view)
    id2row = {r["image_id"]: r for r in rows}
    id2idx = {iid: i for i, iid in enumerate(ids)}
    idx = faiss_index(vecs)
    results = []
    for r in rows:
        if r["image_id"] not in id2idx:
            continue
        qi = id2idx[r["image_id"]]
        # 检索 k+1 排除自身，且排除同组（门店隔离）
        D, I = idx.search(vecs[qi:qi+1], k+1+len(rows))
        votes = []
        for dist, j in zip(D[0], I[0]):
            if j == qi:
                continue
            nb = id2row[ids[j]]
            if nb["group"] == r["group"]:
                continue
            votes.append((nb["label"], float(dist)))
            if len(votes) >= k:
                break
        pos_votes = sum(1 for lab, _ in votes if lab == "positive")
        pred = "left_right" if pos_votes > k/2 else "other_brand"
        gt = "left_right" if r["label"] == "positive" else "other_brand"
        results.append({"image_id": r["image_id"], "label": r["label"],
                        "neg_type": r.get("neg_type"), "group": r["group"],
                        "pred_brand": pred, "gt_brand": gt,
                        "correct": pred == gt, "pos_votes": pos_votes,
                        "top_dist": votes[0][1] if votes else 0.0})
    return results


def eval_cluster_loso(rows, name, view, k=5):
    """L1 候选簇检索：留一店（LOSO）。正例 query 从其余店正例参考集检索。"""
    vecs, ids = load_emb(name, view)
    id2row = {r["image_id"]: r for r in rows}
    id2idx = {iid: i for i, iid in enumerate(ids)}
    pos = [r for r in rows if r["label"] == "positive" and r["candidate_style"]]
    pos_ids = [r["image_id"] for r in pos if r["image_id"] in id2idx]
    pvecs = vecs[[id2idx[i] for i in pos_ids]]
    idx = faiss_index(pvecs)
    results = []
    for r in pos:
        if r["image_id"] not in id2idx:
            continue
        qi = pos_ids.index(r["image_id"])
        D, I = idx.search(pvecs[qi:qi+1], k+1+len(pos))
        nbrs = []
        for dist, j in zip(D[0], I[0]):
            if j == qi:
                continue
            nb = pos[j]
            if nb["group"] == r["group"]:   # 留一店
                continue
            nbrs.append((nb["candidate_style"], float(dist)))
            if len(nbrs) >= k:
                break
        if not nbrs:
            continue
        top1 = nbrs[0][0]
        recall1 = top1 == r["candidate_style"]
        recall5 = any(s == r["candidate_style"] for s, _ in nbrs[:5])
        # kNN 簇投票
        vote = Counter(s for s, _ in nbrs).most_common(1)[0][0]
        results.append({"image_id": r["image_id"], "group": r["group"],
                        "gt_style": r["candidate_style"], "top1": top1,
                        "vote": vote, "recall1": recall1, "recall5": recall5,
                        "vote_correct": vote == r["candidate_style"]})
    return results


def summarize_brand(res):
    pos = [r for r in res if r["label"] == "positive"]
    neg = [r for r in res if r["label"] == "negative"]
    rec = sum(r["correct"] for r in pos)/max(len(pos), 1)
    spec = sum(r["correct"] for r in neg)/max(len(neg), 1)
    return {"n_pos": len(pos), "n_neg": len(neg),
            "brand_recall": round(rec, 3), "brand_specificity": round(spec, 3)}


def summarize_cluster(res):
    if not res:
        return {"n": 0}
    return {"n": len(res),
            "recall1": round(sum(r["recall1"] for r in res)/len(res), 3),
            "recall5": round(sum(r["recall5"] for r in res)/len(res), 3),
            "vote_acc": round(sum(r["vote_correct"] for r in res)/len(res), 3),
            "per_style": {s: {"n": sum(1 for r in res if r["gt_style"] == s),
                              "r1": round(sum(r["recall1"] for r in res if r["gt_style"] == s)/
                                          max(sum(1 for r in res if r["gt_style"] == s), 1), 3)}
                          for s in sorted({r["gt_style"] for r in res})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--models", default="dinov2,clip,siglip")
    ap.add_argument("--views", default="full,crop")
    args = ap.parse_args()
    rows = load_registry()
    print(f"registry rows: {len(rows)} "
          f"(pos {sum(r['label']=='positive' for r in rows)}, "
          f"neg {sum(r['label']=='negative' for r in rows)})")
    models = args.models.split(",")
    views = args.views.split(",")

    if args.embed or args.all:
        compute_embeddings(rows, views, models=models)

    if args.eval or args.all:
        allres = {}
        for name in models:
            for view in views:
                if not (CACHE / f"{name}_{view}.npz").exists():
                    continue
                bres = eval_brand(rows, name, view)
                cres = eval_cluster_loso(rows, name, view)
                bs, cs = summarize_brand(bres), summarize_cluster(cres)
                allres[f"{name}/{view}"] = {"brand": bs, "cluster_loso": cs}
                print(f"\n=== {name}/{view} ===")
                print("L0 brand:", json.dumps(bs, ensure_ascii=False))
                print("L1 cluster LOSO:", json.dumps(cs, ensure_ascii=False))
                (CACHE / f"results_{name}_{view}_brand.json").write_text(
                    json.dumps(bres, ensure_ascii=False, indent=1))
                (CACHE / f"results_{name}_{view}_cluster.json").write_text(
                    json.dumps(cres, ensure_ascii=False, indent=1))
        (CACHE / "summary.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1))
        manifest = {"baseline_version": BASELINE_VERSION, "models": models,
                    "views": views, "device": DEVICE, "k": 5,
                    "split": "store_id_grouped_LOSO",
                    "governance": "exploratory_probe_rights_pending_G0_G1_not_passed"}
        (CACHE / "eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        print("\nsummary →", CACHE / "summary.json")


if __name__ == "__main__":
    main()