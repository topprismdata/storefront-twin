#!/usr/bin/env python3
"""score_storefront.py — 单图评分 CLI (部署成品入口)。
用法: python3 synthetic/score_storefront.py <image_path> [--json]
输出: 三通道分数 + 融合分 + 阈值判定。
模型卡: synthetic/vis_cache/e14_sign/model_card.json (E6-v3 LOSO 泛化数字; 部署需权利链)。"""
import json, sys
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))
import visual_baseline as vb
import e8_train as e8
import e8_multicrop as em

ART = ROOT / "synthetic/vis_cache/e14_sign"


def dino_mc_single(img_path, pil):
    """多候选裁剪 DINOv2: 画廊候选向量缓存(npz, 按画廊id列表哈希) + 查询图候选最优相似度。"""
    import cv2, hashlib
    import e8_train as e8
    reg = json.load(open(ROOT / "data/registry_v11.json"))
    sha_idx = json.load(open(ROOT / "data/sha_file_index.json"))
    gal = []
    for r in reg:
        if r["label"] == "positive" and r.get("is_representative"):
            p = sha_idx.get(r["sha256"])
            if p:
                gal.append(r["image_id"])
    key = hashlib.sha256(("|".join(gal)).encode()).hexdigest()[:16]
    cfile = ART / f"dino_gallery_{key}.npz"
    if cfile.exists():
        z = np.load(cfile)
        ref = z["ref"]; groups = list(z["groups"])
    else:
        # 与 rescore 完全一致的 647 行构造 → e8.embed_all 缓存命中
        rows_g = []
        for r in reg:
            if not r.get("is_representative"):
                continue
            p = sha_idx.get(r["sha256"])
            if not p:
                continue
            fp = ROOT / "data" / p
            if fp.exists():
                rows_g.append({"image_id": r["image_id"], "path": str(fp),
                               "label": r["label"], "group": r.get("store_id") or r["image_id"]})
        rf, _ = e8.embed_all(rows_g, "v4_real")
        pmask = np.array([r["label"] == "positive" for r in rows_g])
        ref = e8.l2(np.concatenate([rf["dinov2_full"][pmask], rf["dinov2_crop"][pmask]], axis=0))
        groups = [r["group"] for r, m in zip(rows_g, pmask) if m for _ in range(2)]
        np.savez(cfile, ref=ref, groups=np.array(groups))
    emb = vb.Embedder("dinov2")
    import cv2 as _cv
    bgr = _cv.imread(str(img_path))
    boxes = vb.sign_crop_candidates(bgr)
    cands = []
    for (x0, y0, x1, y1) in boxes:
        c = bgr[max(y0, 0):y1, max(x0, 0):x1]
        if c.size == 0:
            continue
        img = Image.fromarray(_cv.cvtColor(c, _cv.COLOR_BGR2RGB))
        cands.append(emb.embed_pil([img])[0])
    if not cands:
        cands = [emb.embed_pil([pil])[0]]
    Q = e8.l2(np.stack(cands))
    S = Q @ ref.T
    return float(np.clip((S.max() - 0.3) / 0.5, 0, 1))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    img_path = Path(sys.argv[1])
    as_json = "--json" in sys.argv
    assert img_path.exists(), img_path
    z = np.load(ART / "final_probe.npz", allow_pickle=True)
    layout = list(z["feat_layout"])

    from PIL import Image
    import cv2
    pil = Image.open(img_path).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # 特征
    feats = {}
    emb = vb.Embedder("clip")
    feats["clip_full"] = emb.embed_pil([pil])[0][None, :]
    d = vb.Embedder("dinov2")
    crop_bgr = bgr[vb.sign_crop_box(bgr)[1]:vb.sign_crop_box(bgr)[3],
                   vb.sign_crop_box(bgr)[0]:vb.sign_crop_box(bgr)[2]]
    crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    feats["dinov2_full"] = d.embed_pil([pil])[0][None, :]
    feats["dinov2_crop"] = d.embed_pil([crop_pil])[0][None, :]
    x = np.concatenate([feats[k] / max(1e-9, np.linalg.norm(feats[k])) for k in layout], axis=1)

    # 探针分 (必须过训练时的 StandardScaler)
    xs = (x - z["mean"]) / z["scale"]
    s_probe = float(sigmoid(xs @ z["coef"].T + z["intercept"])[0][0])

    # kNN 画廊分 (registry v11 正例画廊)
    reg = json.load(open(ROOT / "data/registry_v11.json"))
    sha_idx = json.load(open(ROOT / "data/sha_file_index.json"))
    gal_ids, gal_paths = [], []
    for r in reg:
        if r["label"] == "positive" and r.get("is_representative"):
            p = sha_idx.get(r["sha256"])
            if p:
                gal_paths.append(str(ROOT / "data" / p)); gal_ids.append(r["image_id"])
    cv_g, ci_g = vb.load_emb("clip", "full")
    id2i = {x_: i for i, x_ in enumerate(ci_g)}
    miss = [i for i in gal_ids if i not in id2i]
    if miss:
        for pth, iid in zip(gal_paths, gal_ids):
            if iid not in id2i:
                p2 = Image.open(pth).convert("RGB")
                v = emb.embed_pil([p2])[0]
                cv_g = np.vstack([cv_g, v[None, :]])
                ci_g.append(iid)
                id2i[iid] = len(ci_g) - 1
    gal = cv_g[[id2i[i] for i in gal_ids]]
    q = feats["clip_full"] / max(1e-9, np.linalg.norm(feats["clip_full"]))
    sim = (gal @ q[0])
    s_clip = float(np.sort(sim)[-5:].mean())  # top-5 均值画廊相似度(协议)

    # multicrop DINO 分 (画廊候选缓存 + 单图查询)
    s_dino = dino_mc_single(img_path, pil)

    s = 0.4 * s_clip + 0.4 * s_dino + 0.2 * s_probe
    verdict = "brand_storefront" if s > 0.80 else ("collect_review" if s > 0.636 else "not_brand")
    out = {"image": str(img_path), "s_clip": round(s_clip, 4), "s_dino_mc": round(s_dino, 4),
           "s_probe": round(s_probe, 4), "fusion": round(s, 4),
           "verdict": verdict,
           "note": "0.636=采集筛选线, 0.80=部署操作点; 证据层四态裁决(E10)另需 VLM 流程"}
    if as_json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        print(f"图: {img_path.name}")
        print(f"  s_clip    {s_clip:.4f}")
        print(f"  s_dino_mc {s_dino:.4f}")
        print(f"  s_probe   {s_probe:.4f}")
        print(f"  融合      {s:.4f}  → {verdict}")


if __name__ == "__main__":
    main()
