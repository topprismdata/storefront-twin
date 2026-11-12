#!/usr/bin/env python3
"""
E4 局部 ROI 多向量匹配 — 左邻右舍门头细粒度识别
=================================================
设计文档 §7.1 B4（最佳基础表征 + ROI 多向量）/ §9.1 E4（多向量局部信息是否必要？）
/ §7.2 s_local（位置或语义区域约束的 patch Top-K matching，须与"只在类相关 ROI 上
比较"的强基线竞争）。

方法（DINOv2-base 冻结，patch tokens = last_hidden_state[:,1:]，16×16 网格）：
  gem          : patch GeM(p=3) 池化单向量 —— "类相关 ROI 全局描述子"强基线
  maxsim       : ColBERT 式 MaxSim，query 每 patch 取参考最大余弦，再对 patch 求均值
  maxsim_top50 : 仅取 query 判别性最强的 50% patch 的 MaxSim 均值
  maxsim_top25 : 同上，25%
  spatial_w2   : MaxSim + 空间窗约束（参考 patch 行列距 ≤2）
  spatial_w4   : MaxSim + 空间窗约束（≤4）
双视图：full（全图）/ crop（E0 检测器招牌 ROI，失败回退中心裁剪）。

评估：L1 候选簇 LOSO（与 E3 完全同协议：store_id 分组、留一店、R@1/R@5/k=5 投票）；
L0 品牌鉴别仅 GeM（单向量可 FAISS；MaxSim 全 330×330 代价高且非 E4 问题焦点）。
所有变体全部报告（敏感性分析），不事后挑选"最优"超参（§9.2 不据测试集调参）。

治理：真实图权利 pending_review，G0/G1 未过 → 探索性探针。确定性：无 RNG。
用法：
  python3 e4_local_match.py --extract   # 提取并缓存 patch tokens
  python3 e4_local_match.py --eval      # 评估（需先 extract）
  python3 e4_local_match.py --all
"""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np

import visual_baseline as vb

ROOT, CACHE, DATA = vb.ROOT, vb.CACHE, vb.DATA
E4_VERSION = "e4-local-v1.0"
GRID = 16                      # 224 / 14
VARIANTS = ["gem", "maxsim", "maxsim_top50", "maxsim_top25",
            "spatial_w2", "spatial_w4"]
VIEWS = ["full", "crop"]


# ---------------------------------------------------------------------------
# patch token 提取
# ---------------------------------------------------------------------------
def extract_patches(rows, views=VIEWS):
    import torch, cv2
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    path = str(vb.MODEL_CACHE / vb.MODELS["dinov2"])
    model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(vb.DEVICE)
    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    for view in views:
        out = CACHE / f"patches_{view}.npz"
        if out.exists():
            print(f"skip patches/{view} (cached)")
            continue
        toks, ids = [], []
        for i, r in enumerate(rows):
            bgr = cv2.imread(r["path"])
            if bgr is None:
                continue
            if view == "crop":
                x0, y0, x1, y1 = vb.sign_crop_box(bgr)
                bgr = bgr[max(y0, 0):y1, max(x0, 0):x1]
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inp = proc(images=[img], return_tensors="pt").to(vb.DEVICE)
            with torch.no_grad():
                t = model(**inp).last_hidden_state[0, 1:]   # (256, 768)
            t = t / t.norm(dim=-1, keepdim=True)
            toks.append(t.cpu().numpy().astype(np.float32))
            ids.append(r["image_id"])
            if (i + 1) % 100 == 0:
                print(f"  patches/{view} {i+1}/{len(rows)}")
        V = np.stack(toks)
        np.savez_compressed(out, vecs=V, ids=np.array(ids))
        print(f"saved patches/{view} {V.shape}")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def load_patches(view):
    z = np.load(CACHE / f"patches_{view}.npz", allow_pickle=True)
    return z["vecs"], list(z["ids"])


# ---------------------------------------------------------------------------
# 匹配打分
# ---------------------------------------------------------------------------
def gem_pool(P, p=3.0):
    """P: (N, 256, D) 已归一化 patch → GeM 池化单向量 (N, D)，再 L2 归一化。"""
    g = np.power(np.maximum(P, 0).mean(axis=1), 1.0 / p)
    return (g / np.linalg.norm(g, axis=-1, keepdims=True)).astype(np.float32)


def _spatial_mask(w):
    """(256,256) bool：参考 patch 与 query patch 行列距 ≤ w。"""
    rc = np.arange(GRID * GRID)
    qr, qc = rc // GRID, rc % GRID
    dr = np.abs(qr[:, None] - qr[None, :])
    dc = np.abs(qc[:, None] - qc[None, :])
    return (dr <= w) & (dc <= w)


def pair_scores(qp, rp, variant):
    """qp: (256,D)；rp: (M,256,D) → (M,)。"""
    M = rp.shape[0]
    if variant == "gem":
        gq = gem_pool(qp[None])[0]
        return gem_pool(rp) @ gq
    # (M, 256q, 256r) 余弦相似度
    S = np.einsum("mrd,qd->mrq", rp, qp, optimize=True)
    if variant.startswith("spatial"):
        w = int(variant.split("_w")[1])
        mask = _spatial_mask(w)                    # (q, r)
        S = np.where(mask[None, :, :], S, -np.inf)
    m = S.max(axis=2)                              # (M, 256q) 每 query patch 对参考 patch 取最大余弦
    if variant.startswith("maxsim_top"):
        f = int(variant.split("top")[1]) / 100.0
        k = max(1, int(np.ceil(f * GRID * GRID)))
        m = np.sort(m, axis=1)[:, -k:]
    valid = np.isfinite(m)
    denom = valid.sum(axis=1)
    s = np.where(valid, m, 0.0).sum(axis=1) / np.maximum(denom, 1)
    s[denom == 0] = -np.inf                        # 缺失 ROI：不拿 0 伪装负证据（§7.2）
    return s


# ---------------------------------------------------------------------------
# 评估（与 E3 同协议）
# ---------------------------------------------------------------------------
def score_matrix(pos, P, variant):
    """30×30（或 n×n）成对分数；对角线无意义（评估时按组排除）。"""
    n = len(pos)
    S = np.full((n, n), -np.inf, dtype=np.float32)
    for i in range(n):
        refs = np.concatenate([P[:i], P[i+1:]], axis=0) if n > 1 else P[:0]
        s = pair_scores(P[i], refs, variant)
        S[i, :i] = s[:i]
        S[i, i+1:] = s[i:]
    return S


def loso_from_scores(pos, S, k=5):
    results = []
    for i, r in enumerate(pos):
        order = np.argsort(-S[i])
        nbrs = []
        for j in order:
            if j == i or pos[j]["group"] == r["group"]:
                continue
            if not np.isfinite(S[i, j]):
                continue
            nbrs.append((pos[j]["candidate_style"], float(S[i, j])))
            if len(nbrs) >= k:
                break
        if not nbrs:
            continue
        top1 = nbrs[0][0]
        vote = Counter(s for s, _ in nbrs).most_common(1)[0][0]
        results.append({"image_id": r["image_id"], "group": r["group"],
                        "gt_style": r["candidate_style"], "top1": top1,
                        "vote": vote,
                        "recall1": top1 == r["candidate_style"],
                        "recall5": any(s == r["candidate_style"] for s, _ in nbrs),
                        "vote_correct": vote == r["candidate_style"]})
    return results


def eval_l0_gem(rows, view):
    """L0 品牌：GeM 单向量 kNN 投票（与 E3 eval_brand 同协议）。"""
    P, ids = load_patches(view)
    id2idx = {iid: i for i, iid in enumerate(ids)}
    rows = [r for r in rows if r["image_id"] in id2idx]
    G = gem_pool(P[[id2idx[r["image_id"]] for r in rows]])
    S = G @ G.T
    results = []
    for i, r in enumerate(rows):
        order = np.argsort(-S[i])
        votes = []
        for j in order:
            if j == i or rows[j]["group"] == r["group"]:
                continue
            votes.append(rows[j]["label"])
            if len(votes) >= 5:
                break
        pos_votes = sum(v == "positive" for v in votes)
        pred = "left_right" if pos_votes > 2.5 else "other_brand"
        gt = "left_right" if r["label"] == "positive" else "other_brand"
        results.append({"image_id": r["image_id"], "label": r["label"],
                        "group": r["group"], "pred_brand": pred, "gt_brand": gt,
                        "correct": pred == gt, "pos_votes": pos_votes})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    rows = vb.load_registry()
    print(f"registry rows: {len(rows)}")

    if args.extract or args.all:
        extract_patches(rows)

    if args.eval or args.all:
        pos_all = [r for r in rows if r["label"] == "positive" and r["candidate_style"]]
        summary = {}
        for view in VIEWS:
            P, ids = load_patches(view)
            id2idx = {iid: i for i, iid in enumerate(ids)}
            pos = [r for r in pos_all if r["image_id"] in id2idx]
            Ppos = P[[id2idx[r["image_id"]] for r in pos]]
            for variant in VARIANTS:
                S = score_matrix(pos, Ppos, variant)
                res = loso_from_scores(pos, S)
                key = f"{view}/{variant}"
                summary[key] = vb.summarize_cluster(res)
                json.dump(res, open(CACHE / f"e4_results_{view}_{variant}.json", "w"),
                          ensure_ascii=False, indent=1)
                c = summary[key]
                print(f"{key:22s} R1 {c['recall1']:.3f}  R5 {c['recall5']:.3f}  vote {c['vote_acc']:.3f}")
            b = eval_l0_gem(rows, view)
            bs = vb.summarize_brand(b)
            summary[f"{view}/gem_L0"] = bs
            json.dump(b, open(CACHE / f"e4_results_{view}_gem_brand.json", "w"),
                      ensure_ascii=False, indent=1)
            print(f"{view}/gem L0 brand: recall {bs['brand_recall']:.3f} spec {bs['brand_specificity']:.3f}")
        json.dump(summary, open(CACHE / "e4_summary.json", "w"), ensure_ascii=False, indent=1)
        manifest = {
            "experiment": "E4", "version": E4_VERSION,
            "question": "多向量局部信息是否必要（相对全局 CLS 与 ROI-GeM 强基线）",
            "backbone": "dinov2-base (frozen)", "patch_grid": [GRID, GRID],
            "views": VIEWS, "variants": VARIANTS,
            "gem_p": 3.0, "k_vote": 5,
            "split": "store_id_grouped_LOSO",
            "hyperparam_policy": "all variants reported; no post-hoc selection on test",
            "determinism": "no RNG; frozen weights; fixed preprocessing",
            "governance": "exploratory_probe_rights_pending_G0_G1_not_passed",
            "e3_reference": "vis_cache/summary.json (dinov2/clip/siglip global)",
        }
        json.dump(manifest, open(CACHE / "e4_manifest.json", "w"), ensure_ascii=False, indent=1)
        print("\nsummary →", CACHE / "e4_summary.json")


if __name__ == "__main__":
    main()