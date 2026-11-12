#!/usr/bin/env python3
"""
E9 AnyLoc 式局部聚合 — 撤回 E4 错误结论的修正实验
====================================================
文献依据 AnyLoc (arXiv:2308.00688, RA-L 2023)：
  - DINOv2 中间偏深层 value facet（非最后层 token）局部判别最强；
  - hard-assignment VLAD（聚类残差 + intra/inter 归一化）显著优于 GeM/MaxSim/CLS；
  - 域特定词表（domain-specific vocabulary）再提升。
E4 用了最后层 token + MaxSim/GeM，等于没测正确配方 → 本实验修正。

免训练、免微调（治理 ✓）。确定性：KMeans 固定 seed。
评估：L1 LOSO（store_id 分组）+ L0 品牌 kNN，对照 E3 CLS / E4 配方。
用法：python3 e9_anyloc.py --all
"""
import argparse, json
from collections import Counter
import numpy as np
import cv2
from PIL import Image
import visual_baseline as vb

CACHE = vb.CACHE
E9_VERSION = "e9-anyloc-v1.0"
# DINOv2-base 12 层；AnyLoc 用 ~3/4 深度（ViT-G 31/40）。base 对应层 8/9/10。
LAYERS = [6, 8, 10]
FACETS = ["value", "token"]          # token=残差流（E4 用的），value=AnyLoc 正确选择
K_CLUSTERS = 32                      # AnyLoc ViT-G 用 32；base 特征 768d


def extract_facets(rows, view="crop"):
    """提取多中间层 value/token facet patch 特征。缓存 npz。"""
    import torch
    from transformers import AutoModel, AutoProcessor
    out = CACHE / f"e9_facets_{view}.npz"
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return {k: z[k] for k in z.files if k != "ids"}, list(z["ids"])
    path = str(vb.MODEL_CACHE / vb.MODELS["dinov2"])
    model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(vb.DEVICE)
    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    hooks = {}
    def mk_hook(name):
        def hook(mod, inp, outp):
            hooks[name] = outp.detach()
        return hook
    handles = []
    for li in LAYERS:
        attn = model.encoder.layer[li].attention.attention
        handles.append(attn.value.register_forward_hook(mk_hook(f"L{li}_value")))
    feats = {f"L{li}_{f}": [] for li in LAYERS for f in FACETS}
    ids = []
    for i, r in enumerate(rows):
        bgr = cv2.imread(r["path"])
        if view == "crop":
            x0, y0, x1, y1 = vb.sign_crop_box(bgr)
            bgr = bgr[max(y0,0):y1, max(x0,0):x1]
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inp = proc(images=[img], return_tensors="pt").to(vb.DEVICE)
        with torch.no_grad():
            hout = model(**inp, output_hidden_states=True)
        for li in LAYERS:
            tok = hout.hidden_states[li+1][0, 1:]      # 残差流（层输出）
            val = hooks[f"L{li}_value"][0, 1:]        # (256, D) 去 CLS，与 token 对齐
            feats[f"L{li}_token"].append(tok.cpu().numpy().astype(np.float32))
            feats[f"L{li}_value"].append(val.cpu().numpy().astype(np.float32))
        ids.append(r["image_id"])
        if (i+1) % 100 == 0:
            print(f"  facets/{view} {i+1}/{len(rows)}")
    for h in handles:
        h.remove()
    feats = {k: np.stack(v) for k, v in feats.items()}
    np.savez_compressed(out, ids=np.array(ids), **feats)
    print(f"saved facets/{view}", {k: v.shape for k, v in feats.items()})
    return feats, ids


def hard_vlad(P, centers):
    """P: (N, 256, D) 局部特征；centers: (K, D)。hard-VLAD + intra/inter 归一化。
    返回 (N, K*D) 全局描述子。"""
    N, T, D = P.shape
    K = centers.shape[0]
    Pf = P.reshape(N*T, D)
    Pf = Pf / np.maximum(np.linalg.norm(Pf, axis=-1, keepdims=True), 1e-9)
    C = centers / np.maximum(np.linalg.norm(centers, axis=-1, keepdims=True), 1e-9)
    sim = Pf @ C.T                              # (N*T, K)
    assign = sim.argmax(axis=1)                 # hard assignment
    resid = Pf - C[assign]                      # 残差
    V = np.zeros((N, K, D), dtype=np.float32)
    np.add.at(V, (np.repeat(np.arange(N), T), assign), resid)
    # intra-normalization（每簇）
    V = V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-9)
    V = V.reshape(N, K*D)
    # inter-normalization（整体 L2）
    V = V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-9)
    return V.astype(np.float32)


def build_vocab(P, k=K_CLUSTERS, seed=42):
    """域词表：从所有 patch 特征 KMeans 聚类（AnyLoc 域特定词表）。"""
    from sklearn.cluster import MiniBatchKMeans
    Pf = P.reshape(-1, P.shape[-1])
    Pf = Pf / np.maximum(np.linalg.norm(Pf, axis=-1, keepdims=True), 1e-9)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096).fit(Pf)
    return km.cluster_centers_.astype(np.float32)


def gem_pool(P, p=3.0):
    g = np.power(np.maximum(P, 0).mean(axis=1), 1.0/p)
    return (g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)).astype(np.float32)


def loso_eval(rows, desc, pos_mask, k=5):
    """L1 LOSO：正例 query 从其余店正例检索。desc 已 L2 归一化。"""
    pos = [i for i in range(len(rows)) if pos_mask[i]]
    D = desc[pos]
    S = D @ D.T
    res = []
    for a, i in enumerate(pos):
        r = rows[i]
        order = np.argsort(-S[a])
        nbrs = []
        for b in order:
            j = pos[b]
            if j == i or rows[j]["group"] == r["group"]:
                continue
            nbrs.append(rows[j]["candidate_style"])
            if len(nbrs) >= k:
                break
        if not nbrs:
            continue
        vote = Counter(nbrs).most_common(1)[0][0]
        res.append({"image_id": r["image_id"], "gt_style": r["candidate_style"],
                    "recall1": nbrs[0] == r["candidate_style"],
                    "recall5": r["candidate_style"] in nbrs,
                    "vote_correct": vote == r["candidate_style"]})
    return res


def brand_eval(rows, desc, k=5):
    S = desc @ desc.T
    res = []
    for i, r in enumerate(rows):
        order = np.argsort(-S[i])
        votes = []
        for j in order:
            if j == i or rows[j]["group"] == r["group"]:
                continue
            votes.append(rows[j]["label"] == "positive")
            if len(votes) >= k:
                break
        score = np.mean(votes) if votes else 0.0
        res.append({"image_id": r["image_id"], "label": r["label"], "score": float(score)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--view", default="crop")
    args = ap.parse_args()
    rows = vb.load_registry()
    pos_mask = np.array([(r["label"] == "positive") and bool(r["candidate_style"]) for r in rows])
    feats, ids = extract_facets(rows, args.view)
    id2i = {iid: i for i, iid in enumerate(ids)}
    rows = [r for r in rows if r["image_id"] in id2i]
    pos_mask = np.array([(r["label"] == "positive") and bool(r["candidate_style"]) for r in rows])
    idx = [id2i[r["image_id"]] for r in rows]

    summary = {}
    for key in feats:
        P = feats[key][idx]
        # 词表用全部图（域特定：真实+合成店面域）
        centers = build_vocab(P)
        for agg, desc in [("vlad", hard_vlad(P, centers)), ("gem", gem_pool(P))]:
            desc = desc / np.maximum(np.linalg.norm(desc, axis=-1, keepdims=True), 1e-9)
            c = loso_eval(rows, desc, pos_mask)
            cs = vb.summarize_cluster([dict(r, group="") for r in c])
            b = brand_eval(rows, desc)
            bs = vb.summarize_brand([dict(r, correct=(r["score"] > 0.5) == (r["label"] == "positive"),
                                          pred_brand="left_right" if r["score"] > 0.5 else "other_brand",
                                          gt_brand="left_right" if r["label"] == "positive" else "other_brand") for r in b])
            tag = f"{key}/{agg}"
            summary[tag] = {"cluster": cs, "brand": bs}
            print(f"{tag:22s} L1 R1 {cs['recall1']:.3f} R5 {cs['recall5']:.3f} | L0 rec {bs['brand_recall']:.3f} spec {bs['brand_specificity']:.3f}")
    json.dump(summary, open(CACHE/"e9_summary.json", "w"), ensure_ascii=False, indent=1)
    json.dump(dict(experiment="E9", version=E9_VERSION,
                   method="AnyLoc-style: intermediate-layer value facet + hard-VLAD + domain vocab",
                   reference="arXiv:2308.00688", layers=LAYERS, facets=FACETS, k_clusters=K_CLUSTERS,
                   view=args.view, split="store_id_grouped_LOSO",
                   governance="exploratory_probe_rights_pending_G0_G1_not_passed",
                   note="corrects E4 which used last-layer token + MaxSim/GeM"),
              open(CACHE/"e9_manifest.json", "w"), ensure_ascii=False, indent=1)
    print("\n→", CACHE/"e9_summary.json")


if __name__ == "__main__":
    main()