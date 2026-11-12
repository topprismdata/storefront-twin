#!/usr/bin/env python3
"""
E11 — 真实图驱动风格簇审计：CS-01..06 候选簇 vs 1271+ 张已核验真图分布
====================================================================
目的：候选簇(candidate_unverified)第一次接受真实分布检验。
方法：
  1. 合成参考：renders_3d 360 张 positive(CS-01..06×60) 按簇求质心
  2. 真实正例：E6 池 A 类(已核验,12省) 最近质心指派 + 相似度分布
  3. 交叉验证：真图 kMeans(k=4..10) 与 CS 指类的 ARI/NMI 一致性
  4. 覆盖缺口：低相似度图 = 现有 6 簇不覆盖的真实变体 → 孪生 v2 渲染目标
产物：vis_cache/e6/style_audit.json + 不覆盖图清单(供接触表人工核验)
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import e6_real_probe as e1
import e6_real_probe_v2 as e2
import visual_baseline as vb

E6 = e1.E6
ROOT = e1.ROOT


def norm_block(X, d_clip=512):
    """clip(512)与dino(768)各自L2后concat再整体L2——与探针特征同构。"""
    c = X[:, :d_clip] / (np.linalg.norm(X[:, :d_clip], axis=1, keepdims=True) + 1e-9)
    d = X[:, d_clip:] / (np.linalg.norm(X[:, d_clip:], axis=1, keepdims=True) + 1e-9)
    F = np.concatenate([c, d], 1)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def embed_synth_styles():
    """360 张合成 positive → 每簇质心。缓存。"""
    from PIL import Image
    import cv2
    cf = E6 / "synth_style_emb.npz"
    if cf.exists():
        z = np.load(cf, allow_pickle=True)
        return dict(zip(z["ids"].tolist(), z["vecs"]))
    rows = []
    for line in open(ROOT / "synthetic/renders_3d/manifest_3d.jsonl"):
        r = json.loads(line)
        if r.get("brand") == "left_right":
            fp = ROOT / "synthetic/renders_3d" / r["image"]
            if fp.exists():
                rows.append({"image_id": "s_" + r["image"], "path": str(fp),
                             "style": r.get("style_id")})
    emb_c = vb.Embedder("clip")
    emb_d = vb.Embedder("dinov2")
    Vc, Vd, ids = [], [], []
    B = []
    for r in rows:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        B.append((r, Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
    for i in range(0, len(B), 16):
        chunk = B[i:i + 16]
        pil = [x[1] for x in chunk]
        Vc.append(emb_c.embed_pil(pil))
        Vd.append(emb_d.embed_pil(pil))
        ids.extend(x[0]["image_id"] for x in chunk)
    V = np.concatenate([np.concatenate(Vc, 0), np.concatenate(Vd, 0)], 1).astype(np.float32)
    style_of = {x[0]["image_id"]: x[0]["style"] for x in B}
    np.savez(cf, vecs=V, ids=np.array(ids),
             styles=np.array([style_of[i] for i in ids]))
    del emb_c, emb_d
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return dict(zip(ids, V))


def main():
    # 1) 真实正例（E6 v2 池的 A 类，排除合成）
    rows = [r for r in e2.build_pool_v2() if r["lab"] == "A" and r["src"] != "render3d"]
    # 真图特征：pool_* + x_* 缓存
    m = {}
    for n in ("clip", "dinov2"):
        for v in ("full", "crop"):
            for tag in ("pool", "x"):
                f = E6 / f"{tag}_{n}_{v}.npz"
                if f.exists():
                    z = np.load(f, allow_pickle=True)
                    m[(n, v)] = m.get((n, v), {})
                    m[(n, v)].update(zip(z["ids"].tolist(), z["vecs"]))
    feats = {}
    for n in ("clip", "dinov2"):
        for v in ("full", "crop"):
            for r in rows:
                if r["image_id"] in m[(n, v)]:
                    feats.setdefault(r["image_id"], {})[(n, v)] = m[(n, v)][r["image_id"]]
    keep = [r for r in rows
            if r["image_id"] in feats and all((n, v) in feats[r["image_id"]] for n in ("clip", "dinov2") for v in ("full", "crop"))]
    X = np.concatenate([np.stack([feats[r["image_id"]][(n, v)] for r in keep])
                        for n in ("clip", "dinov2") for v in ("full",)], 1).astype(np.float32)
    print(f"真实正例: {len(keep)} 图")
    F = norm_block(X)

    # 2) 合成簇质心
    sv = embed_synth_styles()
    z = np.load(E6 / "synth_style_emb.npz", allow_pickle=True)
    styles = z["styles"].tolist()
    cent = {}
    for s in sorted(set(styles)):
        idx = [i for i, x in enumerate(styles) if x == s]
        c = z["vecs"][idx].mean(0)
        c = c / (np.linalg.norm(c) + 1e-9)
        cent[s] = c
    S = np.stack([cent[s] for s in sorted(cent)])
    snames = sorted(cent)
    sims = F @ S.T  # (N,6)
    assign = [snames[i] for i in sims.argmax(1)]
    top1 = sims.max(1)
    top2 = np.sort(sims, 1)[:, -2]
    margin = top1 - top2

    dist = Counter(assign)
    print("CS指派分布:", dict(dist))
    print(f"top1相似度: mean={top1.mean():.3f} p10={np.quantile(top1,0.1):.3f}")
    print(f"指派边际(top1-top2): mean={margin.mean():.3f} p10={np.quantile(margin,0.1):.3f}")

    # 省份×簇
    byprov = defaultdict(Counter)
    for r, a in zip(keep, assign):
        byprov[r["province"]][a] += 1
    print("省×簇:", {p: dict(c) for p, c in list(byprov.items())})

    # 3) kMeans 一致性
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    lab_cs = [snames.index(a) for a in assign]
    aris = {}
    for k in range(4, 11):
        km = KMeans(k, n_init=10, random_state=2026).fit_predict(F)
        aris[k] = (round(adjusted_rand_score(lab_cs, km), 3),
                   round(normalized_mutual_info_score(lab_cs, km), 3))
    print("kMeans与CS指派一致性 (k: ARI, NMI):", aris)

    # 4) 覆盖缺口: 低top1图
    tau = float(np.quantile(top1, 0.05))
    uncovered = [{"image_id": r["image_id"], "path": r["path"], "province": r["province"],
                  "top1": round(float(t), 3), "assigned": a}
                 for r, t, a in zip(keep, top1, assign) if t < tau]
    print(f"覆盖缺口候选(top1<{tau:.3f}): {len(uncovered)} 张 -> uncovered.json")
    json.dump([{"image_id": u["image_id"], "province": u["province"], "top1": u["top1"],
                "assigned": u["assigned"]} for u in uncovered],
              open(E6 / "uncovered.json", "w"), ensure_ascii=False, indent=1)

    out = {"n_real": len(keep),
           "cs_assign_dist": dict(dist),
           "top1_sim": {"mean": round(float(top1.mean()), 3), "p10": round(float(np.quantile(top1, 0.1)), 3)},
           "margin": {"mean": round(float(margin.mean()), 3)},
           "province_x_cluster": {p: dict(c) for p, c in byprov.items()},
           "kmeans_agreement": {str(k): v for k, v in aris.items()},
           "uncovered_tau": round(tau, 3), "uncovered_n": len(uncovered)}
    json.dump(out, open(E6 / "style_audit.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "province_x_cluster"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
