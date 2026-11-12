#!/usr/bin/env python3
"""
GEN — 独立泛化集 E8-G 冻结评测（首次真泛化数字）
=================================================
223 图(110正/113负,148组) sha 级独立于 330/池/金标注。
E8-G 三通道完全同构复制 e8g_stats: s=0.4*s_clip+0.4*s_dino_mc+0.2*s_probe, thr=0.636 严格>。
kNN/多裁剪参考集均在 gen 集内部(排除同店组),零泄漏。
"""
import json, sys
import numpy as np

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import visual_baseline as vb
import e8_train as e8
import e8_fusion as ef
import e8_multicrop as em
from pathlib import Path

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
CACHE = vb.CACHE


def metrics(scores, labels, thr):
    pred = scores > thr
    tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum()); fn = int((~pred & labels).sum())
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    return {"P": round(P, 3), "R": round(R, 3), "F1": round(2 * P * R / max(1e-9, P + R), 3),
            "confusion": [tp, fp, fn]}


def main():
    reg = json.load(open(ROOT / "data/registry_gen.json"))
    rows = [{"image_id": r["image_id"],
             "path": str(ROOT / r["path"]) if not r["path"].startswith("/") else r["path"],
             "label": r["label"], "group": r["group"]} for r in reg["rows"]]
    labels = np.array([r["label"] == "positive" for r in rows])
    print(f"gen rows: {len(rows)} pos={labels.sum()} groups={len({r['group'] for r in rows})}", flush=True)

    # 1) s_clip: gen 内部 kNN
    emb = vb.Embedder("clip")
    import cv2
    from PIL import Image
    vecs = []
    for i, r in enumerate(rows):
        bgr = cv2.imread(r["path"])
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        vecs.append(emb.embed_pil([pil])[0])
        if (i + 1) % 100 == 0:
            print(f"  clip {i+1}/{len(rows)}", flush=True)
    V = np.stack(vecs)
    s_clip = ef.knn_scores(rows, V)
    del emb

    # 2) s_dino_mc
    s_dino_mc, _ = em.multicrop_dino_scores(rows)
    s_dino_mc = np.clip((s_dino_mc - 0.3) / 0.5, 0, 1)

    # 3) s_probe: synth 训练(种子0确定性) -> gen 实图
    sr = e8.load_synth()
    rf, _ = e8.embed_all(rows, "gen_real")
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    probe = e8.train_probes(sf, sr, seed=0)
    s_probe = probe["brand_clf"].predict_proba(probe["scaler"].transform(e8.concat_feats(rf)))[:, 1]

    s = 0.4 * s_clip + 0.4 * s_dino_mc + 0.2 * s_probe
    m = metrics(s, labels, 0.636)
    print("E8-G on GEN @0.636:", m, flush=True)

    np.savez_compressed(CACHE / "gen_e8g_scores.npz", scores=s, labels=labels,
                        ids=np.array([r["image_id"] for r in rows]),
                        groups=np.array([r["group"] for r in rows]),
                        s_clip=s_clip, s_dino_mc=s_dino_mc, s_probe=s_probe)
    json.dump({"thr": 0.636, "metrics": m, "n": len(rows),
               "note": "first true generalization estimate; 322-group methodology mirrored"},
              open(CACHE / "gen_e8g_results.json", "w"), indent=1)
    print("saved gen_e8g_scores.npz / gen_e8g_results.json", flush=True)


if __name__ == "__main__":
    main()
