#!/usr/bin/env python3
"""
E8-F 选择性预测融合 — 精确率/召回率权衡（设计文档 §7.4）
==========================================================
CLIP 高召回(90%/特异90.7%) + DINOv2 高特异(96.3%) + 合成探针第三通道。
分数级融合 s = w_c·s_clip + w_d·s_dino + w_p·s_probe，权重网格全报告（不事后挑选）。
kNN 分数 = 排除同组后 k=5 正例占比（LOSO）；探针分数 = 合成训练 logreg/MLP。
输出：各配置 P/R/F1 + 最优 F1 阈值 + 风险-覆盖率曲线。探索性，无训练于真实。
"""
import json
import numpy as np
import visual_baseline as vb
import e8_train as e8

CACHE = vb.CACHE


def knn_scores(rows, vecs, k=5):
    """逐样本 kNN 正例占比分数（排除同组）。"""
    V = e8.l2(vecs)
    S = V @ V.T
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        order = np.argsort(-S[i])
        votes = []
        for j in order:
            if j == i or rows[j]["group"] == r["group"]:
                continue
            votes.append(rows[j]["label"] == "positive")
            if len(votes) >= k:
                break
        out[i] = np.mean(votes) if votes else 0.0
    return out


def metrics_at(scores, labels, thr):
    pred = scores > thr
    tp = int(np.sum(pred & labels)); fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(thr=round(float(thr), 3), tp=tp, fp=fp, fn=fn,
                precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3))


def best_f1(scores, labels):
    thrs = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best = max((metrics_at(scores, labels, t) for t in thrs), key=lambda m: m["f1"])
    return best


def risk_coverage(scores, labels, n_pos):
    order = np.argsort(-scores)
    curve = []
    tp = fp = 0
    for i, j in enumerate(order):
        if labels[j]:
            tp += 1
        else:
            fp += 1
        if (i + 1) % 30 == 0 or i == len(order) - 1:
            curve.append(dict(coverage=round((i + 1) / len(order), 3),
                              precision=round(tp / max(tp + fp, 1), 3),
                              recall=round(tp / n_pos, 3)))
    return curve


def main():
    rows = vb.load_registry()
    labels = np.array([r["label"] == "positive" for r in rows])
    n_pos = int(labels.sum())

    # 单模型 kNN 分数（E3 缓存）
    clip_vecs, clip_ids = vb.load_emb("clip", "full")
    dino_vecs, dino_ids = vb.load_emb("dinov2", "crop")
    id2i_c = {iid: i for i, iid in enumerate(clip_ids)}
    id2i_d = {iid: i for i, iid in enumerate(dino_ids)}
    rc = [id2i_c[r["image_id"]] for r in rows]
    rd = [id2i_d[r["image_id"]] for r in rows]
    s_clip = knn_scores(rows, clip_vecs[rc])
    s_dino = knn_scores(rows, dino_vecs[rd])

    # 合成探针分数（E8 缓存嵌入）
    synth_rows = e8.load_synth()
    rf, _ = e8.embed_all(rows, "real")
    sf, _ = e8.embed_all(synth_rows, "synth", use_truth_crop=True)
    probe = e8.train_probes(sf, synth_rows, seed=0)
    X = probe["scaler"].transform(e8.concat_feats(rf))
    s_lin = probe["brand_clf"].predict_proba(X)[:, 1]
    s_mlp = probe["mlp_brand"].predict_proba(X)[:, 1]

    summary = {}
    singles = {"clip_knn": s_clip, "dino_knn": s_dino,
               "synth_linear": s_lin, "synth_mlp": s_mlp,
               "clip_and_dino": s_clip * s_dino,
               "clip_or_dino": np.maximum(s_clip, s_dino)}
    for name, s in singles.items():
        b = best_f1(s, labels)
        summary[name] = dict(best_f1=b, risk_coverage=risk_coverage(s, labels, n_pos))
        print(f"{name:16s} P {b['precision']:.3f} R {b['recall']:.3f} F1 {b['f1']:.3f} @thr {b['thr']}")

    # 权重网格（w_clip, w_dino, w_probe），全报告
    grid = []
    for wc in (0.25, 0.5, 0.75):
        for wd in (0.25, 0.5, 0.75):
            for wp in (0.0, 0.25, 0.5):
                if abs(wc + wd + wp - 1.0) > 1e-6:
                    continue
                s = wc * s_clip + wd * s_dino + wp * s_lin
                b = best_f1(s, labels)
                grid.append(dict(w_clip=wc, w_dino=wd, w_probe=wp, **b))
    grid.sort(key=lambda g: -g["f1"])
    summary["weight_grid_top10"] = grid[:10]
    summary["weight_grid_all"] = grid
    print("\ntop fusion configs:")
    for g in grid[:8]:
        print(f"  wc{g['w_clip']} wd{g['w_dino']} wp{g['w_probe']}: "
              f"P {g['precision']:.3f} R {g['recall']:.3f} F1 {g['f1']:.3f} @thr {g['thr']}")

    json.dump(summary, open(CACHE / "e8f_summary.json", "w"), ensure_ascii=False, indent=1)
    json.dump(dict(experiment="E8-F", version="e8f-fusion-v1.0",
                   governance="exploratory_probe_rights_pending_G0_G1_not_passed",
                   split="store_id_grouped_LOSO_kNN",
                   note="weights grid fully reported; thresholds selected by F1 on exploratory probe only, not deployable claims"),
              open(CACHE / "e8f_manifest.json", "w"), ensure_ascii=False, indent=1)
    print("\n→", CACHE / "e8f_summary.json")


if __name__ == "__main__":
    main()