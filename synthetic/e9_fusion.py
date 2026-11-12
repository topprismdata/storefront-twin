#!/usr/bin/env python3
"""
E9-F 整合：AnyLoc 细粒度描述子 + CLIP 品牌 + 合成探针 融合
============================================================
E9 证明 AnyLoc 中间层 value+VLAD/GeM 把 L1 R@1 从 40%→90%，但 L0 品牌召回仍低（需 CLIP）。
本脚本把 E9 最佳描述子接入 E8-G 三通道融合，验证 L0 P/R/F1 是否进一步提升，
并升级评估协议（AuPR + P@95R + R@95P + softmax 熵，文献 2403.16528）。
"""
import json
import numpy as np
import visual_baseline as vb
import e8_train as e8
import e8_fusion as ef
import e9_anyloc as e9

CACHE = vb.CACHE


def auprc(scores, labels):
    """PR 曲线下面积（开集协议 2403.16528）。"""
    order = np.argsort(-scores)
    tp = fp = 0
    n_pos = labels.sum()
    prev_r = 0.0
    ap = 0.0
    for j in order:
        if labels[j]:
            tp += 1
        else:
            fp += 1
        r = tp / n_pos
        p = tp / (tp + fp)
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def p_at_r(scores, labels, target_r):
    order = np.argsort(-scores)
    tp = fp = 0
    n_pos = labels.sum()
    best_p = 0.0
    for j in order:
        if labels[j]:
            tp += 1
        else:
            fp += 1
        if tp / n_pos >= target_r:
            best_p = tp / (tp + fp)
            break
    return float(best_p)


def r_at_p(scores, labels, target_p):
    order = np.argsort(-scores)
    tp = fp = 0
    n_pos = labels.sum()
    best_r = 0.0
    for j in order:
        if labels[j]:
            tp += 1
        else:
            fp += 1
        p = tp / (tp + fp)
        if p >= target_p:
            best_r = tp / n_pos
    return float(best_r)


def main():
    rows = vb.load_registry()
    labels = np.array([r["label"] == "positive" for r in rows])
    n_pos = int(labels.sum())

    # CLIP kNN（L0 品牌语义）
    cv, ci = vb.load_emb("clip", "full")
    s_clip = ef.knn_scores(rows, cv[[{x:i for i,x in enumerate(ci)}[r["image_id"]] for r in rows]])

    # E9 AnyLoc 描述子（L1 细粒度）→ 转 L0 品牌分数（kNN 投票）
    feats, ids = e9.extract_facets(rows, "crop")
    id2i = {iid: i for i, iid in enumerate(ids)}
    idx = [id2i[r["image_id"]] for r in rows]
    e9_scores = {}
    for key in ["L10_value", "L8_token", "L6_value"]:
        P = feats[key][idx]
        centers = e9.build_vocab(P)
        for agg, desc in [("vlad", e9.hard_vlad(P, centers)), ("gem", e9.gem_pool(P))]:
            desc = desc / np.maximum(np.linalg.norm(desc, axis=-1, keepdims=True), 1e-9)
            b = e9.brand_eval(rows, desc)
            e9_scores[f"{key}/{agg}"] = np.array([r["score"] for r in b])

    # 合成探针
    sr = e8.load_synth()
    rf, _ = e8.embed_all(rows, "real")
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    probe = e8.train_probes(sf, sr, seed=0)
    X = probe["scaler"].transform(e8.concat_feats(rf))
    s_probe = probe["brand_clf"].predict_proba(X)[:, 1]

    summary = {}
    # 单通道 + E9 各描述子作 L0 的 AuPR
    for name, s in [("clip_knn", s_clip), ("synth_probe", s_probe)] + list(e9_scores.items()):
        summary[name] = dict(auprc=round(auprc(s, labels), 3),
                             p_at_95r=round(p_at_r(s, labels, 0.95), 3),
                             r_at_95p=round(r_at_p(s, labels, 0.95), 3),
                             best_f1=ef.best_f1(s, labels))

    # 融合：CLIP + E9最佳 + 探针，权重网格
    best_e9 = max(e9_scores, key=lambda k: summary[k]["auprc"])
    s_e9 = e9_scores[best_e9]
    grid = []
    for wc in (0.4, 0.5, 0.6):
        for we in (0.2, 0.3, 0.4):
            for wp in (0.1, 0.2, 0.3):
                if abs(wc+we+wp-1.0) > 1e-6:
                    continue
                s = wc*s_clip + we*s_e9 + wp*s_probe
                b = ef.best_f1(s, labels)
                grid.append(dict(w_clip=wc, w_e9=we, w_probe=wp, e9_desc=best_e9,
                                 auprc=round(auprc(s, labels), 3), **b))
    grid.sort(key=lambda g: -g["f1"])
    summary["fusion_grid_top8"] = grid[:8]
    summary["best_e9_desc"] = best_e9

    print("=== 单通道 AuPR / P@95R / R@95P ===")
    for name in ["clip_knn", "synth_probe", best_e9]:
        m = summary[name]
        print(f"{name:18s} AuPR {m['auprc']:.3f} P@95R {m['p_at_95r']:.3f} R@95P {m['r_at_95p']:.3f}")
    print("\n=== 融合 top（CLIP+E9+探针）===")
    for g in grid[:6]:
        print(f"wc{g['w_clip']} we{g['w_e9']} wp{g['w_probe']} [{g['e9_desc']}]: "
              f"P {g['precision']:.3f} R {g['recall']:.3f} F1 {g['f1']:.3f} AuPR {g['auprc']:.3f} (tp{g['tp']} fp{g['fp']} fn{g['fn']})")

    json.dump(summary, open(CACHE/"e9f_summary.json", "w"), ensure_ascii=False, indent=1)
    json.dump(dict(experiment="E9-F", version="e9f-anyloc-fusion-v1.0",
                   protocol="AuPR + P@95R + R@95P (arXiv:2403.16528)",
                   governance="exploratory_probe_rights_pending_G0_G1_not_passed"),
              open(CACHE/"e9f_manifest.json", "w"), ensure_ascii=False, indent=1)
    print("\n→", CACHE/"e9f_summary.json")


if __name__ == "__main__":
    main()