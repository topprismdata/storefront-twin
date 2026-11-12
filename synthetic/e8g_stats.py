#!/usr/bin/env python3
"""E8-G 统计可靠性（GPT 裁决 D / P3）：
1) 门店级 Bootstrap（按 store_id 整簇重采样，2000 次）→ P/R/F1/AuPR 置信区间
2) 嵌套 LOSO 阈值隔离：每折在其余门店上选阈值，应用到留出门店 → 消除阈值选择偏差
产物 vis_cache/e8g_stats.json。冻结 E8 分数，不重训。"""
import json
import numpy as np
import visual_baseline as vb
import e8_train as e8
import e8_fusion as ef
import e8_multicrop as em
import e9_fusion as e9f

CACHE = vb.CACHE
rng = np.random.default_rng(42)


def metrics(scores, labels, thr):
    pred = scores > thr
    tp = int(np.sum(pred & labels)); fp = int(np.sum(pred & ~labels)); fn = int(np.sum(~pred & labels))
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1, tp, fp, fn


def main():
    rows = vb.load_registry()
    labels = np.array([r["label"] == "positive" for r in rows])
    groups = np.array([r["group"] for r in rows])

    cv, ci = vb.load_emb("clip", "full")
    id2i = {x: i for i, x in enumerate(ci)}
    s_clip = ef.knn_scores(rows, cv[[id2i[r["image_id"]] for r in rows]])
    s_dino_mc, _ = em.multicrop_dino_scores(rows)
    s_dino_mc = np.clip((s_dino_mc - 0.3) / 0.5, 0, 1)
    sr = e8.load_synth()
    rf, _ = e8.embed_all(rows, "real")
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    probe = e8.train_probes(sf, sr, seed=0)
    s_probe = probe["brand_clf"].predict_proba(probe["scaler"].transform(e8.concat_feats(rf)))[:, 1]
    s = 0.4 * s_clip + 0.4 * s_dino_mc + 0.2 * s_probe

    # 冻结分数落盘
    np.savez_compressed(CACHE / "e8g_frozen_scores.npz",
                        scores=s, labels=labels, groups=groups,
                        ids=np.array([r["image_id"] for r in rows]))

    uniq_groups = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq_groups}
    n_pos_groups = sum(1 for g in uniq_groups if labels[gidx[g]].any())

    # 1) 门店级 Bootstrap（固定阈值 0.636，重采样门店）
    thr_fixed = 0.636
    boot = {k: [] for k in ["p", "r", "f1", "auprc"]}
    B = 2000
    for b in range(B):
        sampled = rng.choice(uniq_groups, size=len(uniq_groups), replace=True)
        idx = np.concatenate([gidx[g] for g in sampled])
        sc, lb = s[idx], labels[idx]
        if lb.sum() == 0 or (~lb).sum() == 0:
            continue
        p, r, f1, *_ = metrics(sc, lb, thr_fixed)
        boot["p"].append(p); boot["r"].append(r); boot["f1"].append(f1)
        boot["auprc"].append(e9f.auprc(sc, lb))
    ci = {}
    for k, v in boot.items():
        v = np.array(v)
        ci[k] = dict(mean=round(float(v.mean()), 3),
                     lo=round(float(np.percentile(v, 2.5)), 3),
                     hi=round(float(np.percentile(v, 97.5)), 3))

    # 2) 嵌套 LOSO 阈值隔离：每折留出 1 门店，在其余门店选最优阈值，应用到留出店
    nested_pred = np.zeros(len(rows), dtype=bool)
    nested_thr = []
    for g in uniq_groups:
        te = gidx[g]; tr = np.where(groups != g)[0]
        if labels[tr].sum() == 0:
            continue
        # 在训练门店上选阈值（最大化 F1）
        thrs = np.unique(np.concatenate([s[tr], [0.0, 1.0]]))
        best_t, best_f1 = 0.5, -1
        for t in thrs:
            _, _, f1, *_ = metrics(s[tr], labels[tr], t)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        nested_thr.append(float(best_t))
        nested_pred[te] = s[te] > best_t
    tp = int(np.sum(nested_pred & labels)); fp = int(np.sum(nested_pred & ~labels)); fn = int(np.sum(~nested_pred & labels))
    np_p = tp / max(tp + fp, 1); np_r = tp / max(tp + fn, 1)
    nested = dict(precision=round(np_p, 3), recall=round(np_r, 3),
                  f1=round(2 * np_p * np_r / max(np_p + np_r, 1e-9), 3),
                  tp=tp, fp=fp, fn=fn,
                  thr_mean=round(float(np.mean(nested_thr)), 3),
                  thr_std=round(float(np.std(nested_thr)), 3))

    out = dict(n_groups=len(uniq_groups), n_pos_groups=n_pos_groups,
               n_rows=len(rows), n_pos=int(labels.sum()),
               bootstrap_B=B, bootstrap_ci=ci, nested_loso=nested,
               fixed_thr=thr_fixed,
               note="门店级Bootstrap固定阈值;嵌套LOSO每折重选阈值消除选择偏差;正例门店少→区间偏宽")
    json.dump(out, open(CACHE / "e8g_stats.json", "w"), ensure_ascii=False, indent=1)
    print("groups:", len(uniq_groups), "pos_groups:", n_pos_groups)
    print("Bootstrap CI (store-level, B=2000):")
    for k, v in ci.items():
        print(f"  {k}: {v['mean']} [{v['lo']}, {v['hi']}]")
    print("Nested LOSO (threshold isolated):", nested)


if __name__ == "__main__":
    main()