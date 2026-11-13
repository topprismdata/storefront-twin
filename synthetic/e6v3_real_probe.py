#!/usr/bin/env python3
"""E6-v3 — 真实图探针微调 (授权 GD-2026-11-14-01: 研究训练放行)。
唯一变量: s_probe 从"仅合成图训练"换成"合成+真实图训练"(店组 LOSO, 评测图所在店组绝不进训练集)。
s_clip / s_dino_mc / 融合权重 / 阈值协议全部冻结, 与 E8G v11 基线逐项可比。
输出 vis_cache/e6v3_results.json。"""
import json, sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import visual_baseline as vb
import e8_train as e8
import e8_fusion as ef
import e8_multicrop as em

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
CACHE = vb.CACHE
THR = 0.636


def metrics(scores, labels):
    pred = scores > THR
    tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum()); fn = int((~pred & labels).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"P": round(p, 3), "R": round(r, 3), "F1": round(2 * p * r / max(1e-9, p + r), 3),
            "tp": tp, "fp": fp, "fn": fn}


def nested_metrics(scores, labels, groups, hard_types):
    """每折: 阈值在训练店组上选(最大F1), 应用到留出店组, 聚合混淆矩阵。"""
    ug = np.unique(groups)
    cands = np.unique(np.round(scores, 3))
    tp = fp = fn = 0
    fps = {}
    for g in ug:
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        best, thr = -1, THR
        for t in cands:
            pred = scores[tr] > t
            a = int((pred & labels[tr]).sum()); b = int((pred & ~labels[tr]).sum()); c = int((~pred & labels[tr]).sum())
            f1 = 2 * a / max(1, 2 * a + b + c)
            if f1 > best:
                best, thr = f1, t
        pred = scores[te] > thr
        tp += int((pred & labels[te]).sum()); fp += int((pred & ~labels[te]).sum()); fn += int((~pred & labels[te]).sum())
        for j, i in enumerate(te):
            if pred[j] and not labels[i]:
                k = hard_types[i]
                fps[k] = fps.get(k, 0) + 1
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"P": round(p, 3), "R": round(r, 3), "F1": round(2 * p * r / max(1e-9, p + r), 3),
            "tp": tp, "fp": fp, "fn": fn, "fp_by_type": fps}


def main():
    reg = json.load(open(ROOT / "data/registry_v11.json"))
    sha_idx = json.load(open(ROOT / "data/sha_file_index.json"))
    rows = []
    for r in reg:
        if not r.get("is_representative"):
            continue
        p = sha_idx.get(r["sha256"])
        if not p:
            continue
        fp = ROOT / "data" / p
        if not fp.exists():
            continue
        rows.append({"image_id": r["image_id"], "path": str(fp), "label": r["label"],
                     "neg_type": r.get("neg_type", ""), "group": r.get("store_id") or r["image_id"],
                     "candidate_style": (r.get("candidate_style") or "").split("_")[0]})
    labels = np.array([r["label"] == "positive" for r in rows])
    groups = np.array([r["group"] for r in rows])
    hard = np.array([r["neg_type"] if (r["label"] != "positive" and r["neg_type"] in
                     ("confusable_brand", "same_name_unrelated", "competitor", "render_stock", "logo", "indoor", "uncertain")) else "-"
                     for r in rows])
    print(f"rows: {len(rows)} pos={labels.sum()}", flush=True)

    # 特征 (缓存命中: rows 构造与 e8g_v11_rescore 完全一致)
    rf, _ = e8.embed_all(rows, "v4_real")
    X_real = e8.concat_feats(rf)
    sr = e8.load_synth()
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    X_syn = e8.concat_feats(sf)
    y_syn = np.array([1 if r["label"] == "positive" else 0 for r in sr])
    print(f"synth: {X_syn.shape}, real: {X_real.shape}", flush=True)

    # 两通道冻结
    cv, ci = vb.load_emb("clip", "full")
    id2i = {x: i for i, x in enumerate(ci)}
    missing = [r for r in rows if r["image_id"] not in id2i]
    if missing:
        emb = vb.Embedder("clip")
        import cv2
        from PIL import Image
        for r in missing:
            bgr = cv2.imread(r["path"])
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            v = emb.embed_pil([pil])[0]
            cv = np.vstack([cv, v[None, :]])
            ci.append(r["image_id"])
            id2i[r["image_id"]] = len(ci) - 1
    s_clip = ef.knn_scores(rows, cv[[id2i[r["image_id"]] for r in rows]])
    s_dino_mc, _ = em.multicrop_dino_scores(rows)
    s_dino_mc = np.clip((s_dino_mc - 0.3) / 0.5, 0, 1)

    # 基线探针(仅合成) — 校验与 e8g_v11_scores 一致
    probe0 = e8.train_probes(sf, sr, seed=0)
    s_probe0 = probe0["brand_clf"].predict_proba(probe0["scaler"].transform(X_real))[:, 1]
    s_base = 0.4 * s_clip + 0.4 * s_dino_mc + 0.2 * s_probe0
    try:
        old = np.load(CACHE / "e8g_v11_scores.npz", allow_pickle=True)
        om = {x: float(s) for x, s in zip(list(old["ids"]), list(old["scores"]))}
        dev = max(abs(s_base[i] - om[r["image_id"]]) for i, r in enumerate(rows) if r["image_id"] in om)
        print(f"基线校验 max|Δs| = {dev:.4f}", flush=True)
    except Exception as e:
        print("skip verify:", e, flush=True)

    # LOSO 真实+合成探针
    ug = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in ug}
    s_probe_r = np.zeros(len(rows))
    Xall_syn_s, sc_syn = StandardScaler(), None
    for fi, g in enumerate(ug):
        te = gidx[g]
        tr = np.concatenate([gidx[h] for h in ug if h != g])
        X_tr = np.vstack([X_syn, X_real[tr]])
        y_tr = np.concatenate([y_syn, labels[tr]])
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
        clf.fit(sc.transform(X_tr), y_tr)
        s_probe_r[te] = clf.predict_proba(sc.transform(X_real[te]))[:, 1]
        if fi % 60 == 0:
            print(f"loso {fi}/{len(ug)}", flush=True)
    s_real = 0.4 * s_clip + 0.4 * s_dino_mc + 0.2 * s_probe_r

    out = {"decision": "GD-2026-11-14-01", "rows": len(rows), "n_pos": int(labels.sum()), "n_groups": len(ug),
           "frozen_0636": {"base_syn_only": metrics(s_base, labels), "e6v3_real_loso": metrics(s_real, labels)},
           "nested_loso": {"base_syn_only": nested_metrics(s_base, labels, groups, hard),
                           "e6v3_real_loso": nested_metrics(s_real, labels, groups, hard)}}
    json.dump(out, open(CACHE / "e6v3_results.json", "w"), ensure_ascii=False, indent=1)
    np.savez(CACHE / "e6v3_scores.npz", ids=np.array([r["image_id"] for r in rows]),
             s_base=s_base, s_real=s_real, labels=labels, groups=groups)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
