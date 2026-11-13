#!/usr/bin/env python3
"""E8-G v5 全量重评分 (registry_v5, 344 行) + E10 沿用 v4 329 行重对账。
三通道冻结公式不变: s=0.4*s_clip+0.4*s_dino_mc+0.2*s_probe, thr=0.636 严格>。
画廊=v4 全部正例(组排除)。输出 vis_cache/e8g_v5_results.json。"""
import json, re, sys
import numpy as np
from pathlib import Path

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
            "tp": tp, "fp": fp, "fn": fn, "n_pos": int(labels.sum())}


def bootstrap_ci(scores, labels, groups, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in ug}
    out = {"f1": [], "p": [], "r": []}
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([gidx[g] for g in pick])
        m = metrics(scores[idx], labels[idx])
        out["f1"].append(m["F1"]); out["p"].append(m["P"]); out["r"].append(m["R"])
    def ci(k):
        v = np.array(out[k])
        return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return {"f1_ci95": ci("f1"), "p_ci95": ci("p"), "r_ci95": ci("r")}


def main():
    import os
    reg = json.load(open(ROOT / "data/registry_v5.json"))
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
    print(f"v5 eval rows: {len(rows)}", flush=True)
    labels = np.array([r["label"] == "positive" for r in rows])

    # s_clip: 全库 clip 嵌入(缓存330 + 活体补嵌)
    cv, ci = vb.load_emb("clip", "full")
    id2i = {x: i for i, x in enumerate(ci)}
    missing = [r for r in rows if r["image_id"] not in id2i]
    if missing:
        print(f"live clip embed: {len(missing)}", flush=True)
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

    # s_dino_mc
    s_dino_mc, _ = em.multicrop_dino_scores(rows)
    s_dino_mc = np.clip((s_dino_mc - 0.3) / 0.5, 0, 1)

    # s_probe
    sr = e8.load_synth()
    rf, _ = e8.embed_all(rows, "v4_real")
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    probe = e8.train_probes(sf, sr, seed=0)
    s_probe = probe["brand_clf"].predict_proba(probe["scaler"].transform(e8.concat_feats(rf)))[:, 1]

    s = 0.4 * s_clip + 0.4 * s_dino_mc + 0.2 * s_probe
    m = metrics(s, labels)
    print("E8G v4:", json.dumps(m))
    ci95 = bootstrap_ci(s, labels, np.array([r["group"] for r in rows]))
    print("bootstrap:", json.dumps(ci95))

    # E10 v4: 329 行重对账(0488剔除, 6张truth翻正; VLM裁决沿用历史盲测)
    rec = json.load(open(CACHE / "e10_full/records.json"))
    flip = {"img_0140", "img_0141", "img_0155", "img_0217", "img_0404", "img_0479"}
    rowsv4 = [r for r in rec if r["image_id"] != "img_0488"]
    for r in rowsv4:
        if r["image_id"] in flip:
            r["truth"] = "positive"
    def mode_metrics(key):
        def final(r):
            v = r[key]
            if v == "MATCH": return True
            if v in ("NON_MATCH", "OUT_OF_SCOPE"): return False
            return r["e8g_pred"] in (True, "True")
        pred = np.array([final(r) for r in rowsv4])
        lab = np.array([r["truth"] == "positive" for r in rowsv4])
        tp = int((pred & lab).sum()); fp = int((pred & ~lab).sum()); fn = int((~pred & lab).sum())
        p = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        return {"P": round(p, 3), "R": round(rc, 3), "F1": round(2 * p * rc / max(1e-9, p + rc), 3),
                "tp": tp, "fp": fp, "fn": fn, "n_pos": int(lab.sum())}
    vlm_alone = mode_metrics("vlm_verdict")
    lab = np.array([r["truth"] == "positive" for r in rowsv4])
    # Mode A(记录公式): final = e8g_pred ? verdict not in {NON_MATCH,OUT_OF_SCOPE} : False
    def final_a(r):
        ep = r["e8g_pred"] in (True, "True")
        return ep and r["vlm_verdict"] not in ("NON_MATCH", "OUT_OF_SCOPE")
    # Mode B: MATCH→True; NON_MATCH/OOS→False; else→e8g_pred
    def final_b(r):
        v = r["vlm_verdict"]
        if v == "MATCH": return True
        if v in ("NON_MATCH", "OUT_OF_SCOPE"): return False
        return r["e8g_pred"] in (True, "True")
    def stat(pred):
        tp = int((pred & lab).sum()); fp = int((pred & ~lab).sum()); fn = int((~pred & lab).sum())
        p = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        return {"P": round(p, 3), "R": round(rc, 3), "F1": round(2 * p * rc / max(1e-9, p + rc), 3),
                "tp": tp, "fp": fp, "fn": fn}
    mode_a = stat(np.array([final_a(r) for r in rowsv4]))
    mode_b = stat(np.array([final_b(r) for r in rowsv4]))

    # 逐行分数落盘
    np.savez_compressed(CACHE / "e8g_v5_scores.npz", scores=s, labels=labels,
                        groups=np.array([r["group"] for r in rows]),
                        ids=np.array([r["image_id"] for r in rows]))
    # 嵌套 LOSO 阈值重选(每折留一组, 其余组上扫 thr, 应用到留出组)
    groups = np.array([r["group"] for r in rows])
    ug = np.unique([g for g in groups if labels[groups == g].any()])
    thr_grid = np.unique(np.round(s, 4))
    nested_pred = np.zeros(len(rows), dtype=bool)
    for g in ug:
        te = groups == g
        tr = ~te
        best = (-1, 0.636)
        for t in thr_grid:
            pred_tr = s[tr] > t
            tp = int((pred_tr & labels[tr]).sum()); fp = int((pred_tr & ~labels[tr]).sum())
            fn = int((~pred_tr & labels[tr]).sum())
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            if f1 > best[0]: best = (f1, float(t))
        nested_pred[te] = s[te] > best[1]
    tp = int((nested_pred & labels).sum()); fp = int((nested_pred & ~labels).sum()); fn = int((~nested_pred & labels).sum())
    p_n = tp / max(1, tp + fp); r_n = tp / max(1, tp + fn)
    nested = {"P": round(p_n, 3), "R": round(r_n, 3), "F1": round(2 * p_n * r_n / max(1e-9, p_n + r_n), 3),
              "tp": tp, "fp": fp, "fn": fn}
    print("nested:", json.dumps(nested))

    out = {
        "date": "2026-11-13", "registry": "v5", "n": len(rows),
        "e8g_v4": {**m, "ci95": ci95, "thr_frozen_v3": THR, "nested_loso": nested},
        "e10_v4": {"rows": len(rowsv4), "vlm_alone": vlm_alone, "mode_a": mode_a, "mode_b": mode_b},
        "flips_in_eval": sorted(flip), "dropped": "img_0488(占位图)",
    }
    json.dump(out, open(CACHE / "e8g_v5_results.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:1500])


if __name__ == "__main__":
    main()
