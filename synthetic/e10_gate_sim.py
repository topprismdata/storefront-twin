#!/usr/bin/env python3
"""E10 灰区门控成本曲线（离线模拟，零 VLM 调用）。
用冻结 E8-G 全精度分数 + 已冻结 VLM 裁决，模拟"只对灰区调 VLM"的成本-精度权衡。
门控外：s<a 直接拒、s>b 直接受（模式A）；模式B 门控外回退 E8-G 原预测。
产物 vis_cache/e10_full/gate_cost_curve.json。
注意：同一 330 探索性模拟，带宽在本数据上选取→有过拟合风险，须 v2/新门店复核。"""
import json
import numpy as np

ROOT = "/Users/mac/chatgpt-harvest/6af41562/synthetic"
V = f"{ROOT}/vis_cache/e10_full"

z = np.load(f"{ROOT}/vis_cache/e8g_frozen_scores.npz", allow_pickle=True)
id2score = {str(i): float(s) for i, s in zip(z["ids"], z["scores"])}
scores_all = z["scores"]; labels_all = z["labels"]

# 真实（未舍入）最优阈值：在冻结分数上重选 best-F1
thrs = np.unique(np.concatenate([scores_all, [0.0, 1.0]]))
best_f1, THR = -1, 0.5
for t in thrs:
    pred = scores_all > t
    tp = int((pred & labels_all).sum()); fp = int((pred & ~labels_all).sum()); fn = int((~pred & labels_all).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f1 = 2 * p * r / max(p + r, 1e-9)
    if f1 > best_f1:
        best_f1, THR = f1, float(t)

recs = json.load(open(f"{V}/records.json"))
rows = []
for r in recs:
    rows.append(dict(id=r["image_id"], s=id2score[r["image_id"]],
                     truth=r["truth"] == "positive", v=r["vlm_verdict"],
                     e8=bool(id2score[r["image_id"]] > THR)))

REJ = {"NON_MATCH", "OUT_OF_SCOPE"}


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return round(p, 3), round(r, 3), round(2 * p * r / max(p + r, 1e-9), 3)


def simulate(mode, lo, hi):
    """lo/hi 为相对 THR 的带宽。门控区间 [THR-lo, THR+hi]。"""
    a, b = THR - lo, THR + hi
    tp = fp = fn = calls = 0
    for r in rows:
        in_band = a <= r["s"] <= b
        if mode == "A":
            # 门控外：低于带拒、高于带受；带内用 VLM 拒识
            if r["s"] < a:
                f = False
            elif r["s"] > b:
                f = True
            else:
                calls += 1
                f = r["e8"] and r["v"] not in REJ
        else:  # B
            if in_band:
                calls += 1
                f = True if r["v"] == "MATCH" else (False if r["v"] in REJ else r["e8"])
            else:
                f = r["e8"]
        if r["truth"] and f: tp += 1
        elif r["truth"]: fn += 1
        elif f: fp += 1
    p, rc, f1 = prf(tp, fp, fn)
    return dict(mode=mode, lo=round(lo, 3), hi=round(hi, 3), calls=calls,
                call_pct=round(100 * calls / len(rows), 1),
                tp=tp, fp=fp, fn=fn, precision=p, recall=rc, f1=f1)


out = {"thr_unrounded": round(THR, 6), "n": len(rows), "curves": []}
# 模式A：pos_only（s>THR 全调，无损）+ 上带宽收窄
out["curves"].append(simulate("A", 0.0, 1.0))  # 全量参考
out["curves"].append({**simulate("A", 0.0, 0.40), "note": "pos_only gate (s>thr all called)"})
for hi in [0.30, 0.20, 0.15, 0.10, 0.05, 0.02]:
    out["curves"].append(simulate("A", 0.0, hi))
# 模式B：对称带宽扫描
for w in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
    out["curves"].append(simulate("B", w, w))

# 关键样本分数（解释曲线形状）
key_ids = ["img_0012", "img_0487", "img_0038", "img_0483", "img_0101",
           "img_0227", "img_0308", "img_0022", "img_0345"]
out["key_scores"] = {i: round(id2score[i], 4) for i in key_ids if i in id2score}
fp_scores = sorted([r["s"] for r in rows if not r["truth"] and r["e8"]], reverse=True)
tp_scores = sorted([r["s"] for r in rows if r["truth"] and r["e8"]], reverse=True)
out["fp_scores_desc"] = [round(x, 4) for x in fp_scores]
out["tp_scores_desc"] = [round(x, 4) for x in tp_scores]

json.dump(out, open(f"{V}/gate_cost_curve.json", "w"), ensure_ascii=False, indent=1)
print(f"THR(unrounded)={THR:.6f}  best_F1={best_f1:.3f}")
print(f"{'mode':4} {'lo':>5} {'hi':>5} {'calls':>5} {'%':>5} {'TP':>3} {'FP':>3} {'FN':>3} {'P':>6} {'R':>6} {'F1':>6}")
for c in out["curves"]:
    print(f"{c['mode']:4} {c['lo']:5.2f} {c['hi']:5.2f} {c['calls']:5d} {c['call_pct']:5.1f} "
          f"{c['tp']:3d} {c['fp']:3d} {c['fn']:3d} {c['precision']:6.3f} {c['recall']:6.3f} {c['f1']:6.3f}")
print("FP scores desc:", out["fp_scores_desc"])
print("TP scores desc:", out["tp_scores_desc"][:8], "...")
print("key:", out["key_scores"])