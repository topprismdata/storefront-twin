#!/usr/bin/env python3
"""
E6-E10 离线端到端模拟：E6-v2 作为第二召回头加入 E10 回退层的收益
================================================================
不烧 API：复用 E10 v1 全盲 330 裁决 + E6-v2 冻结分数，离线重算两种回退策略：
  base-e8g   现役：回退 = e8g_pred
  base-or    候选：回退 = OR(e8g_pred, e6v2_pred)（E6 第二召回头）
Mode A(拒绝式)/Mode B(救援式) × 两种回退 = 2×2 网格 + 门店级 Bootstrap。
"""
import json, sys
import numpy as np
from collections import Counter

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
CACHE = None
import e6_real_probe as e1
CACHE = e1.CACHE

rows = json.load(open(CACHE / "e10_full/records.json"))
z = np.load(CACHE / "e6/e6v2_probe.npz")
w, b, thr = z["coef"][0].astype(np.float64), float(z["intercept"]), float(z["thr"])
m = {}
for n in ("clip", "dinov2"):
    for v in ("full", "crop"):
        zz = np.load(CACHE / f"{n}_{v}.npz", allow_pickle=True)
        m[(n, v)] = dict(zip(zz["ids"].tolist(), zz["vecs"]))
X = np.concatenate([np.stack([m[(n, v)][r["image_id"]] for r in rows])
                    for n in ("clip", "dinov2") for v in ("full", "crop")], 1).astype(np.float64)
s6 = X @ w + b
p6 = s6 > thr
e8 = np.array([r["e8g_pred"] for r in rows], bool)
y = np.array([r["truth"] == "positive" for r in rows])
vd = [r["vlm_verdict"] for r in rows]
groups = json.load(open(CACHE / "e10_full/truth_key.json"))
groups = groups if isinstance(groups, list) else list(groups.values())
groups = [g["group"] for g in groups]


def metrics(pred):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    return P, R, 2 * P * R / max(1e-9, P + R), (tp, fp, fn)


def boot(pred):
    rng = np.random.default_rng(2026)
    ug = np.unique(groups); out = []
    for _ in range(2000):
        sg = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([np.where(np.array(groups) == g)[0] for g in sg])
        yy, pp = y[idx], pred[idx]
        t = int((pp & (yy == 1)).sum()); f = int((pp & (yy == 0)).sum()); n2 = int((~pp & (yy == 1)).sum())
        out.append(2 * t / max(1, 2 * t + f + n2))
    return [round(float(x), 3) for x in np.quantile(out, [0.025, 0.975])]


base = np.array(e8)
second = e8 | p6
NA = base & np.array([v not in ("NON_MATCH", "OUT_OF_SCOPE") for v in vd])
OA = second & np.array([v not in ("NON_MATCH", "OUT_OF_SCOPE") for v in vd])
NB = np.array([v == "MATCH" for v in vd]) | (~np.array([v in ("MATCH", "NON_MATCH", "OUT_OF_SCOPE") for v in vd]) & base)
OB = np.array([v == "MATCH" for v in vd]) | (~np.array([v in ("MATCH", "NON_MATCH", "OUT_OF_SCOPE") for v in vd]) & second)

out = {}
for name, pred in [("ModeA_base_e8g", NA), ("ModeA_or_e6", OA), ("ModeB_base_e8g", NB), ("ModeB_or_e6", OB)]:
    P, R, F1, cf = metrics(pred)
    out[name] = {"P": round(P, 3), "R": round(R, 3), "F1": round(F1, 3), "confusion": cf,
                 "CI95": boot(pred) if name.endswith("or_e6") else None}
# 新增FP来自哪
new_fp = OA & ~NA & ~y
print("Mode A 新增FP的neg_type:", Counter(r["neg_type"] for r, f in zip(rows, new_fp) if f))
new_fpB = OB & ~NB & ~y
print("Mode B 新增FP的neg_type:", Counter(r["neg_type"] for r, f in zip(rows, new_fpB) if f))
saved = (~OA & NA & ~y)
print("Mode A 被E6撤掉的FP:", [r["image_id"] for r, s in zip(rows, saved) if s])
json.dump(out, open(CACHE / "e6/e10_fallback_sim.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
