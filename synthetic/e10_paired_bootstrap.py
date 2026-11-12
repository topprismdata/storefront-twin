#!/usr/bin/env python3
"""配对门店级 Bootstrap ΔF1（GPT 评审 C1）。
同一门店重采样集合上同时算两系统，报告 ΔF1 点估计+95%CI+是否跨0。
对 v1 与 v2 分别做 B-vs-A、B-vs-VLM、A-vs-VLM。固定 seed=42，B=2000。
产物 v1_paired_bootstrap.json / v2_paired_bootstrap.json。"""
import json, sys
import numpy as np

ROOT = "/Users/mac/chatgpt-harvest/6af41562/synthetic/vis_cache/e10_full"
version = sys.argv[1] if len(sys.argv) > 1 else "v1"
recs = json.load(open(f"{ROOT}/{'records' if version=='v1' else 'v2_records'}.json"))
key = json.load(open(f"{ROOT}/truth_key.json"))
anon_by_id = {k["image_id"]: a for a, k in key.items()}
for r in recs:
    r["group"] = key[anon_by_id[r["image_id"]]]["group"]

rng = np.random.default_rng(42)
groups = np.array([r["group"] for r in recs])
truth = np.array([r["truth"] == "positive" for r in recs])
e8 = np.array([r["e8g_pred"] for r in recs])
verd = np.array([r["vlm_verdict"] for r in recs])
uniq = np.unique(groups)
gidx = {g: np.where(groups == g)[0] for g in uniq}
REJ = ["NON_MATCH", "OUT_OF_SCOPE"]


def preds(mode, idx):
    e = e8[idx]; v = verd[idx]
    if mode == "VLM":
        return v == "MATCH"
    if mode == "A":
        return np.where(e, ~np.isin(v, REJ), False)
    out = e.copy()
    out[v == "MATCH"] = True
    out[np.isin(v, REJ)] = False
    return out


def prf(pred, t):
    tp = int(np.sum(pred & t)); fp = int(np.sum(pred & ~t)); fn = int(np.sum(~pred & t))
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


B = 2000
pairs = [("B", "A"), ("B", "VLM"), ("A", "VLM")]
boot = {f"{x}-{y}": [] for x, y in pairs}
for b in range(B):
    sampled = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([gidx[g] for g in sampled])
    t = truth[idx]
    if t.sum() == 0 or (~t).sum() == 0:
        continue
    f1s = {m: prf(preds(m, idx), t)[2] for m in ["A", "B", "VLM"]}
    for x, y in pairs:
        boot[f"{x}-{y}"].append(f1s[x] - f1s[y])

out = {"version": version, "n_groups": len(uniq), "n_bootstrap": B, "seed": 42, "pairs": {}}
for name, vals in boot.items():
    v = np.array(vals)
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    x, y = name.split("-")
    point = prf(preds(x, np.arange(len(recs))), truth)[2] - prf(preds(y, np.arange(len(recs))), truth)[2]
    out["pairs"][name] = dict(delta_f1=round(point, 3), mean=round(float(v.mean()), 3),
                              lo=round(lo, 3), hi=round(hi, 3), crosses_zero=bool(lo <= 0 <= hi),
                              significant=bool(lo > 0 or hi < 0))
json.dump(out, open(f"{ROOT}/{version}_paired_bootstrap.json", "w"), ensure_ascii=False, indent=1)
print(f"=== {version} 配对门店 Bootstrap ΔF1 (B=2000, seed=42) ===")
for name, d in out["pairs"].items():
    sig = "显著" if d["significant"] else "不显著(跨0)"
    print(f"  {name}: ΔF1={d['delta_f1']:+.3f} CI[{d['lo']:+.3f},{d['hi']:+.3f}] {sig}")