#!/usr/bin/env python3
"""E10 全量盲测门店级 Bootstrap 置信区间（模式A纯拒识 / 模式B拒识+救援）。
按 store_id/group 整簇重采样 B=2000，固定裁决（VLM 输出已冻结）。
产物 vis_cache/e10_full/bootstrap_ci.json。"""
import json
import numpy as np

ROOT = "/Users/mac/chatgpt-harvest/6af41562/synthetic/vis_cache/e10_full"
recs = json.load(open(f"{ROOT}/records.json"))
key = json.load(open(f"{ROOT}/truth_key.json"))
# 补 group
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

def final_pred(mode, idx):
    e = e8[idx]; v = verd[idx]
    if mode == "A":  # 纯拒识
        return np.where(e, ~np.isin(v, ["NON_MATCH", "OUT_OF_SCOPE"]), False)
    else:  # B 拒识+救援
        out = e.copy()
        out[v == "MATCH"] = True
        out[np.isin(v, ["NON_MATCH", "OUT_OF_SCOPE"])] = False
        return out

def prf(pred, t):
    tp = int(np.sum(pred & t)); fp = int(np.sum(pred & ~t)); fn = int(np.sum(~pred & t))
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)

B = 2000
out = {}
for mode in ["A", "B"]:
    boot = {k: [] for k in ["p", "r", "f1"]}
    for b in range(B):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in sampled])
        t = truth[idx]
        if t.sum() == 0 or (~t).sum() == 0:
            continue
        pred = final_pred(mode, idx)
        p, r, f1 = prf(pred, t)
        boot["p"].append(p); boot["r"].append(r); boot["f1"].append(f1)
    ci = {}
    for k, v in boot.items():
        v = np.array(v)
        ci[k] = dict(mean=round(float(v.mean()), 3),
                     lo=round(float(np.percentile(v, 2.5)), 3),
                     hi=round(float(np.percentile(v, 97.5)), 3))
    # 点估计
    pred = final_pred(mode, np.arange(len(recs)))
    p, r, f1 = prf(pred, truth)
    out[mode] = dict(point=dict(precision=round(p, 3), recall=round(r, 3), f1=round(f1, 3)), ci=ci)

out["n_groups"] = len(uniq)
out["n_pos_groups"] = sum(1 for g in uniq if truth[gidx[g]].any())
out["n_bootstrap"] = B
json.dump(out, open(f"{ROOT}/bootstrap_ci.json", "w"), ensure_ascii=False, indent=1)
print("groups:", len(uniq), "pos_groups:", out["n_pos_groups"])
for mode in ["A", "B"]:
    print(f"模式{mode}: point {out[mode]['point']}")
    for k, v in out[mode]["ci"].items():
        print(f"   {k}: {v['mean']} [{v['lo']}, {v['hi']}]")