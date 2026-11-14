#!/usr/bin/env python3
"""e18b — 诚实回归检查: 639 店组 LOSO, 每折训练=合成+registry他组+v12池293。
对照基线: 无池 E6-v3 冻结 0.636=0.839(fp90) / 0.80=0.913(fp26)。"""
import json, sys
import numpy as np
from pathlib import Path
ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))
import e8_train as e8
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

z = np.load(ROOT / "synthetic/vis_cache/e16/feats.npz", allow_pickle=True)
clip_full, dino_full, crop_fix = z["clip_full"], z["dino_full"], z["crop_fix"]
ids_order = [str(x) for x in z["ids"]]
reg = json.load(open(ROOT / "data/registry_v11.json"))
sha = json.load(open(ROOT / "data/sha_file_index.json"))
rows = []
for r in reg:
    if r.get("is_representative"):
        p = sha.get(r["sha256"])
        if p and (ROOT/"data"/p).exists():
            rows.append({"image_id": r["image_id"], "label": r["label"],
                         "group": r.get("store_id") or r["image_id"]})
assert [r["image_id"] for r in rows] == ids_order
y = np.array([r["label"] == "positive" for r in rows])
groups = np.array([r["group"] for r in rows])

zp = np.load(ROOT / "synthetic/vis_cache/e18_feats.npz", allow_pickle=True)
X_pool = zp["X"]

sr = e8.load_synth()
sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
X_syn = e8.concat_feats(sf)
y_syn = np.array([r["label"] == "positive" for r in sr])

def l2(x):
    return x / np.maximum(1e-9, np.linalg.norm(x, axis=1, keepdims=True))

X_reg = np.concatenate([l2(clip_full), l2(dino_full), l2(crop_fix)], axis=1)
glist = sorted(set(groups))
s_out = np.zeros(len(rows))
for fi, g in enumerate(glist):
    te = groups == g
    Xtr = np.vstack([X_syn, X_reg[~te], X_pool])
    ytr = np.concatenate([y_syn, y[~te], np.zeros(len(X_pool), bool)])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1500, C=1.0, random_state=0).fit(sc.transform(Xtr), ytr)
    s_out[te] = clf.predict_proba(sc.transform(X_reg[te]))[:, 1]
    if (fi+1) % 150 == 0:
        print(f"loso {fi+1}/{len(glist)}", flush=True)

def prf(s, t):
    p = s > t
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    P = tp/max(1, tp+fp); R = tp/max(1, tp+fn); F = 2*P*R/max(1e-12, P+R)
    return dict(P=round(P,3), R=round(R,3), F1=round(F,3), tp=tp, fp=fp, fn=fn)

out = {"with_pool_loso": {"thr0.636": prf(s_out, 0.636), "thr0.80": prf(s_out, 0.80)},
       "baseline_no_pool": {"thr0.636": {"F1": 0.839, "fp": 90, "fn": 6}, "thr0.80": {"F1": 0.913, "fp": 26, "fn": 19}}}
json.dump(out, open(ROOT / "synthetic/vis_cache/e18b_results.json", "w"), ensure_ascii=False, indent=1)
np.save(ROOT / "synthetic/vis_cache/e18b_scores.npy", s_out)
print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
