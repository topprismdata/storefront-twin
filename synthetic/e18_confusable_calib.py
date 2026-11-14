#!/usr/bin/env python3
"""e18 — 用 293 张核验 confusable 负例校准。
B1 池LOSO(5折按POI): 探针+池内他折重训 -> 留出折误报率;
B2 registry回归检查: 探针+池全量重训 -> registry 冻结0.636 F1 vs 0.839 基线;
T  阈值扫描(不重训): e6v3分 在池上的误报 vs registry 召回 包络。"""
import json, sys
import numpy as np, cv2
from pathlib import Path
from PIL import Image
ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))
import visual_baseline as vb
import e8_train as e8
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

labels = json.load(open(ROOT / "data/v12_labels.json"))
pool = [f for f, v in labels.items() if v["verdict"] == "A"]
print(f"池 {len(pool)} 张", flush=True)

cf = ROOT / "synthetic/vis_cache/e18_feats.npz"
if cf.exists():
    z = np.load(cf); X_pool, f_pool = z["X"], list(z["files"])
else:
    emb_c, emb_d = vb.Embedder("clip"), vb.Embedder("dinov2")
    X, files = [], []
    for i, f in enumerate(pool):
        p = ROOT / "data/images_v12_confusable" / f
        pil = Image.open(p).convert("RGB"); bgr = cv2.imread(str(p))
        c = e8.l2(emb_c.embed_pil([pil]))
        df = e8.l2(emb_d.embed_pil([pil]))
        box = vb.sign_crop_box(bgr)
        cc = bgr[box[1]:box[3], box[0]:box[2]]
        dc = e8.l2(emb_d.embed_pil([Image.fromarray(cv2.cvtColor(cc, cv2.COLOR_BGR2RGB))])) if cc.size else df
        X.append(np.concatenate([c, df, dc], axis=1)); files.append(f)
        if (i+1) % 100 == 0: print(f"embed {i+1}/{len(pool)}", flush=True)
    X_pool, f_pool = np.vstack(X), files
    np.savez(cf, X=X_pool, files=np.array(files))
    print("embed 完成", flush=True)

# 训练底座: 合成+registry 647
reg = json.load(open(ROOT / "data/registry_v11.json"))
sha = json.load(open(ROOT / "data/sha_file_index.json"))
rows = []
for r in reg:
    if r.get("is_representative"):
        p = sha.get(r["sha256"])
        if p and (ROOT/"data"/p).exists():
            rows.append({"image_id": r["image_id"], "path": str(ROOT/"data"/p), "label": r["label"]})
rf, _ = e8.embed_all(rows, "v4_real")
X_reg = e8.concat_feats(rf)
y_reg = np.array([r["label"] == "positive" for r in rows])
sr = e8.load_synth()
sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
X_syn = e8.concat_feats(sf)
y_syn = np.array([r["label"] == "positive" for r in sr])

pois = [labels[f]["poi"] for f in f_pool]
upois = sorted(set(pois))
rng = np.random.RandomState(0); rng.shuffle(upois)
folds = [set(upois[i::5]) for i in range(5)]

def train(X, y):
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=1500, C=1.0, random_state=0).fit(sc.transform(X), y)

out = {}
# B1 池LOSO
fp06, fp08, n_eval = 0, 0, 0
for k, hold in enumerate(folds):
    te = np.array([p in hold for p in pois])
    tr = ~te
    Xtr = np.vstack([X_syn, X_reg, X_pool[tr]])
    ytr = np.concatenate([y_syn, y_reg, np.zeros(int(tr.sum()), bool)])
    sc, clf = train(Xtr, ytr)
    s = clf.predict_proba(sc.transform(X_pool[te]))[:, 1]
    fp06 += int((s > 0.636).sum()); fp08 += int((s > 0.80).sum()); n_eval += int(te.sum())
    print(f"fold{k}: {int(te.sum())} eval, >0.636 {int((s>0.636).sum())}, >0.80 {int((s>0.80).sum())}", flush=True)
out["B1_pool_loso"] = {"n": n_eval, "fp@0.636": fp06, "fp@0.80": fp08,
                       "rate@0.636": round(fp06/n_eval, 3), "rate@0.80": round(fp08/n_eval, 3)}
# 基线(A臂, e6v3分)
old = {f: labels[f].get("fusion") for f in f_pool if labels[f].get("fusion")}
out["A_baseline_frozen"] = {"n": len(old),
    "fp@0.636": sum(1 for v in old.values() if v > 0.636), "fp@0.80": sum(1 for v in old.values() if v > 0.80)}
# B2 registry回归
X_all = np.vstack([X_syn, X_reg, X_pool])
y_all = np.concatenate([y_syn, y_reg, np.zeros(len(X_pool), bool)])
sc2, clf2 = train(X_all, y_all)
s_reg = clf2.predict_proba(sc2.transform(X_reg))[:, 1]
def prf(s, t):
    p = s > t
    tp = int((p & y_reg).sum()); fp = int((p & ~y_reg).sum()); fn = int((~p & y_reg).sum())
    P = tp/max(1, tp+fp); R = tp/max(1, tp+fn); F = 2*P*R/max(1e-12, P+R)
    return dict(P=round(P,3), R=round(R,3), F1=round(F,3), fp=fp, fn=fn)
out["B2_registry_with_pool"] = {"thr0.636": prf(s_reg, 0.636), "thr0.80": prf(s_reg, 0.80)}
out["B2_registry_baseline"] = {"thr0.636": {"F1": 0.839, "fp": 90}, "thr0.80": {"F1": 0.913, "fp": 26}}
# T 阈值扫描(e6v3分 vs 池)
old_scores = np.array([labels[f]["fusion"] for f in f_pool if labels[f].get("fusion") is not None])
sw = []
for t in (0.636, 0.70, 0.80, 0.85, 0.90, 0.95):
    fp_rate = float((old_scores > t).mean())
    sw.append({"thr": t, "pool_fp_rate": round(fp_rate, 3)})
out["T_threshold_sweep"] = sw
json.dump(out, open(ROOT / "synthetic/vis_cache/e18_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
