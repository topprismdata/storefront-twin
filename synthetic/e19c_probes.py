#!/usr/bin/env python3
"""e19c — 多品牌 per-brand 探针(one-vs-rest, 2048维 CLIP+DINO) vs kNN 基准。"""
import json, sys, collections
import numpy as np, os
from pathlib import Path
ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))
import e8_train as e8
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def canon(t):
    t = str(t)
    if "左邻" in t: return "左邻右舍"
    if "美宜佳" in t: return "美宜佳"
    if "芙蓉兴盛" in t: return "芙蓉兴盛"
    if "京东" in t: return "京东便利店"
    if "天猫" in t: return "天猫小店"
    if "便利蜂" in t: return "便利蜂"
    if "十足" in t: return "十足"
    if "罗森" in t or "LAWSON" in t.upper(): return "罗森"
    if "7-EL" in t.upper() or "711" in t: return "7-11"
    if "全家" in t or "Fami" in t or "FamilyMart" in t.upper(): return "全家"
    return "local"

rows = []
for src, ddir in (("data/v12_labels.json", "data/images_v12_confusable"),
                  ("data/v13_labels.json", "data/images_v13_natl")):
    for f, v in json.load(open(ROOT / src)).items():
        if v.get("verdict") == "A":
            rows.append({"file": f, "path": str(ROOT / ddir / f), "brand": canon(v.get("brand_text")), "poi": v["poi"]})
reg = json.load(open(ROOT / "data/registry_v11.json"))
sha = json.load(open(ROOT / "data/sha_file_index.json"))
for r in reg:
    if r.get("is_representative") and r["label"] == "positive" and r.get("store_id"):
        p = sha.get(r["sha256"])
        if p and (ROOT / "data" / p).exists():
            rows.append({"file": r["image_id"], "path": str(ROOT / "data" / p), "brand": "左邻右舍", "poi": r["store_id"]})

z = np.load(ROOT / "synthetic/vis_cache/e19_feats.npz", allow_pickle=True)
Vc, Vd, okrows = z["Vc"], z["Vd"], [dict(x) for x in z["rows"]]
X = np.concatenate([e8.l2(Vc), e8.l2(Vd)], axis=1)  # 1280 维(CLIP512+DINO768)
major = {"美宜佳", "罗森", "7-11", "全家", "芙蓉兴盛", "京东便利店", "天猫小店", "左邻右舍"}
rng = np.random.RandomState(0)
gidx, tidx = {}, {}
for b in set(r["brand"] for r in okrows):
    bi = [i for i, r in enumerate(okrows) if r["brand"] == b]
    stores = sorted(set(okrows[i]["poi"] for i in bi))
    rng.shuffle(stores)
    g_stores = set(stores[:max(1, int(len(stores)*0.7))])
    gidx[b] = [i for i in bi if okrows[i]["poi"] in g_stores]
    tidx[b] = [i for i in bi if okrows[i]["poi"] not in g_stores]
brands = sorted(gidx.keys())

# 每品牌 one-vs-rest 探针(训练=各品牌画廊+local画廊做负例底座)
probes = {}
for b in brands:
    pos = gidx[b]
    neg = [i for bb in brands if bb != b for i in gidx[bb]]
    Xtr = X[pos + neg]
    ytr = np.array([1]*len(pos) + [0]*len(neg))
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1500, C=1.0, random_state=0).fit(sc.transform(Xtr), ytr)
    probes[b] = (sc, clf)

res = collections.defaultdict(lambda: [0, 0])
for b in brands:
    for i in tidx[b]:
        ss = {bb: probes[bb][1].predict_proba(probes[bb][0].transform(X[i][None]))[0, 1] for bb in brands}
        pred = max(ss, key=ss.get)
        gold = okrows[i]["brand"]
        res[gold][1] += 1
        if pred == gold: res[gold][0] += 1
tot_c = sum(v[0] for v in res.values()); tot_n = sum(v[1] for v in res.values())
print(f"per-brand 探针 总acc: {tot_c}/{tot_n} = {tot_c/tot_n:.1%}  (kNN最优 63.0%)")
print(f"{'品牌':<10} {'n':>4} {'acc':>6}")
for b, (c, n) in sorted(res.items(), key=lambda kv: -kv[1][1]):
    print(f"{b:<10} {n:>4} {c/n:>6.1%}")
json.dump({"acc": tot_c/tot_n, "per_brand": {b: {"n": n, "acc": c/n} for b, (c, n) in res.items()}},
          open(ROOT / "synthetic/vis_cache/e19c_results.json", "w"), ensure_ascii=False, indent=1)
