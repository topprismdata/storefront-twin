#!/usr/bin/env python3
"""e19 — 多品牌零训练识别基准。
画廊 kNN: 每品牌按店 70/30 分库/测, CLIP top5 相似度, argmax+margin 判未知。
池: v12+v13 全部核验 A(388) + registry 左邻右舍正例。"""
import json, sys, re
import numpy as np, cv2, os
from pathlib import Path
from PIL import Image
ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))
import visual_baseline as vb
import e8_train as e8

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
    if "7-EL" in t.upper() or "711" in t or "7-Eleven" in t: return "7-11"
    if "全家" in t or "Fami" in t or "FamilyMart" in t.upper(): return "全家"
    return "local"

rows = []
for src, ddir in (("data/v12_labels.json", "data/images_v12_confusable"),
                  ("data/v13_labels.json", "data/images_v13_natl")):
    for f, v in json.load(open(ROOT / src)).items():
        if v.get("verdict") == "A":
            rows.append({"file": f, "path": str(ROOT / ddir / f),
                         "brand": canon(v.get("brand_text")), "poi": v["poi"]})
reg = json.load(open(ROOT / "data/registry_v11.json"))
sha = json.load(open(ROOT / "data/sha_file_index.json"))
for r in reg:
    if r.get("is_representative") and r["label"] == "positive" and r.get("store_id"):
        p = sha.get(r["sha256"])
        if p and (ROOT / "data" / p).exists():
            rows.append({"file": r["image_id"], "path": str(ROOT / "data" / p),
                         "brand": "左邻右舍", "poi": r["store_id"]})
print(f"池 {len(rows)} 张", flush=True)
import collections
print(collections.Counter(r["brand"] for r in rows), flush=True)

emb = vb.Embedder("clip")
vecs, okrows = [], []
for i, r in enumerate(rows):
    try:
        pil = Image.open(r["path"]).convert("RGB")
        vecs.append(emb.embed_pil([pil])[0]); okrows.append(r)
    except Exception: pass
    if (i+1) % 100 == 0: print(f"embed {i+1}/{len(rows)}", flush=True)
V = e8.l2(np.stack(vecs))

# 按品牌×店 70/30 分割
rng = np.random.RandomState(0)
major = {"美宜佳", "罗森", "7-11", "全家", "芙蓉兴盛", "京东便利店", "天猫小店", "左邻右舍"}
for r in okrows:
    r["is_major"] = r["brand"] in major
gidx, tidx = {}, {}
for b in set(r["brand"] for r in okrows):
    bi = [i for i, r in enumerate(okrows) if r["brand"] == b]
    stores = sorted(set(okrows[i]["poi"] for i in bi))
    rng.shuffle(stores)
    g_stores = set(stores[:max(1, int(len(stores)*0.7))])
    gidx[b] = [i for i in bi if okrows[i]["poi"] in g_stores]
    tidx[b] = [i for i in bi if okrows[i]["poi"] not in g_stores]

# 识别: 测试图 vs 每品牌画廊 top5 均值, argmax; margin<0.02 或 max<0.25 -> unknown
brands = sorted(gidx.keys())
gal_mats = {b: V[gidx[b]] for b in brands}
res = collections.Counter(); conf = collections.Counter()
THRESH, MARGIN = 0.25, 0.02
per_brand = collections.defaultdict(lambda: [0, 0])
for i in tidx.get("_all_", []) or []:
    pass
test_items = [(i, r) for b in brands for i in tidx[b] for r in [okrows[i]]]
for i, r in test_items:
    q = V[i]
    scores = {b: float(np.sort(gal_mats[b] @ q)[-5:].mean()) for b in brands}
    best = max(scores, key=scores.get)
    srt = sorted(scores.values(), reverse=True)
    pred = best if (srt[0] >= THRESH and srt[0] - srt[1] >= MARGIN) else "unknown"
    gold = r["brand"] if r["is_major"] else "local"
    if pred == "unknown": pred = "unknown"
    key = (gold, pred)
    conf[key] += 1
    per_brand[gold][1] += 1
    if pred == gold: per_brand[gold][0] += 1; res["acc"] += 1
    res["n"] += 1
print(f"\n总体: {res['acc']}/{res['n']} = {res['acc']/res['n']:.1%}")
print(f"{'品牌':<10} {'n':>4} {'acc':>6}")
for b, (c, n) in sorted(per_brand.items(), key=lambda kv: -kv[1][1]):
    print(f"{b:<10} {n:>4} {c/n:>6.1%}")
print("\n主要混淆(gold->pred):")
for (g, p), n in conf.most_common(12):
    if g != p: print(f"  {g} -> {p}: {n}")
json.dump({"n": res["n"], "acc": res["acc"]/res["n"],
           "per_brand": {b: {"n": n, "acc": c/n} for b, (c, n) in per_brand.items()},
           "confusion": {f"{g}->{p}": n for (g, p), n in conf.items() if g != p}},
          open(ROOT / "synthetic/vis_cache/e19_results.json", "w"), ensure_ascii=False, indent=1)
print("落盘", flush=True)
