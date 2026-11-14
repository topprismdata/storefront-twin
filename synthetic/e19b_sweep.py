#!/usr/bin/env python3
"""e19b — 多品牌基准配置扫描: 特征×topk×阈值, 找最优操作点。"""
import json, sys, collections
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

fc = ROOT / "synthetic/vis_cache/e19_feats.npz"
if fc.exists():
    z = np.load(fc, allow_pickle=True)
    Vc, Vd, okrows = z["Vc"], z["Vd"], list(z["rows"])
else:
    ecc, edd = vb.Embedder("clip"), vb.Embedder("dinov2")
    Vc, Vd, okrows = [], [], []
    for i, r in enumerate(rows):
        try:
            pil = Image.open(r["path"]).convert("RGB")
            Vc.append(ecc.embed_pil([pil])[0]); Vd.append(edd.embed_pil([pil])[0]); okrows.append(r)
        except Exception: pass
        if (i+1) % 150 == 0: print(f"embed {i+1}/{len(rows)}", flush=True)
    Vc, Vd = e8.l2(np.stack(Vc)), e8.l2(np.stack(Vd))
    np.savez(fc, Vc=Vc, Vd=Vd, rows=np.array(okrows, dtype=object))
    print("embed 完成", flush=True)

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

def run(V, topk, thresh, margin=0.0):
    gal = {b: V[gidx[b]] for b in brands}
    res = collections.defaultdict(lambda: [0, 0])
    for b in brands:
        for i in tidx[b]:
            q = V[i]
            sc = {bb: float(np.sort(gal[bb] @ q)[-topk:].mean()) for bb in brands}
            vals = sorted(sc.values(), reverse=True)
            pred = max(sc, key=sc.get)
            if vals[0] < thresh or vals[0] - vals[1] < margin:
                pred = "unknown"
            gold = okrows[i]["brand"]
            res[gold][1] += 1
            if pred == gold: res[gold][0] += 1
    tot_c = sum(v[0] for v in res.values()); tot_n = sum(v[1] for v in res.values())
    return tot_c / tot_n, res

for feat, V in (("CLIP", None), ("DINO", None)):
    Vv = Vc if feat == "CLIP" else Vd
    for topk in (1, 3, 5):
        for th, mg in ((0.0, 0.0), (0.25, 0.0), (0.30, 0.01), (0.35, 0.01)):
            acc, res = run(Vv, topk, th, mg)
            unk = sum(v[1]-v[0] for v in res.values() if v[1] and res is not None)
            tgt = res["左邻右舍"]
            print(f"{feat} top{topk} th={th} mg={mg}: 总acc {acc:.1%} | 左邻右舍 {tgt[0]}/{tgt[1]}", flush=True)
