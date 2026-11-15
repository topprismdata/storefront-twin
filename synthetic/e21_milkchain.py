#!/usr/bin/env python3
"""e21 — 液奶连锁表 10 新品牌并入: 每一天/见福/唐久/红旗连锁/零食很忙/赵一鸣/好想来/家家悦/盒马鲜生/好特卖。
流程 = e19c 探针 + e20 OCR 文字确认, 确定性分割(sorted set), gold 争议通道, 开集拒识。
数据: v14 verdict-A 图 + 既有 e19_feats 增量嵌入新图。"""
import json, sys, collections, re
import numpy as np, cv2, os
from pathlib import Path
from PIL import Image
ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))
import visual_baseline as vb
import e8_train as e8
from e20_text_confirm import det_boxes, merge_lines, rec_crop, confirm as e20_confirm

NEW_BRANDS = {
    "每一天": ["每一天", "每天便利"],
    "见福": ["见福", "见福便利"],
    "唐久": ["唐久", "唐久便利"],
    "红旗连锁": ["红旗连锁", "红旗超市", "红旗"],
    "零食很忙": ["零食很忙"],
    "赵一鸣零食": ["赵一鸣", "赵一鸣零食"],
    "好想来": ["好想来"],
    "家家悦": ["家家悦"],
    "盒马鲜生": ["盒马", "盒马鲜生", "盒马FRESH", "盒马超市"],
    "好特卖": ["好特卖", "HOTMAXX", "好特卖HotMaxx"],
}


def canon(t):
    t = str(t)
    for b, als in NEW_BRANDS.items():
        for a in als:
            if a in t or (len(a) >= 4 and a.upper() in t.upper()):
                return b
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


# 既有行(e19 feats 缓存) + v14 新行
z = np.load(ROOT / "synthetic/vis_cache/e19_feats.npz", allow_pickle=True)
Vc, Vd, okrows = list(z["Vc"]), list(z["Vd"]), list(z["rows"])
have = {r["file"] for r in okrows}
lab = json.load(open(ROOT / "data/v14_labels.json"))
ecc, edd = vb.Embedder("clip"), vb.Embedder("dinov2")
for f, v in lab.items():
    if v.get("verdict") != "A" or f in have:
        continue
    p = ROOT / "data/images_v14_milkchain" / f
    if not p.exists():
        continue
    b = canon(v.get("brand_text"))
    if b == "local":
        continue
    try:
        pil = Image.open(p).convert("RGB")
        Vc.append(ecc.embed_pil([pil])[0]); Vd.append(edd.embed_pil([pil])[0])
        okrows.append({"file": f, "path": str(p), "brand": b, "poi": v["poi"], "new": True})
    except Exception:
        pass
Vc, Vd = e8.l2(np.stack(Vc)), e8.l2(np.stack(Vd))
print(f"总行 {len(okrows)} (含新增 {sum(1 for r in okrows if r.get('new'))})", flush=True)
np.savez(ROOT / "synthetic/vis_cache/e21_feats.npz", Vc=Vc, Vd=Vd, rows=np.array(okrows, dtype=object))

brands = sorted(set(r["brand"] for r in okrows))
rng = np.random.RandomState(0)
gidx, tidx = {}, {}
for b in brands:
    bi = [i for i, r in enumerate(okrows) if r["brand"] == b]
    stores = sorted(set(okrows[i]["poi"] for i in bi))
    rng.shuffle(stores)
    g_stores = set(stores[:max(1, int(len(stores) * 0.7))])
    gidx[b] = [i for i in bi if okrows[i]["poi"] in g_stores]
    tidx[b] = [i for i in bi if okrows[i]["poi"] not in g_stores]

# 探针: 每品牌 one-vs-rest 逻辑回归(特征 = CLIP+DINO 拼接, scaler 在全画廊拟合)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = np.hstack([Vc, Vd])
all_gal = sorted(i for b in brands for i in gidx[b])
gsc = StandardScaler().fit(X[all_gal])
Xg = gsc.transform(X[all_gal])
gbrands = [okrows[i]["brand"] for i in all_gal]
probes = {}
for b in brands:
    y = np.array([1 if g == b else 0 for g in gbrands])
    if y.sum() < 5:
        continue
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xg, y)
    probes[b] = lr
print(f"探针 {len(probes)} 品牌", flush=True)

# 端到端评测(探针 top-3 + OCR 文字确认)
ALIAS = {b: als + [b] for b, als in NEW_BRANDS.items()}
ALIAS.update({
    "左邻右舍": ["左邻", "右舍"], "美宜佳": ["美宜佳"], "芙蓉兴盛": ["芙蓉兴盛", "芙蓉", "兴盛"],
    "京东便利店": ["京东便利店", "京东"], "天猫小店": ["天猫小店", "天猫"],
    "便利蜂": ["便利蜂"], "十足": ["十足"], "罗森": ["罗森", "LAWSON"],
    "7-11": ["7-ELEVEN", "711", "SEVEN"], "全家": ["全家", "FAMILYMART"],
})
STOP = {"便利", "超市", "小店", "便利店", "零食"}


def confirm(cands, texts):
    tn = [re.sub(r"\s+", "", t).upper() for t in texts if t and t.strip()]
    def norm(a):
        return re.sub(r"\s+", "", a).upper()
    def hit(a, tn):
        na = norm(a)
        if not na:
            return False
        joined = "".join(tn)
        if na in joined:
            return True
        fuzzy_ok = len(na) >= 3 and na[:2] not in STOP
        if fuzzy_ok and any(na[:2] in t for t in tn):
            return True
        if len(na) >= 5 and all(ord(c) < 256 for c in na):
            if any(na in t or t in na for t in tn if t):
                return True
            import difflib
            return any(difflib.SequenceMatcher(None, na, t).ratio() >= 0.75 for t in tn if t and len(t) >= 5)
        # CJK 掐头/去尾(容忍首/尾字误读): 盒马鲜生 alias[1:]="马鲜生" ⊂ "食马鲜生"
        if len(na) >= 4 and any(ord(c) > 255 for c in na):
            tail, head = na[1:], na[:-1]
            return any(tail in t or head in t for t in tn)
        return False
    hits = [b for b in cands if any(hit(a, tn) for a in ALIAS.get(b, []))]
    if hits:
        return hits[0], "text_cand"
    for b, als in ALIAS.items():
        if any(hit(a, tn) for a in als):
            return b, "text_all"
    return cands[0], "visual_fallback"


test_items = [(i, b) for b in brands for i in tidx[b]]
out = []
for k, (i, gold) in enumerate(test_items):
    r = okrows[i]
    ss = {}
    try:
        x = X[i]
        ss = {bb: probes[bb].predict_proba(gsc.transform(x[None]))[0, 1] for bb in probes}
    except Exception:
        pass
    cands = [b for b, _ in sorted(ss.items(), key=lambda kv: -kv[1])[:3]]
    bgr = cv2.imread(r["path"])
    texts = [rec_crop(bgr, bx) for bx in merge_lines(det_boxes(bgr))] if bgr is not None else []
    pred, source = confirm(cands, texts)
    out.append({"file": r["file"], "gold": gold, "pred": pred, "source": source,
                "texts": texts[:4], "p_top1": round(float(max(ss.values())), 3) if ss else 0,
                "pred_vision": max(ss, key=ss.get) if ss else "none",
                "is_new": bool(r.get("new"))})
    if (k + 1) % 40 == 0:
        print(f"{k+1}/{len(test_items)}", flush=True)

for o in out:
    o["gold_disputed"] = o["pred"] != o["gold"] and o["source"] in ("text_cand", "text_all")
known = [o for o in out if o["gold"] != "local" and not o["gold_disputed"]]
newb = [o for o in known if o["is_new"]]
oldb = [o for o in known if not o["is_new"]]
fa = sum(1 for o in known if o["pred"] == o["gold"]) / max(1, len(known))
fn = sum(1 for o in newb if o["pred"] == o["gold"]) / max(1, len(newb))
fo = sum(1 for o in oldb if o["pred"] == o["gold"]) / max(1, len(oldb))
per = collections.defaultdict(lambda: [0, 0])
for o in known:
    per[o["gold"]][1] += 1
    if o["pred"] == o["gold"]:
        per[o["gold"]][0] += 1
print(f"\n目录内端到端 acc: {fa:.1%} (新10品牌 {fn:.1%} n={len(newb)} / 老品牌 {fo:.1%} n={len(oldb)})")
print("判决来源:", dict(collections.Counter(o["source"] for o in known)))
print(f"{'品牌':<10} {'n':>4} {'acc':>6}")
for b, (c, n) in sorted(per.items(), key=lambda kv: -kv[1][1]):
    print(f"{b:<10} {n:>4} {c/n:>6.1%}")
json.dump(out, open(ROOT / "synthetic/vis_cache/e21_results.json", "w"), ensure_ascii=False, indent=1)
