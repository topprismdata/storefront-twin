#!/usr/bin/env python3
"""e24 — 开集拒识: local(非目录品牌) 正确行为=拒识。
规则网格: 视觉 top1 概率 τ × 文字规则(OCR 有实质文字但无别名命中 → 非目录信号)。
指标: local 拒识率 / 已知品牌误拒率 / 已知品牌端到端 acc。"""
import json, sys, collections, re
import numpy as np
from pathlib import Path
ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))
import e8_train as e8

z = np.load(ROOT / "synthetic/vis_cache/e22_feats.npz", allow_pickle=True)
Vc, Vd, okrows = z["Vc"], z["Vd"], list(z["rows"])
X = np.hstack([Vc, Vd])
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
all_gal = sorted(i for b in brands for i in gidx[b])
gsc = StandardScaler().fit(X[all_gal])
Xg = gsc.transform(X[all_gal])
gbrands = [okrows[i]["brand"] for i in all_gal]
probes = {}
for b in brands:
    y = np.array([1 if g == b else 0 for g in gbrands])
    if y.sum() < 5:
        continue
    probes[b] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xg, y)

sys.path.insert(0, str(ROOT / "synthetic"))
from e20_text_confirm import det_boxes, merge_lines, rec_crop

ALIAS = {
    "每一天": ["每一天"], "见福": ["见福"], "唐久": ["唐久"], "红旗连锁": ["红旗连锁", "红旗超市", "红旗"],
    "零食很忙": ["零食很忙"], "赵一鸣零食": ["赵一鸣", "赵一鸣省钱超市"], "好想来": ["好想来"],
    "家家悦": ["家家悦"], "盒马鲜生": ["盒马", "盒马鲜生"], "好特卖": ["好特卖", "HOTMAXX"],
    "快客": ["快客"], "恰货铺子": ["恰货铺子"], "天福": ["天福"], "万嘉": ["万嘉"],
    "好德": ["好德"], "悦来悦喜": ["悦来悦喜"], "零食有鸣": ["零食有鸣"], "喜士多": ["喜士多"],
    "丹尼斯": ["丹尼斯"], "苏宁小店": ["苏宁小店", "苏宁"], "京客隆": ["京客隆"],
    "北京华联": ["北京华联", "BHG"], "天虹": ["天虹"], "盒马NB": ["盒马NB"], "TODAY": ["TODAY"],
    "戴永红": ["戴永红"], "新佳宜": ["新佳宜"], "良品铺子": ["良品铺子", "良品"],
    "老婆大人": ["老婆大人"], "谊品生鲜": ["谊品生鲜", "谊品"], "利群": ["利群"],
    "山姆": ["山姆", "SAM", "SAMS"], "华润万家": ["华润万家", "华润"], "开市客": ["开市客", "COSTCO"],
    "左邻右舍": ["左邻", "右舍"], "美宜佳": ["美宜佳"], "芙蓉兴盛": ["芙蓉兴盛", "芙蓉", "兴盛"],
    "京东便利店": ["京东"], "天猫小店": ["天猫"], "便利蜂": ["便利蜂"], "十足": ["十足"],
    "罗森": ["罗森", "LAWSON"], "7-11": ["7-ELEVEN", "711"], "全家": ["全家", "FAMILYMART"],
}
STOP = {"便利", "超市", "小店", "便利店", "零食"}


def confirm(cands, texts):
    tn = [re.sub(r"\s+", "", t).upper() for t in texts if t and t.strip()]

    def hit(a, tn):
        na = re.sub(r"\s+", "", a).upper()
        if not na:
            return False
        if na in "".join(tn):
            return True
        if len(na) >= 3 and na[:2] not in STOP and any(na[:2] in t for t in tn):
            return True
        if len(na) >= 5 and all(ord(c) < 256 for c in na):
            if any(na in t or t in na for t in tn if t):
                return True
            import difflib
            return any(difflib.SequenceMatcher(None, na, t).ratio() >= 0.75 for t in tn if len(t) >= 5)
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
galX = {b: X[gidx[b]] for b in brands}
out = []
for k, (i, gold) in enumerate(test_items):
    r = okrows[i]
    ss = {bb: probes[bb].predict_proba(gsc.transform(X[i][None]))[0, 1] for bb in probes}
    vals = sorted(ss.values(), reverse=True)
    cands = [bb for bb, _ in sorted(ss.items(), key=lambda kv: -kv[1])[:3]]
    # 画廊最近邻余弦(拼接特征, 未归一 → 先归一)
    q = X[i] / (np.linalg.norm(X[i]) + 1e-9)
    near = max(float(((galX[bb] / np.linalg.norm(galX[bb], axis=1, keepdims=True)) @ q).max()) for bb in brands)
    bgr = cv2.imread(r["path"]) if (cv2 := __import__("cv2")) else None
    texts = [rec_crop(bgr, bx) for bx in merge_lines(det_boxes(bgr))] if bgr is not None else []
    pred, source = confirm(cands, texts)
    cjk = sum(1 for t in texts for ch in t if '\u4e00' <= ch <= '\u9fff')
    out.append({"gold": gold, "pred": pred, "source": source,
                "p_top1": round(vals[0], 3), "margin": round(vals[0] - vals[1], 3),
                "near": round(near, 4), "cjk": cjk, "is_local": gold == "local"})
    if (k + 1) % 60 == 0:
        print(f"{k+1}/{len(test_items)}", flush=True)

json.dump(out, open(ROOT / "synthetic/vis_cache/e24_openset.json", "w"), ensure_ascii=False, indent=1)
loc = [o for o in out if o["is_local"]]
kn = [o for o in out if not o["is_local"]]
print(f"\nlocal 测试 {len(loc)}, 已知品牌测试 {len(kn)}")
print("local near 分布:", sorted(o["near"] for o in loc)[:5], "...", sorted(o["near"] for o in loc)[-5:])
print("known near 分布:", sorted(o["near"] for o in kn)[:5], "...", sorted(o["near"] for o in kn)[-5:])
print(f"{'规则':<30} {'local拒识':>9} {'已知误拒':>8} {'已知acc':>7}")


def run_rule(name, rej_fn):
    rej = sum(1 for o in loc if rej_fn(o)) / max(1, len(loc))
    fr = sum(1 for o in kn if rej_fn(o)) / max(1, len(kn))
    acc = sum(1 for o in kn if not rej_fn(o) and o["pred"] == o["gold"]) / max(1, len(kn))
    print(f"{name:<30} {rej:>8.1%} {fr:>7.1%} {acc:>6.1%}")


for nv in (0.62, 0.66, 0.70, 0.74):
    run_rule(f"near<{nv}", lambda o, nv=nv: o["source"] not in ("text_cand", "text_all") and o["near"] < nv)
for nv, mg in ((0.66, 0.02), (0.70, 0.02), (0.70, 0.05)):
    run_rule(f"near<{nv}|margin<{mg}",
             lambda o, nv=nv, mg=mg: o["source"] not in ("text_cand", "text_all") and (o["near"] < nv or o["margin"] < mg))
run_rule("near<0.66 & cjk>=4(无文字命中)",
         lambda o: o["source"] not in ("text_cand", "text_all") and o["near"] < 0.66 and o["cjk"] >= 4)
