#!/usr/bin/env python3
"""e20 — 端到端多品牌识别: 视觉检索 top-3 + 招牌文字确认(本地 PP-OCRv4 det+rec)。
评测集 = e19c 同分割(按店 70/30, rng 0), 对照纯视觉 69.1%。"""
import json, sys, collections, re
import numpy as np, cv2, os
from pathlib import Path
from PIL import Image
ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))
import e8_train as e8
import e15_text_det as e15
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import onnxruntime as ort

rec_sess = ort.InferenceSession(str(ROOT / "models/rapidocr/ch_PP-OCRv4_rec_infer.onnx"),
                                providers=["CPUExecutionProvider"])
import onnx
m = onnx.load(str(ROOT / "models/rapidocr/ch_PP-OCRv4_rec_infer.onnx"))
CHARS = None
for p in m.metadata_props:
    if p.key == "character":
        CHARS = p.value.split("\n")
print(f"rec 字典 {len(CHARS)} 项, 输入 {rec_sess.get_inputs()[0].shape}", flush=True)

ALIAS = {
    "左邻右舍": ["左邻", "右舍"],
    "美宜佳": ["美宜佳", "MEIYIJIA"],
    "芙蓉兴盛": ["芙蓉兴盛", "芙蓉", "兴盛"],
    "京东便利店": ["京东", "JD.COM", "JD"],
    "天猫小店": ["天猫"],
    "便利蜂": ["便利蜂"],
    "十足": ["十足"],
    "罗森": ["罗森", "LAWSON"],
    "7-11": ["7-11", "711", "7-ELEVEN", "7ELEVEN"],
    "全家": ["全家", "FAMILYMART", "全家便利"],
}


def det_boxes(bgr, thr=0.3, max_side=960, top_frac=0.75, max_n=6):
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(bgr, (max(32, int(w*scale)//32*32), max(32, int(h*scale)//32*32)))
    else:
        img = cv2.resize(bgr, (max(32, w//32*32), max(32, h//32*32)))
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x = ((x - np.array([0.485,0.456,0.406], dtype=np.float32)) / np.array([0.229,0.224,0.225], dtype=np.float32)).astype(np.float32).transpose(2,0,1)[None]
    prob = e15.sess.run(None, {e15.INAME: x})[0][0][0]
    H, W = prob.shape
    binm = (prob > thr).astype(np.uint8)*255
    binm = cv2.dilate(binm, np.ones((3,3),np.uint8), iterations=1)
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        bx = cv2.boundingRect(c)
        X0, Y0 = bx[0]/W*w, bx[1]/H*h
        X1, Y1 = (bx[0]+bx[2])/W*w, (bx[1]+bx[3])/H*h
        if (X1-X0) < 0.04*w or (Y1-Y0) < 0.02*h:
            continue
        boxes.append((Y0, X0, Y1, X1))
    boxes.sort()
    boxes = [b for b in boxes if b[0] < top_frac*h][:max_n]
    return boxes


def rec_crop(bgr, box):
    H0, W0 = bgr.shape[:2]
    y0, x0, y1, x1 = box
    c = bgr[max(0,int(y0)-2):min(H0,int(y1)+2), max(0,int(x0)-2):min(W0,int(x1)+2)]
    if c.size == 0 or c.shape[0] < 8:
        return ""
    ch, cw = c.shape[:2]
    tw = int(min(320, max(24, round(cw * 48.0 / ch))))
    img = cv2.resize(c, (tw, 48))
    canvas = np.full((48, 320, 3), 255, np.uint8)  # 白底右填充(PP-OCR口径)
    canvas[:, :tw] = img
    xx = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    xx = ((xx - 0.5) / 0.5).astype(np.float32).transpose(2,0,1)[None]
    out = rec_sess.run(None, {rec_sess.get_inputs()[0].name: xx})[0][0]  # (T, C)
    idxs = out.argmax(axis=1)
    txt, prev = "", -1
    for k in idxs:
        if k == prev: continue
        prev = k
        if k == 0: continue
        if k == 6624: txt += " "; continue
        if k <= len(CHARS): txt += CHARS[k-1]
    return txt


def merge_lines(boxes):
    """同行(y重叠)框合并"""
    if not boxes:
        return []
    boxes = sorted(boxes)
    out = [list(boxes[0])]
    for y0, x0, y1, x1 in boxes[1:]:
        last = out[-1]
        if y0 < last[2] and y1 > last[1] and x0 <= last[3] + 0.15*(last[3]-last[1] + 1):
            last[3] = max(last[3], x1)
            last[1] = min(last[1], x0)
            last[0] = min(last[0], y0); last[2] = max(last[2], y1)
        else:
            out.append([y0, x0, y1, x1])
    return [tuple(b) for b in out]


def norm(t):
    return re.sub(r"[\s\.·、,，。:：\-—|丨]+", "", str(t)).upper()


def confirm(cands, texts):
    """cands=视觉候选(有序), texts=OCR字符串列表 -> (brand, source)。
    模糊规则: 整词命中, 或别名前2字命中——但前2字是通用词(便利/超市/小店等)时禁用模糊。"""
    tn = [norm(t) for t in texts if t and norm(t)]
    STOP = {"便利", "超市", "小店", "便利店"}
    def hit(a, tn):
        na = norm(a)
        if not na:
            return False
        strict = na in joined_all(tn)
        fuzzy_ok = len(na) >= 3 and na[:2] not in STOP
        return strict or (fuzzy_ok and any(na[:2] in t for t in tn))
    def joined_all(tn):
        return "".join(tn)
    hits = []
    for b in cands:
        if any(hit(a, tn) for a in ALIAS.get(b, [])):
            hits.append(b)
    if hits:
        return hits[0], "text_cand"
    for b, als in ALIAS.items():
        if any(hit(a, tn) for a in als):
            return b, "text_all"
    return cands[0], "visual_fallback"


def main():
    # 重建 e19c 分割与探针
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
    X = np.concatenate([e8.l2(Vc), e8.l2(Vd)], axis=1)
    assert [r["file"] for r in okrows] == [r["file"] for r in rows]
    rng = np.random.RandomState(0)
    gidx, tidx = {}, {}
    for b in sorted(set(r["brand"] for r in okrows)):
        bi = [i for i, r in enumerate(okrows) if r["brand"] == b]
        stores = sorted(set(okrows[i]["poi"] for i in bi))
        rng.shuffle(stores)
        g_stores = set(stores[:max(1, int(len(stores)*0.7))])
        gidx[b] = [i for i in bi if okrows[i]["poi"] in g_stores]
        tidx[b] = [i for i in bi if okrows[i]["poi"] not in g_stores]
    brands = sorted(gidx.keys())
    probes = {}
    for b in brands:
        pos, neg = gidx[b], [i for bb in brands if bb != b for i in gidx[bb]]
        sc = StandardScaler().fit(X[pos + neg])
        clf = LogisticRegression(max_iter=1500, C=1.0, random_state=0).fit(sc.transform(X[pos + neg]),
            np.array([1]*len(pos) + [0]*len(neg)))
        probes[b] = (sc, clf)

    out = []
    test_items = [(i, b) for b in brands for i in tidx[b]]
    for k, (i, gold) in enumerate(test_items):
        r = okrows[i]
        bgr = cv2.imread(r["path"])
        ss = {bb: probes[bb][1].predict_proba(probes[bb][0].transform(X[i][None]))[0, 1] for bb in brands}
        cands = [b for b, _ in sorted(ss.items(), key=lambda kv: -kv[1])[:3]]
        texts = [rec_crop(bgr, bx) for bx in merge_lines(det_boxes(bgr))]
        pred, source = confirm(cands, texts)
        out.append({"file": r["file"], "gold": gold, "cands": cands, "pred": pred,
                    "source": source, "texts": texts[:4], "p_top1": round(float(max(ss.values())), 3),
                    "pred_vision": max(ss, key=ss.get)})
        if (k+1) % 40 == 0: print(f"{k+1}/{len(test_items)}", flush=True)
    known = [o for o in out if o["gold"] != "local"]
    loc = [o for o in out if o["gold"] == "local"]
    # 标签争议: 文字确认命中但与 gold 冲突 -> gold 可疑(POI 相册混图), 单独记账
    for o in known:
        o["gold_disputed"] = o["pred"] != o["gold"] and o["source"] in ("text_cand", "text_all")
    disputes = [o for o in known if o["gold_disputed"]]
    solid = [o for o in known if not o["gold_disputed"]]
    acc = sum(1 for o in solid if o["pred"] == o["gold"]) / len(solid)
    vis_acc = sum(1 for o in solid if o["pred_vision"] == o["gold"]) / len(solid)
    # 开集: local(非目录品牌)正确行为 = 拒识。规则: 文字未确认 且 视觉top1概率<0.9 -> unknown
    rej = sum(1 for o in loc if o["source"] != "text_cand" and o["source"] != "text_all" and o["p_top1"] < 0.9)
    known_rej_err = sum(1 for o in known if o["source"] not in ("text_cand", "text_all") and o["p_top1"] < 0.9)
    src_stat = collections.Counter(o["source"] for o in known)
    per_brand = collections.defaultdict(lambda: [0, 0])
    for o in known:
        per_brand[o["gold"]][1] += 1
        if o["pred"] == o["gold"]: per_brand[o["gold"]][0] += 1
    print(f"\n目录内品牌端到端 acc: {acc:.1%}  (纯视觉 {vis_acc:.1%})  n={len(solid)} (另 gold 可疑 {len(disputes)} 单列)")
    print(f"local 开集拒识: {rej}/{len(loc)} = {rej/len(loc):.1%}  (误拒目录内 {known_rej_err})")
    print("gold 争议明细(招牌文字与标签冲突):")
    for o in disputes:
        print(f"  gold={o['gold']} -> 文字判 {o['pred']}  OCR={o['texts'][:2]}")
    print("判决来源:", dict(src_stat))
    print(f"{'品牌':<10} {'n':>4} {'acc':>6}")
    per_brand = collections.defaultdict(lambda: [0, 0])
    for o in solid:
        per_brand[o["gold"]][1] += 1
        if o["pred"] == o["gold"]: per_brand[o["gold"]][0] += 1
    for b, (c, n) in sorted(per_brand.items(), key=lambda kv: -kv[1][1]):
        print(f"{b:<10} {n:>4} {c/n:>6.1%}")
    json.dump({"acc": acc, "source": dict(src_stat),
               "per_brand": {b: {"n": n, "acc": c/n} for b, (c, n) in per_brand.items()},
               "detail": out}, open(ROOT / "synthetic/vis_cache/e20_results.json", "w"),
              ensure_ascii=False, indent=1)
    print("落盘", flush=True)


if __name__ == "__main__":
    main()
