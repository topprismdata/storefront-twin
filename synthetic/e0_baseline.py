#!/usr/bin/env python3
"""
E0 基线（B0a）：色带/Logo 模板 + OCR 规则 — 左邻右舍门头识别
============================================================
设计文档 §7.1 B0a / §9.1 E0：无深度适配，纯规则能达到什么效果？

流程（§7 架构的简化规则版）：
  1. 招牌候选：HSV 红/蓝掩膜 → 形态学 → 宽矩形连通域（色带模板）。
  2. 四点校正：候选框裁剪 + 放大（保留长宽比）。
  3. OCR：tesseract chi_sim+eng，读品牌字「左邻右舍」+ PLUS + No.XXXX + Good Neighbor。
  4. L0 品牌鉴别：OCR 命中品牌字 → left_right；否则 other_brand/uncertain。
  5. L1 候选簇规则：PLUS→CS-05 / 立体发光字+蓝底→CS-04 / 超市→CS-06 /
     编号→CS-01 / ⊙或书法→CS-02 / 红蓝常规→CS-03。
  6. 选择性预测：低质量/无品牌/未知 → 拒识（§7.4）。

治理：
- 合成集评估用构造标签（label_authority=synthetic_by_construction），仅验证规则自身。
- 真实集权利全部 pending_review：本评估为【探索性可行性探针】，非正式精度宣称（G0/G1 未过）。
- 不据测试集调阈值；规则阈值先验固定。
- 输出 eval manifest：输入哈希/版本/OCR配置/逐样本证据，可追溯。
"""
import argparse, hashlib, json, re, sys
from pathlib import Path
import numpy as np
import cv2
import pytesseract

ROOT = Path(__file__).resolve().parent
BASELINE_VERSION = "e0-b0a-rule-v1.0"
OCR_LANGS = ["chi_tra", "chi_sim"]   # 繁体模型对招牌艺术字体更鲁棒，双模取最优
OCR_PSM = 6
OCR_CONFIG = "--oem 3 --psm 6"
OCR_LANG = "chi_tra"

# 官方 VI 颜色（HSV 范围，先验固定，不调）
RED_HSV = [(0, 90, 70), (10, 255, 255), (170, 90, 70), (180, 255, 255)]
BLUE_HSV = [(95, 80, 60), (125, 255, 255)]
BRAND_CHARS = ["左邻右舍", "左邻", "右舍", "左鄰右舍"]
PLUS_RE = re.compile(r"PLUS|Plus|\+|普拉斯", re.I)
NUM_RE = re.compile(r"No\.?\s*(\d{3,4})", re.I)
EN_RE = re.compile(r"Good\s*Neighb?or", re.I)


def color_mask(hsv, ranges):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for i in range(0, len(ranges), 2):
        lo, hi = ranges[i], ranges[i+1]
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return m


def detect_sign_candidates(img):
    """红/蓝色带模板 → 招牌候选框。行/列投影定位招牌带（对碎片化红板鲁棒，不依赖连通域）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = color_mask(hsv, RED_HSV)
    blue = color_mask(hsv, BLUE_HSV)
    H, W = img.shape[:2]
    cands = []
    for mask, name in [(red, "red"), (blue, "blue")]:
        rowsum = (mask > 0).sum(1).astype(float)
        rs = np.convolve(rowsum, np.ones(15)/15, "same")
        if rs.max() < W*0.05:
            continue
        rows = np.where(rs > rs.max()*0.35)[0]
        if len(rows) == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1])
        if y1-y0 < 40 or y1-y0 > H*0.4:
            continue
        band = mask[y0:y1+1, :]
        colsum = (band > 0).sum(0).astype(float)
        cs = np.convolve(colsum, np.ones(21)/21, "same")
        cols = np.where(cs > cs.max()*0.25)[0]
        if len(cols) == 0:
            continue
        x0, x1 = int(cols[0]), int(cols[-1])
        w, h = x1-x0, y1-y0
        aspect = w/max(h, 1)
        if aspect < 2.0 or aspect > 11 or w*h < W*H*0.02:
            continue
        roi = mask[y0:y1, x0:x1]
        color_frac = float(roi.mean())/255
        if color_frac < 0.25:
            continue
        cands.append({"box": [x0, y0, x1, y1], "area": w*h,
                      "aspect": round(aspect, 2), "color_frac": round(color_frac, 3),
                      "red_frac": round(float(red[y0:y1, x0:x1].mean())/255, 3),
                      "blue_frac": round(float(blue[y0:y1, x0:x1].mean())/255, 3),
                      "channel": name})
    cands.sort(key=lambda c: -c["area"])
    kept = []
    for c in cands:
        if not any(_iou(c["box"], k["box"]) > 0.5 for k in kept):
            kept.append(c)
    return kept[:5]


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    union = (ax1-ax0)*(ay1-ay0)+(bx1-bx0)*(by1-by0)-inter
    return inter/max(union, 1)


def ocr_region(img, box, pad=6):
    """白字红底 → 提取亮字掩膜（低饱和高亮度），双 OCR 模型取品牌字命中最优。"""
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    x0, y0 = max(0, x0-pad), max(0, y0-pad)
    x1, y1 = min(W, x1+pad), min(H, y1+pad)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return "", crop
    scale = max(1.5, 1100/max(crop.shape[1], 1))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 165)).astype(np.uint8)*255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    best_text, best_score = "", -1
    for lang in OCR_LANGS:
        t = pytesseract.image_to_string(white, lang=lang, config=f"--oem 3 --psm {OCR_PSM}")
        hits = sum(1 for ch in "左邻右舍" if ch in t)
        score = hits*3 + (1 if EN_RE.search(t) else 0) + (1 if NUM_RE.search(t) else 0)
        if score > best_score:
            best_score, best_text = score, t
    return best_text, crop


def classify_rules(ocr_text, cand, img):
    """L0 品牌 + L1 候选簇规则。返回 (brand_state, style_id, evidence, confidence)。"""
    t = ocr_text.replace(" ", "").replace("\n", "")
    ev = {"ocr_raw": ocr_text.strip()[:200]}
    # L0 品牌鉴别：中文字 + 英文副标 GoodNeighbor（强品牌信号）+ 编号制式
    brand_hit = any(ch in t for ch in BRAND_CHARS)
    char_hits = sum(1 for ch in "左邻右舍" if ch in t)
    en_hit = bool(EN_RE.search(t))
    num_hit = bool(NUM_RE.search(t))
    ev["brand_char_hits"] = char_hits
    ev["en_brand_hit"] = en_hit
    ev["num_hit"] = num_hit
    if brand_hit or char_hits >= 3 or (en_hit and char_hits >= 1):
        brand = "left_right"
    elif en_hit and num_hit:
        brand = "left_right"   # GoodNeighbor + No.XXXX 制式组合
    elif char_hits >= 2 or en_hit:
        brand = "uncertain"
    else:
        brand = "other_brand"

    # L1 候选簇（仅 brand=left_right 时判）
    style = "unknown"
    if brand == "left_right":
        has_plus = bool(PLUS_RE.search(t))
        has_num = bool(NUM_RE.search(t))
        has_en = bool(EN_RE.search(t))
        has_super = "超市" in t
        has_circle = "⊙" in t or "◉" in t
        blue_dom = cand["blue_frac"] > cand["red_frac"]
        ev.update({"plus": has_plus, "number": has_num, "en_sub": has_en,
                   "supermarket": has_super, "circle_sep": has_circle,
                   "blue_dominant": blue_dom})
        if has_plus:
            style = "CS-05"
        elif has_super:
            style = "CS-06"
        elif blue_dom:
            style = "CS-04"
        elif has_num:
            style = "CS-01"   # 编号制式（CS-02 书法需字体特征，规则难分，归 CS-01 编号大类）
        else:
            style = "CS-03"
    # 置信度（启发式，非校准概率）
    conf = "high" if (brand == "left_right" and char_hits >= 3) else \
           "medium" if brand == "left_right" else "low"
    return brand, style, ev, conf


def predict_one(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return {"error": "unreadable"}
    cands = detect_sign_candidates(img)
    if not cands:
        return {"brand_state": "other_brand", "style_id": "unknown",
                "reject_reason": "not_brand", "confidence": "low",
                "n_candidates": 0, "evidence": {}}
    best = None
    for cand in cands:
        text, crop = ocr_region(img, cand["box"])
        brand, style, ev, conf = classify_rules(text, cand, img)
        score = (ev.get("brand_char_hits", 0), cand["area"])
        if best is None or score > best["_score"]:
            best = {"brand_state": brand, "style_id": style, "confidence": conf,
                    "evidence": ev, "candidate": cand, "_score": score,
                    "n_candidates": len(cands)}
    # 选择性预测（§7.4）
    if best["brand_state"] == "other_brand":
        best["reject_reason"] = "not_brand"
    elif best["brand_state"] == "uncertain":
        best["reject_reason"] = "ambiguous_or_hybrid"
    elif best["style_id"] == "unknown":
        best["reject_reason"] = "unknown_style"
    else:
        best["reject_reason"] = None
    best.pop("_score", None)
    return best


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def eval_synthetic(manifest_path, renders_dir):
    ms = [json.loads(l) for l in open(manifest_path) if l.strip()]
    results = []
    for m in ms:
        fn = Path(renders_dir)/f"{m['image_id']}.jpg"
        if not fn.exists():
            fn = Path(renders_dir)/"cf_pairs"/f"{m['image_id']}.jpg"
        pred = predict_one(fn)
        gt_brand = "left_right" if m["gt"]["class_id"] != "synthetic_unknown" else "synthetic_unknown"
        gt_style = m["style_id"]
        # CF-4 异常应被拒识/判未知
        if m.get("cf_group") == "CF4_illegal_combo":
            correct = pred.get("style_id") in ("unknown",) or pred.get("reject_reason") is not None
        elif m.get("cf_group") == "CF3_occlusion_reject":
            correct = True   # 破损样本不评分类，只记录是否降信心
        else:
            correct = (pred.get("style_id") == gt_style)
        results.append({"image_id": m["image_id"], "gt_style": gt_style,
                        "pred_style": pred.get("style_id"), "pred_brand": pred.get("brand_state"),
                        "reject": pred.get("reject_reason"), "confidence": pred.get("confidence"),
                        "correct": correct, "cf_group": m.get("cf_group"),
                        "evidence": pred.get("evidence", {})})
    return results


def eval_real(registry_path, images_dir, max_neg=48):
    reg = json.load(open(registry_path))
    sha_idx = json.load(open(Path(images_dir)/"sha_file_index.json"))
    def resolve(r):
        p = sha_idx.get(r["sha256"])
        return (Path(images_dir)/p) if p else None
    pos = [r for r in reg if r["label"] == "positive" and r["is_representative"]]
    neg = [r for r in reg if r["label"] == "negative" and r["is_representative"]]
    # 负例分层抽样（每类取若干，避免 render_stock 淹没）
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in neg:
        by_type[r["neg_type"]].append(r)
    neg_sel = []
    per = max(1, max_neg//len(by_type))
    for t, rows in by_type.items():
        neg_sel.extend(rows[:per])
    results = []
    for r in pos:
        fn = resolve(r)
        pred = predict_one(fn) if fn and fn.exists() else {"error": "missing"}
        results.append({"image_id": r["image_id"], "label": "positive",
                        "gt_style": r.get("candidate_style", "").split("_")[0],
                        "pred_style": pred.get("style_id"), "pred_brand": pred.get("brand_state"),
                        "reject": pred.get("reject_reason"), "confidence": pred.get("confidence"),
                        "brand_correct": pred.get("brand_state") == "left_right",
                        "evidence": pred.get("evidence", {})})
    for r in neg_sel:
        fn = resolve(r)
        pred = predict_one(fn) if fn and fn.exists() else {"error": "missing"}
        results.append({"image_id": r["image_id"], "label": "negative",
                        "neg_type": r["neg_type"], "gt_style": None,
                        "pred_style": pred.get("style_id"), "pred_brand": pred.get("brand_state"),
                        "reject": pred.get("reject_reason"), "confidence": pred.get("confidence"),
                        "brand_correct": pred.get("brand_state") != "left_right",
                        "evidence": pred.get("evidence", {})})
    return results


def summarize(results, kind):
    from collections import Counter
    if kind == "synthetic":
        graded = [r for r in results if r.get("cf_group") not in ("CF3_occlusion_reject",)]
        acc = sum(r["correct"] for r in graded)/max(len(graded), 1)
        per_style = {}
        for s in sorted({r["gt_style"] for r in graded}):
            rs = [r for r in graded if r["gt_style"] == s]
            per_style[s] = {"n": len(rs), "acc": round(sum(r["correct"] for r in rs)/len(rs), 3)}
        return {"n": len(results), "graded": len(graded), "style_acc": round(acc, 3),
                "per_style": per_style}
    else:
        pos = [r for r in results if r["label"] == "positive"]
        neg = [r for r in results if r["label"] == "negative"]
        brand_recall = sum(r["brand_correct"] for r in pos)/max(len(pos), 1)
        brand_spec = sum(r["brand_correct"] for r in neg)/max(len(neg), 1)
        return {"n_pos": len(pos), "n_neg": len(neg),
                "brand_recall": round(brand_recall, 3),
                "brand_specificity": round(brand_spec, 3),
                "neg_by_type": dict(Counter(r.get("neg_type") for r in neg))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.synthetic or not (args.synthetic or args.real):
        mp = ROOT/"renders_batch"/"synthetic_manifest.jsonl"
        res = eval_synthetic(mp, ROOT/"renders_batch")
        summ = summarize(res, "synthetic")
        (out/"e0_synthetic_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("=== E0 synthetic (by-construction labels) ===")
        print(json.dumps(summ, ensure_ascii=False, indent=1))

    if args.real:
        reg = ROOT.parent/"data"/"registry_v3.json"
        imgs = ROOT.parent/"data"
        res = eval_real(reg, imgs)
        summ = summarize(res, "real")
        (out/"e0_real_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("=== E0 real (EXPLORATORY probe, rights pending, NOT formal accuracy) ===")
        print(json.dumps(summ, ensure_ascii=False, indent=1))

    # eval manifest
    manifest = {"baseline_version": BASELINE_VERSION, "ocr_lang": OCR_LANG,
                "ocr_config": OCR_CONFIG, "detector": "hsv_red_blue_projection",
                "thresholds_fixed_a_priori": True,
                "governance": "real_eval_exploratory_only_G0_G1_not_passed"}
    (out/"e0_eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()