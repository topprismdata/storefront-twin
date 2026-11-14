#!/usr/bin/env python3
"""e15 — 现成文本检测器(PP-OCRv4 DBNet ONNX)作招牌区定位器评测。
回应 GPT 评审 D 项: 定位器必须有独立参照的定量指标。
评测: (a) 合成留出 78 渲染, sign_polygon(凸包)为 GT -> det-union IoU;
      (b) 真实 14 ROI 样本: 与固定裁剪对比, 重点 img_0003/0012(固定裁剪失败案例)。"""
import json, sys
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from PIL import Image

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import e13_sign_detector as e13  # 复用 hull/merge_tiles/iou

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sess = ort.InferenceSession(str(ROOT / "models/rapidocr/ch_PP-OCRv4_det_infer.onnx"),
                            providers=["CPUExecutionProvider"])
INAME = sess.get_inputs()[0].name

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def det_union_box(bgr, thr=0.3, max_side=960):
    """DBNet 前向 + 二值化 + 连通域 -> 所有文本框并集(归一化 xyxy); 无检测返回 None。"""
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(bgr, (int(w * scale // 32 * 32), int(h * scale // 32 * 32)))
    else:
        img = cv2.resize(bgr, (max(32, w // 32 * 32), max(32, h // 32 * 32)))
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]
    out = sess.run(None, {INAME: x})[0][0][0]  # (H,W) prob map
    H, W = out.shape
    binm = (out > thr).astype(np.uint8) * 255
    binm = cv2.dilate(binm, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    H0, W0 = bgr.shape[:2]
    xs0, ys0, xs1, ys1 = [], [], [], []
    for c in cnts:
        if cv2.contourArea(c) < 8:
            continue
        bx = cv2.boundingRect(c)
        xs0.append(bx[0] / W * W0); ys0.append(bx[1] / H * H0)
        xs1.append((bx[0] + bx[2]) / W * W0); ys1.append((bx[1] + bx[3]) / H * H0)
    if not xs0:
        return None
    return [min(xs0) / W0, min(ys0) / H0, max(xs1) / W0, max(ys1) / H0]


def main():
    # (a) 合成留出定量
    man = [json.loads(l) for l in open(ROOT / "synthetic/renders_3d/manifest_3d.jsonl")]
    import random
    random.seed(2026)
    idx = list(range(len(man)))
    random.shuffle(idx)
    hold = idx[-78:]
    ious = []
    for i in hold:
        r = man[i]
        fp = ROOT / "synthetic/renders_3d" / r["image"]
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        d = det_union_box(bgr)
        poly = r["sign_polygon"]
        if d is None:
            ious.append(0.0)
            continue
        # det box -> 采样点, polygon GT 用凸包
        ious.append(e13.iou_box_poly(d, poly))
    ious = np.array(ious)
    print(f"合成留出 n={len(ious)}: DBNet det-union vs sign_polygon IoU")
    print(f"  mean {ious.mean():.3f} median {np.median(ious):.3f} P25 {np.percentile(ious,25):.3f} 检出率 {(ious>0.1).mean():.3f}")

    # 固定裁剪基线对照(同一批渲染)
    fix_ious = []
    for i in hold[:30]:
        r = man[i]
        fix = [0.0, 0.08, 1.0, 0.45]  # 项目固定上中部带
        fix_ious.append(e13.iou_box_poly(fix, r["sign_polygon"]))
    fix_ious = np.array(fix_ious)
    print(f"  对照固定带 n={len(fix_ious)}: mean {fix_ious.mean():.3f} median {np.median(fix_ious):.3f}")

    # (b) 真实 14
    meta = json.load(open(ROOT / "synthetic/vis_cache/e10_roi/meta.json"))
    print("\n真实 14 ROI 样本 (det_union vs 固定带):")
    for iid, m in meta.items():
        bgr = cv2.imread(m["full"])
        if bgr is None:
            print(f"  {iid}: 读图失败")
            continue
        d = det_union_box(bgr)
        tag = " <-固定裁剪失败案例" if iid in ("img_0003", "img_0012") else ""
        if d is None:
            print(f"  {iid}: 无检测{tag}")
            continue
        print(f"  {iid}: det=[{d[0]:.2f},{d[1]:.2f},{d[2]:.2f},{d[3]:.2f}]{tag}")


if __name__ == "__main__":
    main()
