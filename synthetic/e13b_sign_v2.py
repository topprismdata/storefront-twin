#!/usr/bin/env python3
"""e13b — 招牌检测器 v2: 合成多边形 + 真实图 OCR 伪标签混训。
v1 负结果: 纯合成监督在真实图上退化为全图框(域差)。v2 用 tesseract(chi_sim+eng, psm 11)
文本框给真实正例瓦片打正标; 负例仍取自合成(避免 OCR 漏检产生假负例)。
评测: 合成留出 IoU + 真实 14 样本覆盖(含 img_0003/0012 固定裁剪失败案例)。
"""
import json, sys, random, subprocess, tempfile
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import visual_baseline as vb
import e13_sign_detector as e13

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
OUT = ROOT / "synthetic/vis_cache/e13_sign"
N_REAL = 250


def ocr_boxes(im):
    """真实图 OCR → 归一化文本框列表 (上半图, 置信度过滤)。"""
    w, h = im.size
    crop = im.crop((0, 0, w, max(1, int(h * 0.6))))
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        crop.save(f.name)
        try:
            out = subprocess.run(["tesseract", f.name, "stdout", "-l", "chi_sim+eng", "--psm", "11", "tsv"],
                                 capture_output=True, text=True, timeout=30).stdout
        except Exception:
            return []
    boxes = []
    for line in out.splitlines()[1:]:
        p = line.split("\t")
        if len(p) < 12:
            continue
        try:
            conf = float(p[10])
        except ValueError:
            continue
        text = p[11].strip()
        if conf < 55 or len(text) < 1:
            continue
        x, y, bw, bh = int(p[6]), int(p[7]), int(p[8]), int(p[9])
        if bw < 8 or bh < 6:
            continue
        boxes.append([(x) / w, (y) / h, (x + bw) / w, (y + bh) / h, text, conf])
    return boxes


def main():
    reg = json.load(open(ROOT / "data/registry_v11.json"))
    sha_idx = json.load(open(ROOT / "data/sha_file_index.json"))
    pos_rows = []
    for r in reg:
        if r["label"] == "positive" and r.get("is_representative"):
            p = sha_idx.get(r["sha256"])
            if p and (ROOT / "data" / p).exists():
                pos_rows.append(ROOT / "data" / p)
    random.Random(2026).shuffle(pos_rows)
    pos_rows = pos_rows[:N_REAL]
    print(f"real positives: {len(pos_rows)}", flush=True)
    emb = vb.Embedder("clip")

    # 合成训练瓦片 (v1 同款)
    renders = [json.loads(l) for l in open(ROOT / "synthetic/renders_3d/manifest_3d.jsonl")]
    renders = [r for r in renders if r.get("sign_polygon")]
    random.Random(2026).shuffle(renders)
    n_hold = max(20, len(renders) // 10)
    hold, train = renders[:n_hold], renders[n_hold:]

    X, Y = [], []
    for k, r in enumerate(train):
        im = Image.open(ROOT / "synthetic/renders_3d" / r["image"]).convert("RGB")
        V, metas = e13.tiles_of(im, emb)
        for (ci, ri), v in zip(metas, V):
            X.append(v); Y.append(1 if e13.tile_label(r["sign_polygon"], ci, ri) >= e13.THRESH else 0)
        if k % 150 == 0:
            print(f"  synth embed {k}/{len(train)}", flush=True)
    X_syn, Y_syn = np.array(X), np.array(Y)
    print(f"synth tiles: {len(X_syn)} pos {int(Y_syn.sum())}", flush=True)

    # 真实 OCR 瓦片
    n_ocr_hit = 0
    Xr = []
    for k, fp in enumerate(pos_rows):
        im = Image.open(fp).convert("RGB")
        w, h = im.size
        boxes = ocr_boxes(im)
        if boxes:
            n_ocr_hit += 1
        V, metas = e13.tiles_of(im, emb)
        for idx, (ci, ri) in enumerate(metas):
            x0, x1 = ci / e13.COLS, (ci + 1) / e13.COLS
            y0, y1 = ri / e13.ROWS, (ri + 1) / e13.ROWS
            cover = 0.0
            for b in boxes:
                ix = max(0, min(x1, b[2]) - max(x0, b[0])); iy = max(0, min(y1, b[3]) - max(y0, b[1]))
                cover = max(cover, ix * iy / max(1e-9, (x1 - x0) * (y1 - y0)))
            if cover >= 0.35:
                Xr.append(V[idx])
        if k % 50 == 0:
            print(f"  real {k}/{len(pos_rows)} (ocr_hit {n_ocr_hit})", flush=True)
    Xr = np.array(Xr)
    print(f"real OCR-positive tiles: {len(Xr)} from {n_ocr_hit} imgs", flush=True)

    # 负例: 合成负瓦片 3×(正例总数)
    neg = X_syn[Y_syn == 0]
    n_pos_total = int(Y_syn.sum()) + len(Xr)
    rng = np.random.default_rng(0)
    neg = neg[rng.choice(len(neg), min(n_pos_total * 3, len(neg)), replace=False)]
    Xb = np.vstack([X_syn[Y_syn == 1], Xr, neg])
    Yb = np.concatenate([np.ones(int(Y_syn.sum()) + len(Xr)), np.zeros(len(neg))])
    clf = e13.LogisticRegression(max_iter=2000, C=1.0, random_state=0).fit(Xb, Yb)
    print(f"train: pos {n_pos_total}, neg {len(neg)}", flush=True)

    ious = []
    for r in hold:
        im = Image.open(ROOT / "synthetic/renders_3d" / r["image"]).convert("RGB")
        V, _ = e13.tiles_of(im, emb)
        det = e13.merge_tiles(clf.predict_proba(V)[:, 1] > 0.5)
        poly = np.array(r["sign_polygon"])
        gt = [poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()]
        if det:
            ious.append(e13.iou(det, gt))
    ious = np.array(ious)
    print(f"合成留出 IoU: mean {ious.mean():.3f} median {np.median(ious):.3f}", flush=True)

    meta = json.load(open(ROOT / "synthetic/vis_cache/e10_roi/meta.json"))
    real_out = {}
    for iid, m in meta.items():
        im = Image.open(m["full"]).convert("RGB")
        V, _ = e13.tiles_of(im, emb)
        det = e13.merge_tiles(clf.predict_proba(V)[:, 1] > 0.5)
        real_out[iid] = det
        tag = " ←固定裁剪失败案例" if iid in ("img_0003", "img_0012") else ""
        print(f"  {iid}: det={None if det is None else [round(float(x), 2) for x in det]}{tag}", flush=True)
    json.dump({"holdout_iou": {"mean": float(ious.mean()), "median": float(np.median(ious)), "n": int(len(ious))},
               "real": {k: (None if v is None else [round(float(x), 3) for x in v]) for k, v in real_out.items()},
               "config": {"cols": e13.COLS, "rows": e13.ROWS, "ocr_imgs": len(pos_rows), "ocr_hit": n_ocr_hit}},
              open(OUT / "sign_detector_v2_results.json", "w"), ensure_ascii=False, indent=1)
    np.savez(OUT / "sign_clf_v2.npz", coef=clf.coef_, intercept=clf.intercept_)
    print("落盘", flush=True)


if __name__ == "__main__":
    main()
