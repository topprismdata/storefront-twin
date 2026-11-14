#!/usr/bin/env python3
"""e13 — 招牌区域检测器 (ROI 生产化前置)。
监督: renders_3d 的 sign_polygon (合成, 自建, split=synthetic_train_only)。
方法: 8×6 瓦片网格, CLIP 瓦片嵌入 + 逻辑回归 → 瓦片热力 → 连通域合并为框。
评测: 合成留出渲染 IoU (定量) + ROI 原型 14 样本真实图覆盖 (定性, 含 img_0003/0012 固定裁剪失败案例)。
"""
import json, sys, random
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import visual_baseline as vb

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
OUT = ROOT / "synthetic/vis_cache/e13_sign"
OUT.mkdir(parents=True, exist_ok=True)
COLS, ROWS, THRESH = 8, 6, 0.30


def hull(pts):
    """单调链凸包 — sign_polygon 顶点顺序是环绕框, ray casting 需先取凸包。"""
    pts = sorted(map(tuple, pts))
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    up = []
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def tile_label(poly, ci, ri):
    """凸包(归一化点列)与瓦片覆盖采样比。"""
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    x0, x1 = ci / COLS, (ci + 1) / COLS
    y0, y1 = ri / ROWS, (ri + 1) / ROWS
    import numpy as np
    hit = 0
    pts = [(x0 + (i + 0.5) / 5 * (x1 - x0), y0 + (j + 0.5) / 5 * (y1 - y0)) for i in range(5) for j in range(5)]
    H = hull(poly)
    px = np.array([p[0] for p in H]); py = np.array([p[1] for p in H])
    n = len(H)
    for (qx, qy) in pts:
        inside = False
        for k in range(n):
            x1p, y1p, x2p, y2p = px[k], py[k], px[(k + 1) % n], py[(k + 1) % n]
            if (y1p > qy) != (y2p > qy) and qx < (x2p - x1p) * (qy - y1p) / (y2p - y1p + 1e-12) + x1p:
                inside = not inside
        hit += inside
    return hit / 25


def tiles_of(im, embedder):
    w, h = im.size
    vecs, metas = [], []
    for ri in range(ROWS):
        for ci in range(COLS):
            box = (int(ci * w / COLS), int(ri * h / ROWS), int((ci + 1) * w / COLS), int((ri + 1) * h / ROWS))
            t = im.crop(box)
            vecs.append(embedder.embed_pil([t])[0])
            metas.append((ci, ri))
    return np.array(vecs), metas


def merge_tiles(mask):
    """瓦片掩码 → 最小外接框(归一化)。空则 None。"""
    idx = np.where(mask.reshape(ROWS, COLS))
    if len(idx[0]) == 0:
        return None
    r0, r1 = idx[0].min(), idx[0].max() + 1
    c0, c1 = idx[1].min(), idx[1].max() + 1
    return [c0 / COLS, r0 / ROWS, c1 / COLS, r1 / ROWS]


def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0)); iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    u = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(1e-9, u)


def iou_box_poly(box, poly):
    """归一化 box [x0,y0,x1,y1] vs sign_polygon(凸包) 的 IoU (栅格采样 200x200)。"""
    H = hull(poly)
    px = np.array([p[0] for p in H]); py = np.array([p[1] for p in H])
    n = len(H)
    x0, y0, x1, y1 = box
    N = 200
    gx, gy = np.meshgrid(np.linspace(0, 1, N), np.linspace(1, 0, N))
    inside = np.zeros((N, N), dtype=bool)
    for i in range(n):
        x1p, y1p = px[i], py[i]
        x2p, y2p = px[(i + 1) % n], py[(i + 1) % n]
        cond = ((y1p > gy) != (y2p > gy)) & (gx < (x2p - x1p) * (gy - y1p) / (y2p - y1p + 1e-12) + x1p)
        inside ^= cond
    inb = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
    inter = (inside & inb).sum()
    union = (inside | inb).sum()
    return float(inter / max(1, union))


def main():
    rows = [json.loads(l) for l in open(ROOT / "synthetic/renders_3d/manifest_3d.jsonl")]
    rows = [r for r in rows if r.get("sign_polygon")]
    random.Random(2026).shuffle(rows)
    n_hold = max(20, len(rows) // 10)
    hold, train = rows[:n_hold], rows[n_hold:]
    print(f"renders: train {len(train)}, hold {len(hold)}", flush=True)
    emb = vb.Embedder("clip")

    def build(chunk):
        X, Y, G = [], [], []
        for k, r in enumerate(chunk):
            im = Image.open(ROOT / "synthetic/renders_3d" / r["image"]).convert("RGB")
            V, metas = tiles_of(im, emb)
            poly = r["sign_polygon"]
            for (ci, ri), v in zip(metas, V):
                X.append(v)
                Y.append(1 if tile_label(poly, ci, ri) >= THRESH else 0)
            if k % 100 == 0:
                print(f"  embed {k}/{len(chunk)}", flush=True)
        return np.array(X), np.array(Y)

    Xtr, Ytr = build(train)
    pos, neg = Xtr[Ytr == 1], Xtr[Ytr == 0]
    rng = np.random.default_rng(0)
    neg = neg[rng.choice(len(neg), min(len(pos) * 3, len(neg)), replace=False)]
    Xb = np.vstack([pos, neg]); Yb = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0).fit(Xb, Yb)
    print(f"train tiles: pos {len(pos)}, neg {len(neg)}", flush=True)

    # 合成留出 IoU
    ious = []
    for r in hold:
        im = Image.open(ROOT / "synthetic/renders_3d" / r["image"]).convert("RGB")
        V, metas = tiles_of(im, emb)
        mask = clf.predict_proba(V)[:, 1] > 0.5
        det = merge_tiles(mask)
        poly = np.array(r["sign_polygon"])
        gt = [poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()]
        if det:
            ious.append(iou(det, gt))
    ious = np.array(ious)
    print(f"合成留出 IoU: mean {ious.mean():.3f} median {np.median(ious):.3f} P25 {np.percentile(ious,25):.3f} n={len(ious)}", flush=True)

    # 真实 14 样本: 检测框 vs 固定裁剪覆盖率
    meta = json.load(open(ROOT / "synthetic/vis_cache/e10_roi/meta.json"))
    real_out = {}
    for iid, m in meta.items():
        im = Image.open(m["full"]).convert("RGB")
        w, h = im.size
        V, _ = tiles_of(im, emb)
        mask = clf.predict_proba(V)[:, 1] > 0.5
        det = merge_tiles(mask)
        fixed = [0.10, 0.0, 0.90, 0.45]
        cov_det = iou(det, fixed) if det else 0.0
        real_out[iid] = {"det_box_norm": det, "iou_vs_fixedcrop": round(cov_det, 3)}
        tag = " ←固定裁剪失败案例" if iid in ("img_0003", "img_0012") else ""
        print(f"  {iid}: det={None if not det else [round(x,2) for x in det]} iou_vs_fixed={cov_det:.2f}{tag}", flush=True)
    json.dump({"holdout_iou": {"mean": float(ious.mean()), "median": float(np.median(ious)), "n": int(len(ious))},
               "real": real_out, "config": {"cols": COLS, "rows": ROWS, "thresh": THRESH}},
              open(OUT / "sign_detector_results.json", "w"), ensure_ascii=False, indent=1)
    np.savez(OUT / "sign_clf.npz", coef=clf.coef_, intercept=clf.intercept_,
             classes=clf.classes_, mean=clf.mean_ if hasattr(clf, "mean_") else np.zeros(1))
    print("落盘", flush=True)


if __name__ == "__main__":
    main()
