#!/usr/bin/env python3
"""
E6-v2 — 修域缺口: v1 的 FP 23/31 是 render_stock(池内无此负例型)。
v2 池(与评测330 sha 级零重叠,已程序化排除):
  pos = 爬取池 A 1199 + registry 非330正例 6
  neg = 爬取池 B/C 394 + registry 非330负例 163(138 render_stock!) + renders_3d 合成负例 436
协议同 v1: 阈值池内 GroupKFold OOF 选,评测 330 只打分。
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import e6_real_probe as e1
import visual_baseline as vb

ROOT = e1.ROOT
DATA, CACHE, E6 = e1.DATA, e1.CACHE, e1.E6


def build_pool_v2():
    rows = e1.build_pool()  # 1593 (1199A+394neg)
    # registry 非330 行(sha 级排除)
    r3 = json.load(open(DATA / "registry_v3.json"))
    tk = json.load(open(CACHE / "e10_full/truth_key.json"))
    tr = tk if isinstance(tk, list) else list(tk.values())
    ev_shas = set()
    byid = {r["image_id"]: r for r in r3}
    for r in tr:
        if r["image_id"] in byid:
            ev_shas.add(byid[r["image_id"]]["sha256"])
    sha_idx = json.load(open(DATA / "sha_file_index.json"))
    extra = [r for r in r3 if r["sha256"] not in ev_shas and r["sha256"] in sha_idx]
    for r in extra:
        fp = DATA / sha_idx[r["sha256"]]
        if not fp.exists():
            continue
        neg = r["label"] != "positive"
        rows.append({"image_id": r["image_id"], "path": str(fp), "poi": "reg_" + r["image_id"],
                     "lab": "N" if neg else "A", "src": "registry_extra",
                     "province": r.get("geo_bucket", "")})
    # renders_3d 负例
    mf = ROOT / "synthetic/renders_3d/manifest_3d.jsonl"
    for line in open(mf):
        r = json.loads(line)
        if r.get("brand") == "left_right":
            continue  # 只要合成负例
        fp = ROOT / "synthetic/renders_3d" / r["image"]
        if not fp.exists():
            continue
        rows.append({"image_id": "s3d_" + r["image"], "path": str(fp), "poi": "s3d_" + (r.get("style_id") or "x"),
                     "lab": "N", "src": "render3d", "province": "synth"})
    return rows


def embed_extra(rows):
    """只嵌 registry_extra + render3d 两源,缓存独立。"""
    from PIL import Image
    import cv2
    sub = [r for r in rows if r["src"] in ("registry_extra", "render3d")]
    for name in ("clip", "dinov2"):
        emb = vb.Embedder(name)
        for view in ("full", "crop"):
            cf = E6 / f"x_{name}_{view}.npz"
            if cf.exists():
                print(f"skip x {name}/{view}")
                continue
            vecs, ids, batch, bids = [], [], [], []

            def flush():
                if batch:
                    vecs.append(emb.embed_pil(batch))
                    ids.extend(bids)
                    batch.clear(); bids.clear()

            for i, r in enumerate(sub):
                img = cv2.imread(r["path"])
                if img is None:
                    continue
                if view == "crop":
                    x0, y0, x1, y1 = vb.sign_crop_box(img)
                    img = img[y0:y1, x0:x1]
                batch.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                bids.append(r["image_id"])
                if len(batch) >= 16:
                    flush()
            flush()
            V = np.concatenate(vecs, 0)
            np.savez(cf, vecs=V, ids=np.array(ids))
            print(f"saved x {name}/{view} {V.shape}", flush=True)
        del emb
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def load_feats_v2(rows):
    cols = []
    m = {}
    for name in ("clip", "dinov2"):
        for view in ("full", "crop"):
            z1 = np.load(E6 / f"pool_{name}_{view}.npz", allow_pickle=True)
            m[(name, view)] = dict(zip(z1["ids"].tolist(), z1["vecs"]))
    for name in ("clip", "dinov2"):
        for view in ("full", "crop"):
            z2 = np.load(E6 / f"x_{name}_{view}.npz", allow_pickle=True)
            m[(name, view)].update(zip(z2["ids"].tolist(), z2["vecs"]))
    for name in ("clip", "dinov2"):
        for view in ("full", "crop"):
            cols.append(np.stack([m[(name, view)][r["image_id"]] for r in rows]))
    return np.concatenate(cols, 1).astype(np.float32)


def _readable(rows, m):
    keep = [r for r in rows if all(r["image_id"] in m[(n, v)] for n in ("clip", "dinov2") for v in ("full", "crop"))]
    drop = len(rows) - len(keep)
    if drop:
        print(f"drop {drop} unreadable imgs")
    return keep


def train_v2(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    m = {}
    for name in ("clip", "dinov2"):
        for view in ("full", "crop"):
            z1 = np.load(E6 / f"pool_{name}_{view}.npz", allow_pickle=True)
            m[(name, view)] = dict(zip(z1["ids"].tolist(), z1["vecs"]))
    for name in ("clip", "dinov2"):
        for view in ("full", "crop"):
            z2 = np.load(E6 / f"x_{name}_{view}.npz", allow_pickle=True)
            m[(name, view)].update(zip(z2["ids"].tolist(), z2["vecs"]))
    rows = _readable(rows, m)
    X = load_feats_v2(rows)
    y = np.array([1 if r["lab"] == "A" else 0 for r in rows])
    g = np.array([r["poi"] for r in rows])
    print(f"v2 pool X={X.shape} pos={y.sum()} neg={len(y)-y.sum()} groups={len(set(g))}")
    best = None
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        thrs = np.unique(np.quantile(oof, np.linspace(0.02, 0.98, 97)))
        f1s = []
        for t in thrs:
            p = oof > t
            tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
            f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
        i = int(np.argmax(f1s))
        print(f"C={C}: OOF F1={f1s[i]:.3f} thr={thrs[i]:.4f}")
        if best is None or f1s[i] > best[1]:
            best = (C, f1s[i], float(thrs[i]))
    clf = LogisticRegression(max_iter=2000, C=best[0], class_weight="balanced").fit(X, y)
    np.savez(E6 / "e6v2_probe.npz", coef=clf.coef_, intercept=clf.intercept_,
             thr=best[2], C=best[0], oof_f1=best[1])
    print(f"FINAL C={best[0]} thr={best[2]:.4f} OOF_F1={best[1]:.3f}")


def eval_v2(rows):
    from sklearn.metrics import average_precision_score
    z = np.load(E6 / "e6v2_probe.npz")
    w, b, thr = z["coef"][0].astype(np.float64), float(z["intercept"]), float(z["thr"])
    tk = json.load(open(CACHE / "e10_full/truth_key.json"))
    tr = tk if isinstance(tk, list) else list(tk.values())
    m = {}
    for name in ("clip", "dinov2"):
        for view in ("full", "crop"):
            zz = np.load(CACHE / f"{name}_{view}.npz", allow_pickle=True)
            m[(name, view)] = dict(zip(zz["ids"].tolist(), zz["vecs"]))
    cols = [np.stack([m[(n, v)][r["image_id"]] for r in tr])
            for n in ("clip", "dinov2") for v in ("full", "crop")]
    X = np.concatenate(cols, 1).astype(np.float64)
    s = X @ w + b
    y = np.array([1 if r["label"] == "positive" else 0 for r in tr])
    pred = s > thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F1 = 2 * P * R / max(1e-9, P + R)
    aupr = average_precision_score(y, s)
    groups = np.array([r["group"] for r in tr])
    rng = np.random.default_rng(2026)
    ug = np.unique(groups); f1b = []
    for _ in range(2000):
        sg = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sg])
        yy, pp = y[idx], pred[idx]
        t = int((pp & (yy == 1)).sum()); f = int((pp & (yy == 0)).sum()); n2 = int((~pp & (yy == 1)).sum())
        f1b.append(2 * t / max(1, 2 * t + f + n2))
    lo, hi = np.quantile(f1b, [0.025, 0.975])
    e6fp = [r["image_id"] for r, si, pi in zip(tr, s, pred) if pi and r["label"] != "positive"]
    fpt = Counter(r["neg_type"] for r in tr if r["image_id"] in set(e6fp))
    out = {"e6v2": {"P": round(P, 3), "R": round(R, 3), "F1": round(F1, 3), "AuPR": round(float(aupr), 3),
                    "confusion": [tp, fp, fn], "thr_poolOOF": thr, "F1_CI95": [round(float(lo), 3), round(float(hi), 3)],
                    "FP_by_negtype": dict(fpt)},
           "e8g_reference": {"P": 0.778, "R": 0.933, "F1": 0.848, "AuPR": 0.926, "confusion": [28, 8, 2]}}
    json.dump(out, open(E6 / "e6v2_eval.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    rows = build_pool_v2()
    print(f"v2池: {len(rows)} | {Counter(r['lab'] for r in rows)} | {Counter(r['src'] for r in rows)}")
    embed_extra(rows)
    train_v2(rows)
    eval_v2(rows)
