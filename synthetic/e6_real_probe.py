#!/usr/bin/env python3
"""
E6 — 真实数据末层微调（弱标签）— 左邻右舍门头识别
==================================================
用户指令 2026-11-12：权限闸已过，解锁真实数据训练。
协议：
- 训练池：爬取候选弱标签（POI级/图级人工-VLM核验裁决）
    pos = A（真门头，校准精度~94-96%）
    neg = B(室内/非门头) + C(同名异业/他牌，难负例)
    D(不清) 剔除——标签歧义
- 评测集：registry 330（全分辨率人工核验，门店级与池零重叠，已验证）
- 泄漏防护：训练阈值只在池内 GroupKFold(poi) OOF 上选；评测 330 只打分不调参
- 路线：冻结骨干四路嵌入(clip/dino × full/crop) concat → 逻辑回归探针
对比：E8-G 冻结融合（truth_key 内 e8g_score/e8g_pred）
"""
import argparse, json, os, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
DATA = ROOT / "data"
CACHE = ROOT / "synthetic" / "vis_cache"
sys.path.insert(0, str(ROOT / "synthetic"))
import visual_baseline as vb  # Embedder, sign_crop_box

E6 = CACHE / "e6"
E6.mkdir(exist_ok=True, parents=True)


def build_pool():
    """训练池：gen_full(PDI级裁决) + gen_x1(图级裁决)。"""
    rows = []
    # POI级裁决: verdicts.jsonl 逐格(运行索引) × grid_index.json(索引→poi_id)
    poi2v = {}
    for line in open(DATA / "gen_verify/verdicts.jsonl"):
        for k, v in json.loads(line)["verdicts"].items():
            poi2v[int(k)] = v
    gi = json.load(open(DATA / "gen_verify/grid_index.json"))
    vs1 = {e["poi_id"]: poi2v[e["grid_id"]] for e in gi}
    covers = {e["file"] for e in gi}  # 仅收已判封面图,兄弟图未判不入池
    gf = json.load(open(DATA / "gen_full/gen_candidates.json"))
    for r in gf:
        v = vs1.get(r["poi_id"])
        if v not in ("A", "B", "C"):
            continue  # D 或未核验剔除
        fn = r["file"] if r["file"].startswith("gen_") else f"gen_{r['file']}"
        if fn not in covers:
            continue  # 未判兄弟图不入池
        fp = DATA / "gen_full" / fn
        if not fp.exists():
            continue
        rows.append({"image_id": r["image_id"], "path": str(fp), "poi": r["poi_id"],
                     "lab": v, "src": "gen_full", "province": r.get("province", "")})
    allv = {}
    for fn in [f"verdicts_b{i}.jsonl" for i in range(1, 6)]:
        for line in open(DATA / "gen_x1/verify" / fn):
            for k, v in json.loads(line)["v"].items():
                allv[k] = v
    gx = json.load(open(DATA / "gen_x1/merged_records.json"))["records"]
    for r in gx:
        v = allv.get(r["image_id"])
        if v not in ("A", "B", "C"):
            continue
        fp = DATA / "gen_x1" / r.get("file", "")
        if not fp.exists():
            continue
        rows.append({"image_id": r["image_id"], "path": str(fp), "poi": r["poi_id"],
                     "lab": v, "src": "gen_x1", "province": r.get("province", "")})
    return rows


def embed_pool(rows):
    """四路嵌入，缓存 e6/pool_{model}_{view}.npz。"""
    from PIL import Image
    import cv2
    for name in ("clip", "dinov2"):
        emb = vb.Embedder(name)
        for view in ("full", "crop"):
            cf = E6 / f"pool_{name}_{view}.npz"
            if cf.exists():
                print(f"skip {name}/{view}")
                continue
            vecs, ids = [], []
            batch, bids = [], []

            def flush():
                if batch:
                    vecs.append(emb.embed_pil(batch))
                    ids.extend(bids)
                    batch.clear(); bids.clear()

            for i, r in enumerate(rows):
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
                if (i + 1) % 200 == 0:
                    print(f"  {name}/{view} {i+1}/{len(rows)}", flush=True)
            flush()
            V = np.concatenate(vecs, 0)
            np.savez(cf, vecs=V, ids=np.array(ids))
            print(f"saved {name}/{view} {V.shape}", flush=True)
        del emb
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def load_pool_feats(rows):
    feats, keep = [], None
    for name in ("clip", "dinov2"):
        for view in ("full", "crop"):
            z = np.load(E6 / f"pool_{name}_{view}.npz", allow_pickle=True)
            m = dict(zip(z["ids"].tolist(), z["vecs"]))
            col = np.stack([m[r["image_id"]] for r in rows])
            feats.append(col)
    return np.concatenate(feats, 1).astype(np.float32)


def train(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = load_pool_feats(rows)
    y = np.array([1 if r["lab"] == "A" else 0 for r in rows])
    g = np.array([r["poi"] for r in rows])
    print(f"pool X={X.shape} pos={y.sum()} neg={len(y)-y.sum()} groups={len(set(g))}")
    best = None
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        # thr 扫描 maximize F1
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
    C = best[0]
    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced").fit(X, y)
    np.savez(E6 / "e6_probe.npz", coef=clf.coef_, intercept=clf.intercept_,
             thr=best[2], C=C, oof_f1=best[1])
    print(f"FINAL C={C} thr={best[2]:.4f} OOF_F1={best[1]:.3f}")
    return clf, best[2]


def evaluate(rows):
    """评测330：缓存 registry 嵌入 + e6_probe 冻结阈值。vs E8-G。"""
    from sklearn.metrics import average_precision_score
    z = np.load(E6 / "e6_probe.npz")
    w, b, thr = z["coef"], float(z["intercept"]), float(z["thr"])
    tk = json.load(open(CACHE / "e10_full/truth_key.json"))
    rows_t = tk if isinstance(tk, list) else list(tk.values())
    cols = []
    for name in ("clip", "dinov2"):
        for view in ("full", "crop"):
            zz = np.load(CACHE / f"{name}_{view}.npz", allow_pickle=True)
            m = dict(zip(zz["ids"].tolist(), zz["vecs"]))
            cols.append(np.stack([m[r["image_id"]] for r in rows_t]))
    X = np.concatenate(cols, 1).astype(np.float32)
    s = X @ w[0] + b
    y = np.array([1 if r["label"] == "positive" else 0 for r in rows_t])
    pred = s > thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F1 = 2 * P * R / max(1e-9, P + R)
    aupr = average_precision_score(y, s)
    # E8-G 对照
    e8 = np.array([r["e8g_score"] for r in rows_t], dtype=float)
    e8p = np.array([r["e8g_pred"] for r in rows_t], dtype=bool)
    tp2 = int((e8p & (y == 1)).sum()); fp2 = int((e8p & (y == 0)).sum()); fn2 = int((~e8p & (y == 1)).sum())
    P2 = tp2 / max(1, tp2 + fp2); R2 = tp2 / max(1, tp2 + fn2)
    F12 = 2 * P2 * R2 / max(1e-9, P2 + R2)
    aupr2 = average_precision_score(y, e8)
    # 门店级 bootstrap CI
    groups = np.array([r["group"] for r in rows_t])
    rng = np.random.default_rng(2026)
    ug = np.unique(groups); f1s_b = []
    for _ in range(2000):
        sg = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sg])
        yy, pp = y[idx], pred[idx]
        t = int((pp & (yy == 1)).sum()); f = int((pp & (yy == 0)).sum()); n = int((~pp & (yy == 1)).sum())
        f1s_b.append(2 * t / max(1, 2 * t + f + n))
    lo, hi = np.quantile(f1s_b, [0.025, 0.975])
    out = {
        "e6": {"P": round(P, 3), "R": round(R, 3), "F1": round(F1, 3), "AuPR": round(float(aupr), 3),
               "confusion": [tp, fp, fn], "thr_poolOOF": thr,
               "F1_CI95": [round(float(lo), 3), round(float(hi), 3)]},
        "e8g": {"P": round(P2, 3), "R": round(R2, 3), "F1": round(F12, 3), "AuPR": round(float(aupr2), 3),
                "confusion": [tp2, fp2, fn2]},
        "note": "thr 在池内OOF选,评测330未参与任何调参;池=弱标签(A/B/C),D剔除",
    }
    json.dump(out, open(E6 / "e6_eval.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    # 分歧样本明细
    dis = [{"image_id": r["image_id"], "label": r["label"], "e6s": round(float(si), 3),
            "e8g": round(float(ei), 3)} for r, si, ei in zip(rows_t, s, e8)
           if (si > thr) != bool(ei)]
    json.dump(dis, open(E6 / "e6_disagreements.json", "w"), ensure_ascii=False, indent=1)
    print(f"分歧样本 {len(dis)} -> e6_disagreements.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    rows = build_pool()
    print(f"池: {len(rows)} 图 | {Counter(r['lab'] for r in rows)} | {len(set(r['poi'] for r in rows))} POI")
    if a.embed or a.all:
        embed_pool(rows)
    if a.train or a.all:
        train(rows)
    if a.eval or a.all:
        evaluate(rows)
