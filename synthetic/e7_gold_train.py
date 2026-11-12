#!/usr/bin/env python3
"""
E7 — 金标准训练（标注完成后运行）
==================================
输入: data/annotation_pack/verdict_template.jsonl（人工填写: verdict A/B/C/D）
训练: 金标准 644 图（缓存嵌入,四路与 E6 同构）→ 逻辑回归（含 gold+weak 混合对照）
评测: 330 注册集（冻结阈值,池内 OOF 选）vs E8-G / E6-v2
守卫: 模板未填完时拒绝运行并报进度。
"""
import json, sys
from collections import Counter
import numpy as np

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import e6_real_probe as e1

E6, DATA, CACHE = e1.E6, e1.DATA, e1.CACHE


def load_gold():
    man = {a["anno_id"]: a for a in json.load(open(DATA / "annotation_pack/anno_manifest.json"))}
    filled, empty = {}, 0
    for line in open(DATA / "annotation_pack/verdict_template.jsonl"):
        o = json.loads(line)
        if o.get("verdict"):
            filled[o["anno_id"]] = o
        else:
            empty += 1
    if empty:
        print(f"[守卫] 模板未填完: 空 {empty} / 已填 {len(filled)}。填完后再跑 E7。")
        sys.exit(1)
    rows = []
    for aid, o in filled.items():
        a = man[aid]
        rows.append({"image_id": a["image_id"], "path": a["path"], "poi": "gold_" + a["image_id"],
                     "lab": o["verdict"], "src": "gold", "province": a["province"],
                     "style_variant": bool(o.get("style_variant"))})
    return rows


def feats(rows):
    m = {}
    for n in ("clip", "dinov2"):
        for v in ("full", "crop"):
            for tag in ("pool", "x"):
                f = E6 / f"{tag}_{n}_{v}.npz"
                if f.exists():
                    z = np.load(f, allow_pickle=True)
                    m[(n, v)] = m.get((n, v), {})
                    m[(n, v)].update(zip(z["ids"].tolist(), z["vecs"]))
    keep = [r for r in rows if all(r["image_id"] in m[(n, v)] for n in ("clip", "dinov2") for v in ("full", "crop"))]
    X = np.concatenate([np.stack([m[(n, v)][r["image_id"]] for r in keep])
                        for n in ("clip", "dinov2") for v in ("full", "crop")], 1).astype(np.float32)
    return keep, X


def train_and_eval(rows, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import average_precision_score
    keep, X = feats(rows)
    y = np.array([1 if r["lab"] == "A" else 0 for r in keep])
    g = np.array([r["poi"] for r in keep])
    print(f"[{tag}] X={X.shape} pos={y.sum()} neg={len(y)-y.sum()}")
    best = None
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced").fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        thrs = np.unique(np.quantile(oof, np.linspace(0.02, 0.98, 97)))
        f1s = [2 * ((oof > t) & (y == 1)).sum() / max(1, 2 * ((oof > t) & (y == 1)).sum()
                + ((oof > t) & (y == 0)).sum() + ((oof <= t) & (y == 1)).sum()) for t in thrs]
        i = int(np.argmax(f1s))
        if best is None or f1s[i] > best[1]:
            best = (C, f1s[i], float(thrs[i]))
    clf = LogisticRegression(max_iter=2000, C=best[0], class_weight="balanced").fit(X, y)
    thr = best[2]
    # 评测 330
    tk = json.load(open(CACHE / "e10_full/truth_key.json"))
    tr = tk if isinstance(tk, list) else list(tk.values())
    mm = {}
    for n in ("clip", "dinov2"):
        for v in ("full", "crop"):
            zz = np.load(CACHE / f"{n}_{v}.npz", allow_pickle=True)
            mm[(n, v)] = dict(zip(zz["ids"].tolist(), zz["vecs"]))
    Xe = np.concatenate([np.stack([mm[(n, v)][r["image_id"]] for r in tr])
                         for n in ("clip", "dinov2") for v in ("full", "crop")], 1).astype(np.float64)
    s = Xe @ clf.coef_[0].astype(np.float64) + float(clf.intercept_)
    yy = np.array([r["label"] == "positive" for r in tr])
    pred = s > thr
    tp = int((pred & yy).sum()); fp = int((pred & ~yy).sum()); fn = int((~pred & yy).sum())
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    res = {"tag": tag, "P": round(P, 3), "R": round(R, 3),
           "F1": round(2 * P * R / max(1e-9, P + R), 3),
           "AuPR": round(float(average_precision_score(yy, s)), 3),
           "confusion": [tp, fp, fn], "thr": round(thr, 4), "C": best[0], "oof_f1": round(best[1], 3)}
    print(res)
    return clf, res


if __name__ == "__main__":
    gold = load_gold()
    print("gold 分布:", Counter(r["lab"] for r in gold))
    results = {}
    clf_a, r1 = train_and_eval(gold, "gold_only")
    results["gold_only"] = r1
    # gold + weak 混合: weak = 池内未入选标注的 B/C + 全部 A 弱标签(权重 0.5)
    pool = [r for r in e1.build_pool() if r["image_id"] not in {g["image_id"] for g in gold}]
    mixed = gold + pool
    clf_b, r2 = train_and_eval(mixed, "gold_plus_weak")
    results["gold_plus_weak"] = r2
    results["reference"] = {"e8g": {"P": 0.778, "R": 0.933, "F1": 0.848},
                            "e6v2_weak": {"P": 0.707, "R": 0.967, "F1": 0.817}}
    json.dump(results, open(E6 / "e7_results.json", "w"), ensure_ascii=False, indent=1)
    np.savez(E6 / "e7_probe.npz", coef=clf_b.coef_, intercept=clf_b.intercept_)
    print("E7 完成 -> e7_results.json / e7_probe.npz")
