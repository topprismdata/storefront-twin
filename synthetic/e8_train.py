#!/usr/bin/env python3
"""
E8-3D 合成训练 → 真实域验收 — 左邻右舍门头识别
=================================================
用户指令 2026-11-11：3D 孪生合成数据做识别训练，提升精确率/召回率。
协议（设计文档 §9.1 E8 / §9.2）：
- 训练只用 M1-3D 合成（self_created，构造真值）；真实图权利未审 → 仅探索性验收。
- 合成永不进入真实测试；真实评估按 store_id 分组。
- 分类器只训合成 → 对真实门店天然无泄漏（合成不含任何真实门店）。
- kNN+真实参考的对照保持 LOSO。

三条路线对比 E3 基线：
  A. synth_kNN   : 合成嵌入扩充参考库（不训练），真实 LOSO 检索
  B. linear_probe: 合成上线性探针（L0 品牌 / L1 簇），真实直接预测
  C. mlp_probe   : 合成上小 MLP（3 seeds），真实直接预测
评估：L0 precision/recall/F1 + 风险-覆盖率；L1 LOSO R@1/R@5 + 分类准确率。
"""
import argparse, json, os
from collections import Counter
from pathlib import Path
import numpy as np

import visual_baseline as vb

ROOT, CACHE, DATA = vb.ROOT, vb.CACHE, vb.DATA
RENDERS = ROOT / "renders_3d"
E8_VERSION = "e8-3d-train-v1.0"


def load_synth():
    rows = []
    mf = RENDERS / "manifest_3d.jsonl"
    for line in open(mf):
        r = json.loads(line)
        fp = RENDERS / r["image"]
        if not fp.exists():
            continue
        r["path"] = str(fp)
        r["group"] = f"synth_{r['seed']}"
        r["label"] = "positive" if r["brand"] == "left_right" else "negative"
        r["candidate_style"] = r["style_id"] or ""
        rows.append(r)
    return rows


def crop_from_poly(img, poly):
    """真值 polygon → 裁剪框（合成有构造真值，不用检测器）。"""
    H, W = img.shape[:2]
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    x0, x1 = max(0, int(min(xs)*W)), min(W, int(max(xs)*W))
    y0, y1 = max(0, int(min(ys)*H)), min(H, int(max(ys)*H))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return img[y0:y1, x0:x1]


def embed_all(rows, tag, use_truth_crop=False):
    """CLIP full + DINOv2 full/crop 嵌入；缓存 npz。"""
    import torch, cv2
    from PIL import Image
    import hashlib
    ids_now = [r.get("image_id") or r.get("image") for r in rows]
    fp = hashlib.sha256(("|".join(ids_now)).encode()).hexdigest()[:16]
    out = CACHE / f"e8_{tag}_{fp}.npz"
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return {k: z[k] for k in z.files if k != "ids"}, list(z["ids"])
    feats = {}
    for name in ("clip", "dinov2"):
        emb = vb.Embedder(name)
        views = ["full", "crop"] if name == "dinov2" else ["full"]
        for view in views:
            vecs, ids = [], []
            for i, r in enumerate(rows):
                bgr = cv2.imread(r["path"])
                if bgr is None:
                    continue
                if view == "crop":
                    if use_truth_crop and r.get("sign_polygon"):
                        c = crop_from_poly(bgr, r["sign_polygon"])
                        if c is not None:
                            bgr = c
                    else:
                        x0, y0, x1, y1 = vb.sign_crop_box(bgr)
                        bgr = bgr[max(y0,0):y1, max(x0,0):x1]
                img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                v = emb.embed_pil([img])[0]
                vecs.append(v); ids.append(r.get("image_id") or r.get("image"))
                if (i+1) % 150 == 0:
                    print(f"  {tag} {name}/{view} {i+1}/{len(rows)}")
            feats[f"{name}_{view}"] = np.stack(vecs)
            print(f"embedded {tag} {name}/{view} {np.stack(vecs).shape}")
        del emb
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    np.savez_compressed(out, ids=np.array(ids), **feats)
    return feats, ids


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)


def concat_feats(feats):
    return np.concatenate([l2(feats["clip_full"]), l2(feats["dinov2_full"]),
                           l2(feats["dinov2_crop"])], axis=1)


# ---------------------------------------------------------------------------
# A. 合成扩充参考库 kNN（LOSO 真实）
# ---------------------------------------------------------------------------
def eval_synth_knn(real_rows, synth_rows, rf, sf, k=5):
    """L1：真实 query 从 [其余店真实正例 + 全部合成正例] 检索。"""
    Xr = concat_feats(rf)
    pos = [r for r in real_rows if r["label"] == "positive" and r["candidate_style"]]
    id2i = {r["image_id"]: i for i, r in enumerate(real_rows)}
    Xref = Xr[[id2i[r["image_id"]] for r in pos]]
    spos = [r for r in synth_rows if r["label"] == "positive"]
    if spos:
        Xs = concat_feats(sf)
        Xref = np.concatenate([Xref, Xs[[i for i, r in enumerate(synth_rows) if r["label"] == "positive"]]])
    ref_styles = [r["candidate_style"] for r in pos] + [r["candidate_style"] for r in spos]
    ref_groups = [r["group"] for r in pos] + [r["group"] for r in spos]
    S = l2(Xr) @ l2(Xref).T
    results = []
    for r in pos:
        i = id2i[r["image_id"]]
        order = np.argsort(-S[i])
        nbrs = []
        for j in order:
            if ref_groups[j] == r["group"]:
                continue
            nbrs.append(ref_styles[j])
            if len(nbrs) >= k:
                break
        if not nbrs:
            continue
        vote = Counter(nbrs).most_common(1)[0][0]
        results.append({"image_id": r["image_id"], "gt_style": r["candidate_style"],
                        "top1": nbrs[0], "recall1": nbrs[0] == r["candidate_style"],
                        "recall5": r["candidate_style"] in nbrs,
                        "vote_correct": vote == r["candidate_style"]})
    return results


def eval_brand_knn(real_rows, synth_rows, rf, sf, k=5):
    """L0：参考库 = 其余组真实 + 合成；kNN 投票。返回逐样本分数（正例占比）。"""
    Xr = concat_feats(rf)
    Xref = np.concatenate([Xr, concat_feats(sf)]) if synth_rows else Xr
    ref_labels = [r["label"] for r in real_rows] + [r["label"] for r in synth_rows]
    ref_groups = [r["group"] for r in real_rows] + [r["group"] for r in synth_rows]
    S = l2(Xr) @ l2(Xref).T
    results = []
    for i, r in enumerate(real_rows):
        order = np.argsort(-S[i])
        votes = []
        for j in order:
            if ref_groups[j] == r["group"]:
                continue
            votes.append(ref_labels[j])
            if len(votes) >= k:
                break
        score = sum(v == "positive" for v in votes) / max(len(votes), 1)
        results.append({"image_id": r["image_id"], "label": r["label"],
                        "score": score, "pred": "left_right" if score > 0.5 else "other_brand"})
    return results


# ---------------------------------------------------------------------------
# B/C. 合成上训练探针
# ---------------------------------------------------------------------------
def train_probes(sf, synth_rows, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    X = concat_feats(sf)
    y_brand = np.array([r["label"] == "positive" for r in synth_rows])
    pos_idx = [i for i, r in enumerate(synth_rows) if r["label"] == "positive"]
    classes = sorted({synth_rows[i]["candidate_style"] for i in pos_idx})
    c2i = {c: i for i, c in enumerate(classes)}
    y_style = np.array([c2i[synth_rows[i]["candidate_style"]] for i in pos_idx])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    brand_clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, y_brand)
    style_clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xs[pos_idx], y_style)
    es = len(pos_idx) >= 60
    mlp_brand = MLPClassifier(hidden_layer_sizes=(128,), alpha=1.0, max_iter=600,
                              random_state=seed, early_stopping=es).fit(Xs, y_brand)
    mlp_style = MLPClassifier(hidden_layer_sizes=(128,), alpha=1.0, max_iter=600,
                              random_state=seed, early_stopping=es).fit(Xs[pos_idx], y_style)
    return dict(scaler=scaler, brand_clf=brand_clf, style_clf=style_clf,
                mlp_brand=mlp_brand, mlp_style=mlp_style, classes=classes)


def eval_probes(real_rows, rf, probe, kind="linear"):
    X = probe["scaler"].transform(concat_feats(rf))
    bkey, skey = ("brand_clf", "style_clf") if kind == "linear" else ("mlp_brand", "mlp_style")
    pb = probe[bkey].predict_proba(X)[:, 1]
    brand_res = [{"image_id": r["image_id"], "label": r["label"], "score": float(pb[i]),
                  "pred": "left_right" if pb[i] > 0.5 else "other_brand"}
                 for i, r in enumerate(real_rows)]
    pos = [i for i, r in enumerate(real_rows) if r["label"] == "positive" and r["candidate_style"]]
    ps = probe[skey].predict(X[pos])
    classes = probe["classes"]
    style_res = [{"image_id": real_rows[i]["image_id"],
                  "gt_style": real_rows[i]["candidate_style"],
                  "pred": classes[int(ps[j])],
                  "correct": classes[int(ps[j])] == real_rows[i]["candidate_style"]}
                 for j, i in enumerate(pos)]
    return brand_res, style_res


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def brand_metrics(res, thr=0.5):
    pos = [r for r in res if r["label"] == "positive"]
    neg = [r for r in res if r["label"] == "negative"]
    tp = sum(r["score"] > thr for r in pos)
    fp = sum(r["score"] > thr for r in neg)
    fn = len(pos) - tp
    prec = tp / max(tp + fp, 1); rec = tp / max(len(pos), 1)
    f1 = 2*prec*rec / max(prec + rec, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=round(prec, 3),
                recall=round(rec, 3), f1=round(f1, 3),
                specificity=round(1 - fp/max(len(neg), 1), 3))


def risk_coverage(res):
    """按分数排序的风险-覆盖率曲线（§7.4）。"""
    srt = sorted(res, key=lambda r: -r["score"])
    n = len(srt); curve = []
    tp = fp = 0
    for i, r in enumerate(srt):
        if r["label"] == "positive":
            tp += 1
        else:
            fp += 1
        if (i+1) % 30 == 0 or i == n-1:
            curve.append(dict(coverage=round((i+1)/n, 3),
                              precision=round(tp/max(tp+fp, 1), 3),
                              recall=round(tp/30, 3)))
    return curve


def main():
    global RENDERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--renders", default=str(RENDERS))
    args = ap.parse_args()
    RENDERS = Path(args.renders)

    real_rows = vb.load_registry()
    synth_rows = load_synth()
    print(f"real {len(real_rows)} | synth {len(synth_rows)} "
          f"(pos {sum(r['label']=='positive' for r in synth_rows)})")

    if args.embed or args.all:
        embed_all(real_rows, "real")
        embed_all(synth_rows, "synth", use_truth_crop=True)

    if args.eval or args.all:
        rf, _ = embed_all(real_rows, "real")
        sf, _ = embed_all(synth_rows, "synth", use_truth_crop=True)
        summary = {}
        # 对照基线：E3 同特征 concat，仅真实参考（LOSO），隔离合成增量
        b0 = eval_brand_knn(real_rows, [], rf, {})
        summary["baseline_real_knn_brand"] = brand_metrics(b0)
        c0 = eval_synth_knn(real_rows, [], rf, {})
        summary["baseline_real_knn_cluster"] = vb.summarize_cluster(
            [dict(r, group="") for r in c0])
        json.dump(b0, open(CACHE/"e8_baseline_brand.json", "w"), indent=1)

        # A. synth_kNN
        b = eval_brand_knn(real_rows, synth_rows, rf, sf)
        summary["A_synth_knn_brand"] = brand_metrics(b)
        summary["A_synth_knn_risk_coverage"] = risk_coverage(b)
        c = eval_synth_knn(real_rows, synth_rows, rf, sf)
        summary["A_synth_knn_cluster"] = vb.summarize_cluster(
            [dict(r, vote_correct=r["vote_correct"], group="") for r in c])
        json.dump(b, open(CACHE/"e8_A_brand.json", "w"), indent=1)
        json.dump(c, open(CACHE/"e8_A_cluster.json", "w"), indent=1)

        # B. linear probe（确定性）
        probe = train_probes(sf, synth_rows, seed=0)
        b, c = eval_probes(real_rows, rf, probe, "linear")
        summary["B_linear_brand"] = brand_metrics(b)
        summary["B_linear_risk_coverage"] = risk_coverage(b)
        summary["B_linear_cluster_acc"] = round(sum(r["correct"] for r in c)/max(len(c),1), 3)
        json.dump(b, open(CACHE/"e8_B_brand.json", "w"), indent=1)
        json.dump(c, open(CACHE/"e8_B_cluster.json", "w"), indent=1)

        # C. MLP 3 seeds
        mlps = []
        for seed in (0, 1, 2):
            p = train_probes(sf, synth_rows, seed=seed)
            b, c = eval_probes(real_rows, rf, p, "mlp")
            mlps.append(dict(seed=seed, brand=brand_metrics(b),
                             cluster_acc=round(sum(r["correct"] for r in c)/max(len(c),1), 3)))
            json.dump(b, open(CACHE/f"e8_C_brand_seed{seed}.json", "w"), indent=1)
        summary["C_mlp_3seeds"] = mlps

        json.dump(summary, open(CACHE/"e8_summary.json", "w"), ensure_ascii=False, indent=1)
        manifest = dict(experiment="E8-3D", version=E8_VERSION,
                        train="synthetic_only_self_created", eval="real_exploratory",
                        real_split="store_id_grouped", synth_split="synthetic_train_only",
                        features="clip_full+dinov2_full+dinov2_crop (frozen)",
                        probes=["logreg", "mlp_128_3seeds"],
                        governance="exploratory_probe_rights_pending_G0_G1_not_passed",
                        baseline_ref="vis_cache/summary.json (E3)")
        json.dump(manifest, open(CACHE/"e8_manifest.json", "w"), ensure_ascii=False, indent=1)
        print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()