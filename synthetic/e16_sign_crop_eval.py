#!/usr/bin/env python3
"""e16 — 裁剪源对融合分的影响: DBNet det框(回退固定带) vs 固定带对照。
协议 = E6-v3 金标准: 639 店组 LOSO, 探针每折重训(合成真值裁剪不变+真实他组),
配对双臂共享实现, 只隔离裁剪源变量。输出 vis_cache/e16/e16_results.json。"""
import json, sys, os, random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))
import visual_baseline as vb
import e8_train as e8
import e15_text_det as e15
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

E16 = ROOT / "synthetic/vis_cache/e16"
E16.mkdir(parents=True, exist_ok=True)


def det_crop_box(bgr):
    """det框(pad15%) 或回退固定带 -> 归一化 xyxy"""
    d = e15.det_union_box(bgr)
    if d is None:
        return None
    x0, y0, x1, y1 = d
    pw, ph = (x1 - x0) * 0.15, (y1 - y0) * 0.15
    return [max(0, x0 - pw), max(0, y0 - ph), min(1, x1 + pw), min(1, y1 + ph)]


def crop_vecs(bgr, boxes, emb):
    out = []
    for b in boxes:
        x0, y0, x1, y1 = b
        h, w = bgr.shape[:2]
        c = bgr[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
        if c.size == 0:
            continue
        out.append(emb.embed_pil([Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))])[0])
    return out


def main():
    reg = json.load(open(ROOT / "data/registry_v11.json"))
    sha = json.load(open(ROOT / "data/sha_file_index.json"))
    rows = []
    for r in reg:
        if not r.get("is_representative"):
            continue
        p = sha.get(r["sha256"])
        if not p:
            continue
        fp = ROOT / "data" / p
        if fp.exists():
            rows.append({"image_id": r["image_id"], "path": str(fp), "label": r["label"],
                         "group": r.get("store_id") or r["image_id"],
                         "neg_type": r.get("neg_type") or ""})
    y = np.array([r["label"] == "positive" for r in rows])
    groups = np.array([r["group"] for r in rows])
    print(f"real rows {len(rows)}, pos {y.sum()}, groups {len(set(groups))}", flush=True)

    # 缓存特征: clip_full / dino_full / dino_crop_v2(det) / dino_crop_fix
    cf = E16 / "feats.npz"
    if cf.exists():
        z = np.load(cf, allow_pickle=True)
        clip_full, dino_full = z["clip_full"], z["dino_full"]
        crop_det, crop_fix, ids_order = z["crop_det"], z["crop_fix"], list(z["ids_order"])
        assert ids_order == [r["image_id"] for r in rows], "行序变化, 删缓存重跑"
    else:
        emb_c = vb.Embedder("clip"); emb_d = vb.Embedder("dinov2")
        clip_full, dino_full, crop_det, crop_fix = [], [], [], []
        for i, r in enumerate(rows):
            bgr = cv2.imread(r["path"])
            pil = Image.open(r["path"]).convert("RGB")
            clip_full.append(emb_c.embed_pil([pil])[0])
            dino_full.append(emb_d.embed_pil([pil])[0])
            db = det_crop_box(bgr)
            fx = vb.sign_crop_box(bgr)
            h, w = bgr.shape[:2]
            c_det = bgr[int(db[1]*h):int(db[3]*h), int(db[0]*w):int(db[2]*w)] if db else None
            c_fix = bgr[fx[1]:fx[3], fx[0]:fx[2]]
            crop_det.append(emb_d.embed_pil([Image.fromarray(cv2.cvtColor(c_det, cv2.COLOR_BGR2RGB))])[0] if c_det is not None and c_det.size else emb_d.embed_pil([pil])[0])
            crop_fix.append(emb_d.embed_pil([Image.fromarray(cv2.cvtColor(c_fix, cv2.COLOR_BGR2RGB))])[0] if c_fix.size else emb_d.embed_pil([pil])[0])
            if (i + 1) % 100 == 0:
                print(f"embed {i+1}/{len(rows)}", flush=True)
        clip_full = np.stack(clip_full); dino_full = np.stack(dino_full)
        crop_det = np.stack(crop_det); crop_fix = np.stack(crop_fix)
        np.savez(cf, clip_full=clip_full, dino_full=dino_full, crop_det=crop_det,
                 crop_fix=crop_fix, ids=np.array([r["image_id"] for r in rows]))
        print("embed 完成", flush=True)

    # 合成 (探针训练底座, 真值裁剪不变, 走缓存)
    sr = e8.load_synth()
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    X_syn = e8.concat_feats({"clip_full": sf["clip_full"], "dinov2_full": sf["dinov2_full"],
                             "dinov2_crop": sf["dinov2_crop"]})
    y_syn = np.array([r["label"] == "positive" for r in sr])

    l2c = e8.l2(clip_full)
    pos_ref = l2c[y]  # kNN 画廊(组排除在折内做)
    dino_ref_det = e8.l2(np.vstack([dino_full[y], crop_det[y]]))
    dino_ref_fix = e8.l2(np.vstack([dino_full[y], crop_fix[y]]))
    ref_groups = np.concatenate([groups[y], groups[y]])

    def l2(x):
        return x / np.maximum(1e-9, np.linalg.norm(x, axis=1, keepdims=True))

    l2c = l2(clip_full)
    pos_ref = l2c[y]
    dino_ref_det = l2(np.vstack([dino_full[y], crop_det[y]]))
    dino_ref_fix = l2(np.vstack([dino_full[y], crop_fix[y]]))
    ref_groups = np.concatenate([groups[y], groups[y]])

    glist = sorted(set(groups))
    rng = random.Random(0)
    scores = {"det": np.zeros(len(rows)), "fix": np.zeros(len(rows))}
    pred_dump = {k: np.zeros(len(rows)) for k in scores}
    for fi, g in enumerate(glist):
        te = groups == g
        tr = ~te
        Xr_det = np.concatenate([X_syn, np.concatenate([l2c, l2(dino_full), l2(crop_det)], axis=1)[tr]], axis=0)
        Xr_fix = np.concatenate([X_syn, np.concatenate([l2c, l2(dino_full), l2(crop_fix)], axis=1)[tr]], axis=0)
        yr = np.concatenate([y_syn, y[tr]])
        for arm, Xr in (("det", Xr_det), ("fix", Xr_fix)):
            sc = StandardScaler().fit(Xr)
            clf = LogisticRegression(max_iter=1500, C=1.0, random_state=0).fit(sc.transform(Xr), yr)
            Xt = np.concatenate([l2c, l2(dino_full), l2(crop_det if arm == "det" else crop_fix)], axis=1)[te]
            sp = clf.predict_proba(sc.transform(Xt))[:, 1]
            qi = np.where(te)[0]
            # s_clip: top5 vs 他组正例
            gal = np.where(tr & y)[0]
            gal_ok = gal[groups[gal] != g]
            S = l2c[qi] @ l2c[gal_ok].T
            s_clip = np.sort(S, axis=1)[:, -5:].mean(axis=1)
            # s_dino: 候选(det臂=det框; fix臂=固定带) vs 他组正例参考(full+crop)
            ref = (dino_ref_det if arm == "det" else dino_ref_fix)
            gmask = np.concatenate([groups[y], groups[y]]) != g
            ref2 = ref[gmask]
            Q = l2(np.vstack([ (crop_det if arm=="det" else crop_fix)[qi], dino_full[qi] ]))
            Sd = Q @ ref2.T
            n_c = len(qi)
            best = np.maximum(Sd[:n_c].max(axis=1), Sd[n_c:].max(axis=1))
            s_dino = np.clip((best - 0.3) / 0.5, 0, 1)
            scores[arm][qi] = 0.4 * np.clip(s_clip, 0, 1) + 0.4 * s_dino + 0.2 * sp
        if (fi + 1) % 100 == 0:
            print(f"loso {fi+1}/{len(glist)}", flush=True)

    old = np.load(ROOT / "synthetic/vis_cache/e6v3_scores.npz", allow_pickle=True)
    old_map = {i: float(s) for i, s in zip(old["ids"], old["s_real"])}
    s_e6v3 = np.array([old_map.get(r["image_id"], np.nan) for r in rows])

    def prf(s, t, mask=None):
        m = slice(None) if mask is None else mask
        p = s > t
        tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
        P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
        F = 2 * P * R / max(1e-12, P + R)
        return dict(P=round(P, 4), R=round(R, 4), F1=round(F, 4), tp=tp, fp=fp, fn=fn)

    out = {"n": len(rows), "arms": {}}
    for arm in ("det", "fix"):
        out["arms"][arm] = {"thr0.636": prf(scores[arm], 0.636), "thr0.80": prf(scores[arm], 0.80)}
    # 对照 e6v3 原始分
    ok = ~np.isnan(s_e6v3)
    out["arms"]["e6v3_original"] = {"thr0.636": prf(s_e6v3, 0.636, ok), "thr0.80": prf(s_e6v3, 0.80, ok)}
    # det vs fix 翻转
    fl = (scores["det"] > 0.636) != (scores["fix"] > 0.636)
    out["flips_0636"] = int(fl.sum())
    flip_detail = [{"image_id": rows[i]["image_id"], "label": rows[i]["label"],
                    "neg_type": rows[i]["neg_type"],
                    "det": round(float(scores["det"][i]), 3), "fix": round(float(scores["fix"][i]), 3),
                    "e6v3": round(float(s_e6v3[i]), 3) if not np.isnan(s_e6v3[i]) else None}
                   for i in np.where(fl)[0]][:60]
    out["flip_detail"] = flip_detail
    # FP 明细(neg_type 分布)
    for arm in ("det", "fix"):
        p = scores[arm] > 0.636
        fps = [rows[i]["neg_type"] for i in np.where(p & ~y)[0]]
        from collections import Counter
        out["arms"][arm]["fp_by_type"] = dict(Counter(fps))
    json.dump(out, open(E16 / "e16_results.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "flip_detail"}, ensure_ascii=False, indent=1), flush=True)
    print("落盘", flush=True)


if __name__ == "__main__":
    main()
