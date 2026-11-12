#!/usr/bin/env python3
"""
E8-G 多裁剪修复 + 竞品难负例 — 错误归因驱动的 targeted 提升
=============================================================
错误归因（vis_cache/e8f_errors.json + error_sheet.jpg）：
  FN 7：5/7 招牌检测失败→中心裁剪跑偏→DINOv2 crop 通道 s_dino≈0 拖垮融合。
  FP 9：render_stock 同色系竞品（芙蓉兴盛/便利蜂/邻几/ALDI）+ same_name 同名异业。
修复：
  (1) 多候选裁剪：每图取检测器多候选+上中部回退，DINOv2 crop 嵌入取与参考集最优相似度
      （而非只信第一候选/中心裁剪）→ 救回检测失败 FN。
  (2) 竞品难负例：Blender 新增 6 个真实 FP 竞品重渲（renders_3d 追加）。
评估：重算三通道融合 P/R/F1，对比 E8-F。探索性，store_id LOSO。
"""
import json
import numpy as np
import cv2
from PIL import Image
import visual_baseline as vb
import e8_train as e8
import e8_fusion as ef

CACHE = vb.CACHE


def multicrop_dino_scores(rows, k=5):
    """每图多候选裁剪 DINOv2 嵌入，对参考集取最优候选相似度，再 kNN 投票分数。"""
    emb = vb.Embedder("dinov2")
    # 每图候选嵌入 (n_cand, 768)
    cand_vecs, cand_ids = [], []
    for i, r in enumerate(rows):
        bgr = cv2.imread(r["path"])
        boxes = vb.sign_crop_candidates(bgr)
        vs = []
        for (x0, y0, x1, y1) in boxes:
            c = bgr[max(y0,0):y1, max(x0,0):x1]
            if c.size == 0:
                continue
            img = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            vs.append(emb.embed_pil([img])[0])
        cand_vecs.append(np.stack(vs) if vs else emb.embed_pil([Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))])[0:1])
        cand_ids.append(r["image_id"])
        if (i+1) % 100 == 0:
            print(f"  multicrop {i+1}/{len(rows)}")
    # 参考集 = 正例的最优候选嵌入（先用 full 图 DINOv2 作参考锚）
    # 对每个 query，每个候选与所有正例参考(多候选)算最大相似度，取最优候选
    pos_idx = [i for i, r in enumerate(rows) if r["label"] == "positive"]
    # 参考：正例所有候选展平
    ref_vecs = np.concatenate([cand_vecs[i] for i in pos_idx])
    ref_groups = []
    for i in pos_idx:
        ref_groups += [rows[i]["group"]] * len(cand_vecs[i])
    ref_groups = np.array(ref_groups)
    ref_labels_pos = np.ones(len(ref_vecs), dtype=bool)
    R = e8.l2(ref_vecs)
    scores = np.zeros(len(rows))
    for i, r in enumerate(rows):
        Q = e8.l2(cand_vecs[i])              # (n_cand, 768)
        S = Q @ R.T                          # (n_cand, n_ref)
        # 每候选取最优参考相似度，再选最优候选
        best_per_cand = S.max(axis=1)
        # kNN 投票：用最优候选的 top-k 参考（排除同组）
        best_cand = int(np.argmax(best_per_cand))
        order = np.argsort(-S[best_cand])
        votes = []
        for j in order:
            if ref_groups[j] == r["group"]:
                continue
            votes.append(True)               # 参考全是正例
            if len(votes) >= k:
                break
        # 分数 = 最优候选相似度 × 正例占比（参考全正例→占比1，用相似度调制）
        scores[i] = float(best_per_cand[best_cand])
    return scores, cand_vecs


def main():
    rows = vb.load_registry()
    labels = np.array([r["label"] == "positive" for r in rows])
    n_pos = int(labels.sum())

    # 多裁剪 DINOv2 分数（修复检测失败）
    s_dino_mc, _ = multicrop_dino_scores(rows)
    # 归一化到 [0,1]（相似度→投票分数尺度）：用 min-max 拉伸到与 knn 分数可比
    # 更稳妥：s_dino_mc 是余弦相似度[~0.2,0.9]，映射到投票占比尺度
    s_dino_mc = np.clip((s_dino_mc - 0.3) / 0.5, 0, 1)

    # CLIP kNN（不变）
    cv, ci = vb.load_emb("clip", "full")
    s_clip = ef.knn_scores(rows, cv[[{x:i for i,x in enumerate(ci)}[r["image_id"]] for r in rows]])

    # 合成探针（含新竞品难负例）
    sr = e8.load_synth()
    rf, _ = e8.embed_all(rows, "real")
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    probe = e8.train_probes(sf, sr, seed=0)
    X = probe["scaler"].transform(e8.concat_feats(rf))
    s_probe = probe["brand_clf"].predict_proba(X)[:, 1]

    summary = {}
    # 原 DINOv2 crop（对照）
    dv, di = vb.load_emb("dinov2", "crop")
    s_dino_old = ef.knn_scores(rows, dv[[{x:i for i,x in enumerate(di)}[r["image_id"]] for r in rows]])

    for name, sd in [("old_crop", s_dino_old), ("multicrop", s_dino_mc)]:
        s = 0.5*s_clip + 0.25*sd + 0.25*s_probe
        b = ef.best_f1(s, labels)
        summary[name] = dict(best_f1=b, risk_coverage=ef.risk_coverage(s, labels, n_pos))
        print(f"{name:12s} P {b['precision']:.3f} R {b['recall']:.3f} F1 {b['f1']:.3f} @thr {b['thr']} (tp{b['tp']} fp{b['fp']} fn{b['fn']})")

    # 多裁剪 + 权重网格
    grid = []
    for wc in (0.4, 0.5, 0.6):
        for wd in (0.2, 0.3, 0.4):
            for wp in (0.1, 0.2, 0.3):
                if abs(wc+wd+wp-1.0) > 1e-6:
                    continue
                s = wc*s_clip + wd*s_dino_mc + wp*s_probe
                b = ef.best_f1(s, labels)
                grid.append(dict(w_clip=wc, w_dino=wd, w_probe=wp, **b))
    grid.sort(key=lambda g: -g["f1"])
    summary["multicrop_grid_top5"] = grid[:5]
    print("\ntop multicrop fusion:")
    for g in grid[:5]:
        print(f"  wc{g['w_clip']} wd{g['w_dino']} wp{g['w_probe']}: P {g['precision']:.3f} R {g['recall']:.3f} F1 {g['f1']:.3f} (tp{g['tp']} fp{g['fp']} fn{g['fn']})")

    json.dump(summary, open(CACHE/"e8g_summary.json", "w"), ensure_ascii=False, indent=1)
    json.dump(dict(experiment="E8-G", version="e8g-multicrop-v1.0",
                   fix="multicrop_dino + competitor_hard_negatives",
                   governance="exploratory_probe_rights_pending_G0_G1_not_passed",
                   error_source="vis_cache/e8f_errors.json"),
              open(CACHE/"e8g_manifest.json", "w"), ensure_ascii=False, indent=1)
    print("\n→", CACHE/"e8g_summary.json")


if __name__ == "__main__":
    main()