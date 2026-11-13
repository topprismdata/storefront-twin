#!/usr/bin/env python3
"""P0 实验: 成对验证头 (face-recognition 式 pairwise verification head)。

假设: 冻结嵌入上的逐维关系特征(⊙,|diff|)能区分"品牌真店 vs 品牌风格设计图"——
绝对嵌入位置阈值(s_E8G)做不到的事。

协议:
- 24 正例组 → 6 折组不相交(seed 42);300 负例 6 路随机分折(单图独立,无组泄漏)
- 训练对: 真实正例对(组内禁自配) + renders_3d 正例对(governance: synthetic_train_only)
          vs impostor 对(训练折实拍负例 + renders_3d hard_negative/other_brand)
- 测试: 折外全部图 → max-MLP over gallery(训练折正例个体) → 池化 OOF 分数
- 基线: (a) 纯余弦 gallery(max-cos dino_crop) (b) E8-G 冻结 0.862/LOSO 0.848
- 素材域: 全量重训头 → 220 stock FAR @ OOF 最优 F1 阈值
输出: synthetic/vis_cache/pairwise_verifier_results.json
"""
import json, sys, os
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "synthetic/vis_cache"
sys.path.insert(0, str(ROOT / "synthetic"))
from visual_baseline import load_registry

SEED = 42
N_FOLDS = 6

def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

def pair_feats(a_clip, b_clip, a_dc, b_dc, a_df, b_df):
    """a,b 为 L2 归一化后特征。返回 [scalars(3), dino_crop ⊙|diff (1536), clip ⊙|diff (1024)]"""
    c1 = (a_clip * b_clip).sum(1, keepdims=True)
    c2 = (a_dc * b_dc).sum(1, keepdims=True)
    c3 = (a_df * b_df).sum(1, keepdims=True)
    f_dc = np.hstack([a_dc * b_dc, np.abs(a_dc - b_dc)])
    f_cl = np.hstack([a_clip * b_clip, np.abs(a_clip - b_clip)])
    return np.hstack([c1, c2, c3, f_dc, f_cl])

def main():
    rng = np.random.RandomState(SEED)
    rows = load_registry()
    reg = {r["image_id"]: r for r in rows}
    ids = [str(x) for x in np.load(VC / "e8_real_d32812084f210c6b.npz")["ids"]]
    assert set(ids) == set(reg.keys()), "id mismatch"
    emb = {k: l2n(np.load(VC / f"e8_real_d32812084f210c6b.npz")[k]) for k in ["clip_full", "dinov2_crop", "dinov2_full"]}

    pos_ids = [i for i in ids if reg[i]["label"] == "positive"]
    neg_ids = [i for i in ids if reg[i]["label"] == "negative"]
    groups = sorted(set(reg[i]["group"] for i in pos_ids))
    assert len(groups) == 24, len(groups)
    gorder = rng.permutation(len(groups))
    folds_g = [ [groups[j] for j in gorder[k::N_FOLDS]] for k in range(N_FOLDS) ]
    neg_order = rng.permutation(len(neg_ids))
    folds_neg = [ [neg_ids[j] for j in neg_order[k::N_FOLDS]] for k in range(N_FOLDS) ]

    # 合成域: 类别由 id 前缀解析
    sd = np.load(VC / "e8_synth_7ab27ecef8ebeafd.npz")
    sids = [str(x) for x in sd["ids"]]
    synth = {k: l2n(sd[k]) for k in ["clip_full", "dinov2_crop", "dinov2_full"]}
    s_pos = [i for i in sids if i.startswith("positive_")]
    s_imp = [i for i in sids if i.startswith("hard_negative_") or i.startswith("other_brand_")]
    print(f"synth: pos {len(s_pos)}, impostor {len(s_imp)}")

    def feats(idx_a, idx_b, pool):
        A = emb["clip_full"][idx_a] if pool == "real" else synth["clip_full"][idx_a]
        B = emb["clip_full"][idx_b] if pool == "real" else synth["clip_full"][idx_b]
        Adc = emb["dinov2_crop"][idx_a] if pool == "real" else synth["dinov2_crop"][idx_a]
        Bdc = emb["dinov2_crop"][idx_b] if pool == "real" else synth["dinov2_crop"][idx_b]
        Adf = emb["dinov2_full"][idx_a] if pool == "real" else synth["dinov2_full"][idx_a]
        Bdf = emb["dinov2_full"][idx_b] if pool == "real" else synth["dinov2_full"][idx_b]
        c1 = (A * B).sum(1, keepdims=True)
        c2 = (Adc * Bdc).sum(1, keepdims=True)
        c3 = (Adf * Bdf).sum(1, keepdims=True)
        f_dc = np.hstack([Adc * Bdc, np.abs(Adc - Bdc)])
        f_cl = np.hstack([A * B, np.abs(A - B)])
        return np.hstack([c1, c2, c3, f_dc, f_cl]).astype(np.float32)

    # 跨池配对特征(real/synth/stock 三个嵌入池)
    stk_npz = np.load(VC / "e8_stock_real_a9bad59a2be79e7c.npz")
    pools = {
        "real": emb,
        "synth": synth,
        "stock": {"clip_full": l2n(stk_npz["clip_full"]), "dinov2_crop": l2n(stk_npz["dinov2_crop"]),
                   "dinov2_full": l2n(stk_npz["dinov2_full"])},
    }

    def feats_cross(idx_a, idx_b, pool_a, pool_b):
        pa, pb = pools[pool_a], pools[pool_b]
        A, B = pa["clip_full"][idx_a], pb["clip_full"][idx_b]
        Adc, Bdc = pa["dinov2_crop"][idx_a], pb["dinov2_crop"][idx_b]
        Adf, Bdf = pa["dinov2_full"][idx_a], pb["dinov2_full"][idx_b]
        c1 = (A * B).sum(1, keepdims=True)
        c2 = (Adc * Bdc).sum(1, keepdims=True)
        c3 = (Adf * Bdf).sum(1, keepdims=True)
        return np.hstack([c1, c2, c3, Adc * Bdc, np.abs(Adc - Bdc), A * B, np.abs(A - B)]).astype(np.float32)

    def sample_pairs(list_a, list_b, n, rng_, same=False):
        """返回 (ia, ib) 索引对数组; same=True 时取无序不同元素对"""
        if same:
            K = len(list_a)
            ii = rng_.randint(0, K, size=(n, 2))
            mask = ii[:, 0] != ii[:, 1]
            ii = ii[mask]
            return list_a[ii[:, 0]], list_a[ii[:, 1]]
        ia = rng_.choice(len(list_a), size=n)
        ib = rng_.choice(len(list_b), size=n)
        return list_a[ia], list_b[ib]

    oof_score = {}
    oof_true = {}
    oof_cos = {}
    fold_stats = []
    SYNTH_MODE = os.environ.get("PV_SYNTH", "cross")  # none | synth_only | cross
    NOISE_AUG = int(os.environ.get("PV_NOISEAUG", "3"))
    for k in range(N_FOLDS):
        test_pos_g = set(folds_g[k])
        test_neg = set(folds_neg[k])
        train_pos = [i for i in pos_ids if reg[i]["group"] not in test_pos_g]
        test_pos = [i for i in pos_ids if reg[i]["group"] in test_pos_g]
        train_neg = [i for j, i in enumerate(neg_ids) if i not in test_neg]
        test_neg_l = [i for i in folds_neg[k]]

        pi = np.array([ids.index(i) for i in train_pos])
        ni = np.array([ids.index(i) for i in train_neg])
        ti = np.array([ids.index(i) for i in test_pos + test_neg_l])
        t_lab = np.array([1] * len(test_pos) + [0] * len(test_neg_l))

        # 训练对(按 SYNTH_MODE 配置)
        Xg_list, Xi_list = [], []
        n_genuine_real = 0
        g_a, g_b = sample_pairs(pi, pi, 400, rng, same=True)
        Xg_list.append(feats(g_a, g_b, "real")); n_genuine_real += len(g_a)
        # 噪声增广副本(把真实对 multiply 出来, 保持"跨风格同牌"分布)
        for aug in range(NOISE_AUG):
            g_a2, g_b2 = sample_pairs(pi, pi, 300, rng, same=True)
            Xg_list.append(feats(g_a2, g_b2, "real") + rng.normal(0, 0.01, size=(len(g_a2), 2563)).astype(np.float32))
            n_genuine_real += len(g_a2)
        if SYNTH_MODE == "synth_only":
            sp_a, sp_b = sample_pairs(np.array([sids.index(x) for x in s_pos]),
                                      np.array([sids.index(x) for x in s_pos]), 800, rng, same=True)
            Xg_list.append(feats(sp_a, sp_b, "synth"))
        elif SYNTH_MODE == "cross":
            # 关键: 真实×合成 同牌对 —— 教会头部"域差也是 genuine"
            rp = rng.choice(pi, size=800)
            sp = rng.choice([sids.index(x) for x in s_pos], size=800)
            Xg_list.append(feats_cross(rp, sp, "real", "synth"))
        i_a, i_b = sample_pairs(pi, ni, 2000, rng)
        Xi_list.append(feats(i_a, i_b, "real"))
        if SYNTH_MODE in ("synth_only", "cross"):
            si_a = rng.choice([sids.index(x) for x in s_pos], size=800)
            si_b = rng.choice([sids.index(x) for x in s_imp], size=800)
            Xi_list.append(feats(si_a, si_b, "synth"))
        X = np.vstack(Xg_list + Xi_list)
        y = np.array([1] * sum(len(x) for x in Xg_list) + [0] * sum(len(x) for x in Xi_list))
        clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=120,
                            random_state=SEED, early_stopping=False)
        clf.fit(X, y)

        # 测试: query vs gallery(train_pos 个体), max over gallery
        # 为省时: 对每个测试图, 只用 cos(dino_crop) 预筛 top-8 gallery, 再对这 8 个算 MLP 分数
        gi = pi  # gallery 索引
        q_dc = emb["dinov2_crop"][ti]
        g_dc = emb["dinov2_crop"][gi]
        sim = q_dc @ g_dc.T  # (n_test, n_gallery)
        topk = np.argsort(-sim, axis=1)[:, :8]
        scores = np.zeros(len(ti), dtype=np.float64)
        cos_scores = sim.max(axis=1)
        for q in range(len(ti)):
            cand = gi[topk[q]]
            qa = np.full(len(cand), ti[q])
            Xq = feats(qa, cand, "real")
            scores[q] = clf.predict_proba(Xq)[:, 1].max()
        for j, idx in enumerate(ti):
            oof_score[ids[idx]] = float(scores[j])
            oof_true[ids[idx]] = int(t_lab[j])
            oof_cos[ids[idx]] = float(cos_scores[j])
        fold_stats.append({"fold": k, "train_pos": len(train_pos), "test_pos": len(test_pos),
                           "train_pairs": int(len(X)), "pos_rate": float(y.mean())})
        print(f"fold{k}: train_pairs={len(X)} pos_rate={y.mean():.2f}")

    # 池化 OOF
    ids_oof = list(oof_score.keys())
    s = np.array([oof_score[i] for i in ids_oof])
    yt = np.array([oof_true[i] for i in ids_oof])
    auroc = roc_auc_score(yt, s)
    # best-F1 阈值 (strict >, 未取整)
    thrs = np.unique(s)
    best = (-1, None)
    for t in thrs:
        pred = s > t
        tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum()); fn = int((~pred & (yt == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, float(t), tp, fp, fn)
    f1v, thr, tp, fp, fn = best
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)

    # 余弦 gallery 基线
    cs = np.array([oof_cos[i] for i in ids_oof])
    bestc = (-1, None)
    for t in np.unique(cs):
        pred = cs > t
        tp2 = int((pred & (yt == 1)).sum()); fp2 = int((pred & (yt == 0)).sum()); fn2 = int((~pred & (yt == 1)).sum())
        f12 = 2 * tp2 / max(1, 2 * tp2 + fp2 + fn2)
        if f12 > bestc[0]:
            bestc = (f12, float(t), tp2, fp2, fn2)

    # 全量重训(与折内同配置) → stock FAR
    all_pos = np.array([ids.index(i) for i in pos_ids])
    g_a, g_b = sample_pairs(all_pos, all_pos, 400, rng, same=True)
    g_parts = [feats_cross(g_a, g_b, "real", "real")]
    for aug in range(NOISE_AUG):
        g_a2, g_b2 = sample_pairs(all_pos, all_pos, 300, rng, same=True)
        g_parts.append(feats_cross(g_a2, g_b2, "real", "real") + rng.normal(0, 0.01, size=(len(g_a2), 2563)).astype(np.float32))
    sp_all = np.array([sids.index(x) for x in s_pos])
    si_all = np.array([sids.index(x) for x in s_imp])
    ni_all = np.array([ids.index(i) for i in neg_ids])
    i_a, i_b = sample_pairs(all_pos, ni_all, 3000, rng)
    i_parts = [feats_cross(i_a, i_b, "real", "real")]
    if SYNTH_MODE == "synth_only":
        sp_a, sp_b = sample_pairs(sp_all, sp_all, 1000, rng, same=True)
        g_parts.append(feats_cross(sp_a, sp_b, "synth", "synth"))
    elif SYNTH_MODE == "cross":
        rp = rng.choice(all_pos, size=1000)
        sp = rng.choice(sp_all, size=1000)
        g_parts.append(feats_cross(rp, sp, "real", "synth"))
    if SYNTH_MODE in ("synth_only", "cross"):
        si_a = rng.choice(sp_all, size=1200)
        si_b = rng.choice(si_all, size=1200)
        i_parts.append(feats_cross(si_a, si_b, "synth", "synth"))
    Xf = np.vstack(g_parts + i_parts)
    yf = np.array([1] * sum(len(x) for x in g_parts) + [0] * sum(len(x) for x in i_parts))
    clfF = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=120, random_state=SEED)
    clfF.fit(Xf, yf)

    stk_ids = [str(x) for x in stk_npz["ids"]]
    s_dc = pools["stock"]["dinov2_crop"]
    g_dc_all = emb["dinov2_crop"][all_pos]
    sim = s_dc @ g_dc_all.T
    topk = np.argsort(-sim, axis=1)[:, :8]
    stk_scores = np.zeros(len(stk_ids))
    for q in range(len(stk_ids)):
        cand = all_pos[topk[q]]
        qidx = np.full(len(cand), q)
        Xq = feats_cross(qidx, cand, "stock", "real")
        stk_scores[q] = clfF.predict_proba(Xq)[:, 1].max()

    far_best = float((stk_scores > thr).mean())
    far_cos_at = float((sim.max(axis=1) > bestc[1]).mean())

    out = {
        "protocol": {"folds": N_FOLDS, "seed": SEED, "pos_groups": 24, "pair_features": 2563},
        "fold_stats": fold_stats,
        "verifier_oof": {"f1": round(f1v, 3), "thr": round(thr, 4), "tp": tp, "fp": fp, "fn": fn,
                          "precision": round(prec, 3), "recall": round(rec, 3), "auroc": round(auroc, 3)},
        "cosine_gallery_baseline": {"f1": round(bestc[0], 3), "thr": round(bestc[1], 4),
                                     "tp": bestc[2], "fp": bestc[3], "fn": bestc[4]},
        "e8g_reference": {"f1_frozen": 0.862, "f1_nested_loso": 0.848, "thr": 0.636},
        "stock_far": {"verifier@oof_thr": round(far_best, 4), "cosine@cos_thr": round(far_cos_at, 4),
                       "e8g@0.636": 0.036, "note": "verifier 阈值=330 OOF best-F1; 概率域与 E8G 分数域不同, 直接比较 FAR"},
    }
    json.dump(out, open(VC / "pairwise_verifier_results.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
