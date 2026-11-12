#!/usr/bin/env python3
"""P0-01 探针融合消融（GPT报告B问/V1）：探针到底帮还是帮倒忙？
关键纠正：AuPR=平均精度AP，随机基线=正例比例π=30/330≈0.091，不是0.5。
对照：CLIP-only / DINO-only / probe-only(v1,v2) / CLIP+DINO(无探针) / 完整融合(0.4c+0.4d+0.2p)。
同一330评估集，逐样本分数落盘，AP+F1+门店组配对ΔAP。探索性（阈值曾在330上选）。"""
import json
import numpy as np
import visual_baseline as vb
import e8_train as e8
import e8_fusion as ef
import e8_multicrop as em
import e9_fusion as e9f

CACHE = vb.CACHE
ROOT = vb.ROOT


def load_synth_from(d):
    rows = []
    for line in open(d / "manifest_3d.jsonl"):
        r = json.loads(line)
        fp = d / r["image"]
        if not fp.exists():
            continue
        r["path"] = str(fp); r["group"] = f"synth_{r['seed']}"
        r["label"] = "positive" if r["brand"] == "left_right" else "negative"
        r["candidate_style"] = r["style_id"] or ""
        rows.append(r)
    return rows


def prf_at(scores, labels, thr):
    pred = scores > thr
    tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum()); fn = int((~pred & labels).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=round(p, 3), recall=round(r, 3),
                f1=round(2 * p * r / max(p + r, 1e-9), 3))


def best_f1(scores, labels):
    thrs = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best = None
    for t in thrs:
        m = prf_at(scores, labels, t)
        if best is None or m["f1"] > best["f1"]:
            best = {**m, "thr": float(t)}
    return best


def main():
    rows = vb.load_registry()
    labels = np.array([r["label"] == "positive" for r in rows])
    groups = np.array([r["group"] for r in rows])
    pi = float(labels.mean())
    cv, ci = vb.load_emb("clip", "full")
    id2i = {x: i for i, x in enumerate(ci)}
    s_clip = ef.knn_scores(rows, cv[[id2i[r["image_id"]] for r in rows]])
    s_dino, _ = em.multicrop_dino_scores(rows)
    s_dino = np.clip((s_dino - 0.3) / 0.5, 0, 1)
    rf, _ = e8.embed_all(rows, "real")
    probes = {}
    for ver, dn in [("v1", "renders_3d"), ("v2", "renders_3d_v2")]:
        sr = load_synth_from(ROOT / dn)
        sf, _ = e8.embed_all(sr, f"synth_{ver}", use_truth_crop=True)
        pr = e8.train_probes(sf, sr, seed=0)
        probes[ver] = pr["brand_clf"].predict_proba(pr["scaler"].transform(e8.concat_feats(rf)))[:, 1]
    s_fused_v1 = 0.4 * s_clip + 0.4 * s_dino + 0.2 * probes["v1"]
    s_fused_v2 = 0.4 * s_clip + 0.4 * s_dino + 0.2 * probes["v2"]
    s_noprobe = 0.5 * s_clip + 0.5 * s_dino

    systems = {
        "CLIP_only": s_clip, "DINO_only": s_dino,
        "probe_v1_only": probes["v1"], "probe_v2_only": probes["v2"],
        "CLIP+DINO_noprobe": s_noprobe,
        "fused_v1(0.4c0.4d0.2p)": s_fused_v1,
        "fused_v2(0.4c0.4d0.2p)": s_fused_v2,
    }
    out = {"n": len(rows), "n_pos": int(labels.sum()), "random_AP_baseline_pi": round(pi, 4),
           "note": "AP随机基线=正例比例π≈0.091,非0.5;阈值曾在330选→探索性",
           "systems": {}}
    print(f"随机AP基线 π = {pi:.4f} (不是0.5!)")
    print(f"{'system':28} {'AP':>6} {'bestF1':>7} {'P':>6} {'R':>6} {'thr':>6}")
    for name, sc in systems.items():
        ap = float(e9f.auprc(sc, labels))
        bf = best_f1(sc, labels)
        out["systems"][name] = dict(AP=round(ap, 3), **bf)
        print(f"{name:28} {ap:6.3f} {bf['f1']:7.3f} {bf['precision']:6.3f} {bf['recall']:6.3f} {bf['thr']:6.3f}")

    # 去探针消融：完整融合 vs 无探针（同权重结构对比 + 重归一化对比）
    print("\n=== 去探针消融 (ΔAP / ΔF1) ===")
    for fused_name, fused in [("fused_v1", s_fused_v1), ("fused_v2", s_fused_v2)]:
        ap_f = float(e9f.auprc(fused, labels)); ap_n = float(e9f.auprc(s_noprobe, labels))
        bf_f = best_f1(fused, labels); bf_n = best_f1(s_noprobe, labels)
        print(f"  {fused_name} vs CLIP+DINO: ΔAP={ap_f-ap_n:+.3f}  ΔF1={bf_f['f1']-bf_n['f1']:+.3f}")
        out.setdefault("ablation", {})[fused_name] = dict(delta_AP=round(ap_f - ap_n, 3), delta_F1=round(bf_f["f1"] - bf_n["f1"], 3))

    # 逐样本分数落盘（GPT V0要求）
    per_sample = []
    for i, r in enumerate(rows):
        per_sample.append(dict(image_id=r["image_id"], group=r["group"], truth=r["label"],
                               score_clip=round(float(s_clip[i]), 4), score_dino=round(float(s_dino[i]), 4),
                               score_probe_v1=round(float(probes["v1"][i]), 4), score_probe_v2=round(float(probes["v2"][i]), 4),
                               score_fused_v1=round(float(s_fused_v1[i]), 4), score_noprobe=round(float(s_noprobe[i]), 4)))
    json.dump(per_sample, open(CACHE / "probe_ablation_per_sample.json", "w"), ensure_ascii=False, indent=1)
    json.dump(out, open(CACHE / "probe_ablation.json", "w"), ensure_ascii=False, indent=1)
    print("\nsaved probe_ablation.json + probe_ablation_per_sample.json")


if __name__ == "__main__":
    main()