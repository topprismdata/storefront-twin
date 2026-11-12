#!/usr/bin/env python3
"""E8-G 最终口径复算：最佳融合工作点 + 开集协议指标 + 残余错误清单。
产物 vis_cache/e8g_protocol.json + e8g_errors.json（供 GPT 讨论简报，全部 measured）。"""
import json
import numpy as np
import visual_baseline as vb
import e8_train as e8
import e8_fusion as ef
import e8_multicrop as em
import e9_fusion as e9f

CACHE = vb.CACHE


def main():
    rows = vb.load_registry()
    labels = np.array([r["label"] == "positive" for r in rows])

    cv, ci = vb.load_emb("clip", "full")
    id2i = {x: i for i, x in enumerate(ci)}
    s_clip = ef.knn_scores(rows, cv[[id2i[r["image_id"]] for r in rows]])
    s_dino_mc, _ = em.multicrop_dino_scores(rows)
    s_dino_mc = np.clip((s_dino_mc - 0.3) / 0.5, 0, 1)
    sr = e8.load_synth()
    rf, _ = e8.embed_all(rows, "real")
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    probe = e8.train_probes(sf, sr, seed=0)
    s_probe = probe["brand_clf"].predict_proba(probe["scaler"].transform(e8.concat_feats(rf)))[:, 1]

    s = 0.4 * s_clip + 0.4 * s_dino_mc + 0.2 * s_probe
    thrs = np.unique(np.concatenate([s, [0.0, 1.0]]))
    best_m, thr = None, 0.0
    for t in thrs:
        m = ef.metrics_at(s, labels, t)
        if best_m is None or m["f1"] > best_m["f1"]:
            best_m, thr = m, float(t)
    b = best_m
    pred = s > thr

    errors = []
    for i, r in enumerate(rows):
        err = "FN" if (labels[i] and not pred[i]) else ("FP" if (not labels[i] and pred[i]) else None)
        if err:
            errors.append(dict(image_id=r["image_id"], path=r["path"], label=r["label"],
                               neg_type=r.get("neg_type", ""), group=r["group"],
                               score=round(float(s[i]), 3), s_clip=round(float(s_clip[i]), 3),
                               s_dino=round(float(s_dino_mc[i]), 3), s_probe=round(float(s_probe[i]), 3),
                               err=err))

    protocol = dict(
        operating_point=b,
        auprc=round(e9f.auprc(s, labels), 3),
        p_at_95r=round(e9f.p_at_r(s, labels, 0.95), 3),
        r_at_95p=round(e9f.r_at_p(s, labels, 0.95), 3),
        n_rows=len(rows), n_pos=int(labels.sum()),
        n_errors=len(errors),
        channels=dict(clip=dict(auprc=round(e9f.auprc(s_clip, labels), 3)),
                      dino_mc=dict(auprc=round(e9f.auprc(s_dino_mc, labels), 3)),
                      probe=dict(auprc=round(e9f.auprc(s_probe, labels), 3))))
    json.dump(protocol, open(CACHE / "e8g_protocol.json", "w"), ensure_ascii=False, indent=1)
    json.dump(errors, open(CACHE / "e8g_errors.json", "w"), ensure_ascii=False, indent=1)
    print("E8-G operating point:", b)
    print("AuPR", protocol["auprc"], "P@95R", protocol["p_at_95r"], "R@95P", protocol["r_at_95p"])
    print("channel AuPR:", protocol["channels"])
    print(f"errors: {sum(1 for e in errors if e['err']=='FN')} FN / {sum(1 for e in errors if e['err']=='FP')} FP")
    for e in errors:
        print(f"  {e['err']} {e['image_id']} {e['neg_type'] or e.get('style','')} score={e['score']} "
              f"(clip {e['s_clip']} dino {e['s_dino']} probe {e['s_probe']})")

    # 探针否决规则：s_probe < tau 时强制拒绝（合成探针对 render_stock 几乎全对）
    veto = []
    for tau in (0.02, 0.05, 0.1, 0.2):
        pv = s.copy()
        pv[s_probe < tau] = 0.0
        bv = ef.best_f1(pv, labels)
        veto.append(dict(tau=tau, **bv, auprc=round(e9f.auprc(pv, labels), 3)))
    # 权重网格（含更高探针权重）
    grid = []
    for wc in (0.3, 0.4, 0.5):
        for wd in (0.2, 0.3, 0.4):
            wp = round(1.0 - wc - wd, 2)
            if wp < 0.1 or wp > 0.5:
                continue
            sg = wc * s_clip + wd * s_dino_mc + wp * s_probe
            bg = ef.best_f1(sg, labels)
            grid.append(dict(w_clip=wc, w_dino=wd, w_probe=wp, **bg))
    grid.sort(key=lambda g: (-g["f1"], g["fp"]))
    extra = dict(veto=veto, grid_top6=grid[:6])
    json.dump(extra, open(CACHE / "e8g_veto_grid.json", "w"), ensure_ascii=False, indent=1)
    print("\nveto rule:")
    for v in veto:
        print(f"  tau={v['tau']}: P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} (tp{v['tp']} fp{v['fp']} fn{v['fn']})")
    print("grid top6:")
    for g in grid[:6]:
        print(f"  wc{g['w_clip']} wd{g['w_dino']} wp{g['w_probe']}: P {g['precision']:.3f} R {g['recall']:.3f} F1 {g['f1']:.3f} (tp{g['tp']} fp{g['fp']} fn{g['fn']})")


if __name__ == "__main__":
    main()