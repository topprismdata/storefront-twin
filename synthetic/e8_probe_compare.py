#!/usr/bin/env python3
"""探针域随机化对比实验：v1(窄域) vs v2(宽域+纹理+传感器噪声) 合成训练探针。
核心问题：v2 宽域孪生能否把合成探针 AuPR 从 v1 的 0.324 拉回 0.5+？
方法：同一真实评估集(330)，同一嵌入，仅换合成训练源。线性+MLP 探针。
产物 vis_cache/probe_v1v2_compare.json。探索性，不改 registry/冻结模型。"""
import json
import numpy as np
import visual_baseline as vb
import e8_train as e8
import e9_fusion as e9f

ROOT = vb.ROOT
CACHE = vb.CACHE


def load_synth_from(renders_dir):
    rows = []
    mf = renders_dir / "manifest_3d.jsonl"
    for line in open(mf):
        r = json.loads(line)
        fp = renders_dir / r["image"]
        if not fp.exists():
            continue
        r["path"] = str(fp)
        r["group"] = f"synth_{r['seed']}"
        r["label"] = "positive" if r["brand"] == "left_right" else "negative"
        r["candidate_style"] = r["style_id"] or ""
        rows.append(r)
    return rows


def probe_auprc(real_rows, rf, synth_rows, sf, kind="linear"):
    probe = e8.train_probes(sf, synth_rows, seed=0)
    X = probe["scaler"].transform(e8.concat_feats(rf))
    key = "brand_clf" if kind == "linear" else "mlp_brand"
    pb = probe[key].predict_proba(X)[:, 1]
    labels = np.array([r["label"] == "positive" for r in real_rows])
    return float(e9f.auprc(pb, labels)), pb


def main():
    real_rows = vb.load_registry()
    rf, _ = e8.embed_all(real_rows, "real")
    labels = np.array([r["label"] == "positive" for r in real_rows])
    out = {"n_real": len(real_rows), "n_pos": int(labels.sum()), "versions": {}}
    for ver, dirname in [("v1", "renders_3d"), ("v2", "renders_3d_v2")]:
        sr = load_synth_from(ROOT / dirname)
        if not sr:
            print(f"{ver}: NO SYNTH at {dirname}, skip"); continue
        sf, _ = e8.embed_all(sr, f"synth_{ver}", use_truth_crop=True)
        n_pos = sum(1 for r in sr if r["label"] == "positive")
        lin, pb_lin = probe_auprc(real_rows, rf, sr, sf, "linear")
        mlp, pb_mlp = probe_auprc(real_rows, rf, sr, sf, "mlp")
        out["versions"][ver] = dict(n_synth=len(sr), n_synth_pos=n_pos,
                                    auprc_linear=round(lin, 3), auprc_mlp=round(mlp, 3))
        print(f"{ver}: synth={len(sr)}(pos{n_pos}) AuPR linear={lin:.3f} mlp={mlp:.3f}", flush=True)
    json.dump(out, open(CACHE / "probe_v1v2_compare.json", "w"), ensure_ascii=False, indent=1)
    print("\n=== 探针对比 (AuPR, 0.5=随机) ===")
    for ver, d in out["versions"].items():
        print(f"  {ver}: linear={d['auprc_linear']} mlp={d['auprc_mlp']}")
    if "v1" in out["versions"] and "v2" in out["versions"]:
        dl = out["versions"]["v2"]["auprc_linear"] - out["versions"]["v1"]["auprc_linear"]
        print(f"  v2-v1 linear ΔAuPR = {dl:+.3f}")


if __name__ == "__main__":
    main()