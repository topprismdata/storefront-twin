#!/usr/bin/env python3
"""e14 — 最终探针成品 (部署件)。
训练集: 合成 820 + registry_v11 全部代表行(真实, 含 E10 评测行——评测数字由 E6-v3 LOSO 建立,
本探针是部署工件, 不产生新的评测宣称)。授权: GD-2026-11-14-01。
输出 vis_cache/e14_sign/final_probe.npz + model_card.json。"""
import json, sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import e8_train as e8

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
OUT = ROOT / "synthetic/vis_cache/e14_sign"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    reg = json.load(open(ROOT / "data/registry_v11.json"))
    sha_idx = json.load(open(ROOT / "data/sha_file_index.json"))
    rows = []
    for r in reg:
        if not r.get("is_representative"):
            continue
        p = sha_idx.get(r["sha256"])
        if not p:
            continue
        fp = ROOT / "data" / p
        if fp.exists():
            rows.append({"image_id": r["image_id"], "path": str(fp), "label": r["label"],
                         "group": r.get("store_id") or r["image_id"]})
    rf, _ = e8.embed_all(rows, "v4_real")
    X_real = e8.concat_feats(rf)
    y_real = np.array([r["label"] == "positive" for r in rows])

    sr = e8.load_synth()
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    X_syn = e8.concat_feats(sf)
    y_syn = np.array([r["label"] == "positive" for r in sr])

    X = np.vstack([X_syn, X_real])
    y = np.concatenate([y_syn, y_real])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0).fit(sc.transform(X), y)
    acc = clf.score(sc.transform(X), y)
    print(f"final probe: train n={len(y)} ({int(y.sum())} pos, real {len(y_real)}), in-sample acc {acc:.4f}")

    np.savez(OUT / "final_probe.npz", coef=clf.coef_, intercept=clf.intercept_,
             mean=sc.mean_, scale=sc.scale_,
             feat_layout=np.array(["clip_full", "dinov2_full", "dinov2_crop"]))
    card = {
        "artifact": "final_probe.npz", "date": "2026-11-14",
        "decision": "GD-2026-11-14-01 (研究训练放行; 部署仍需正式权利链)",
        "architecture": "CLIP-full 512 + DINOv2-full 768 + DINOv2-crop 768 (各L2归一) = 2048维 → StandardScaler → LogisticRegression",
        "training": {"synth": int(len(y_syn)), "real_registry_v11": int(len(y_real)),
                     "pos_total": int(y.sum()), "in_sample_acc": round(float(acc), 4)},
        "eval_provenance": "泛化数字以 E6-v3 店组LOSO为准(嵌套0.905/操作点0.80→0.916/配对ΔF1+0.043显著); 本工件不另作评测宣称",
        "thresholds": {"collect_screen": 0.636, "deploy_op": 0.80},
        "known_limits": ["multicrop 通道含 RNG(~0.1 分数噪声)", "间隔点低分辨率敏感(E10 已证)", "真实标签为双通道仲裁, rights=pending_review"],
    }
    json.dump(card, open(OUT / "model_card.json", "w"), ensure_ascii=False, indent=1)
    print("落盘", OUT)


if __name__ == "__main__":
    main()
