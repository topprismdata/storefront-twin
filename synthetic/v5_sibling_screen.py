#!/usr/bin/env python3
"""v5 兄弟照片二轮筛选: 340 品牌 POI 的 359 张未筛照片 vs 画廊=50 v5 正例。
同一公式/阈值, 输出 vis_cache/v5_sibling_scores.json。"""
import json, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import visual_baseline as vb
import e8_train as e8
import e8_fusion as ef
import e8_multicrop as em

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
CACHE = vb.CACHE
THR = 0.636


def main():
    reg = json.load(open(ROOT / "data/registry_v5.json"))
    sha_idx = json.load(open(ROOT / "data/sha_file_index.json"))
    rows = []
    for r in reg:
        if not r.get("is_representative") or r["label"] != "positive":
            continue
        p = sha_idx.get(r["sha256"])
        if not p:
            continue
        fp = ROOT / "data" / p
        if fp.exists():
            rows.append({"image_id": r["image_id"], "path": str(fp), "label": "positive",
                         "group": r.get("store_id") or r["image_id"]})
    print(f"gallery: {len(rows)}", flush=True)

    sibs = json.load(open(ROOT / "data/amap_v4_siblings.json"))
    crows = [{"image_id": f"s5_{x['file'].split('_')[1]}", "path": str(ROOT / "data/images_v4" / x["file"]),
              "label": "negative", "group": f"poi_{x['poi']}",
              "poi": x["poi"], "poi_name": x["poi_name"], "city": x["city"]} for x in sibs]
    print(f"siblings: {len(crows)}", flush=True)

    allrows = rows + crows
    rf, ids_emb = e8.embed_all(allrows, "v5sib")
    s_clip = ef.knn_scores(allrows, rf["clip_full"])
    s_dino_mc, _ = em.multicrop_dino_scores(allrows)
    s_dino_mc = np.clip((s_dino_mc - 0.3) / 0.5, 0, 1)
    sr = e8.load_synth()
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    probe = e8.train_probes(sf, sr, seed=0)
    s_probe = probe["brand_clf"].predict_proba(probe["scaler"].transform(e8.concat_feats(rf)))[:, 1]
    s = 0.4 * s_clip + 0.4 * s_dino_mc + 0.2 * s_probe

    out = []
    for r, sc, c_s in zip(allrows, s, s_clip):
        if not r["image_id"].startswith("s5_"):
            continue
        out.append({"image_id": r["image_id"], "file": Path(r["path"]).name, "score": round(float(sc), 4),
                    "pred": bool(sc > THR), "s_clip": round(float(c_s), 4),
                    "poi": r["poi"], "poi_name": r["poi_name"], "city": r["city"]})
    out.sort(key=lambda x: -x["score"])
    hits = [o for o in out if o["pred"]]
    json.dump({"thr": THR, "n": len(out), "hits": len(hits), "ranked": out},
              open(CACHE / "v5_sibling_scores.json", "w"), ensure_ascii=False, indent=1)
    print(f"scored {len(out)}, hits@0.636: {len(hits)}")
    for o in out[:30]:
        print(f"  {o['image_id']} {o['score']:.3f} clip={o['s_clip']:.1f} {o['city']} {o['poi_name'][:18]}")


if __name__ == "__main__":
    main()
