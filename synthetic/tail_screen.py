#!/usr/bin/env python3
"""抓取长尾图 E8-G 冻结预筛分 (v2)。
目标 = registry_v3(499) 之外的 cNNNN_* 文件 (即 76 张未入账本长尾)。
画廊 = registry_v3 全部有文件正例 (32/36, 4 正例无文件剔除)。
三通道完全同构 e8g_stats: s=0.4*s_clip+0.4*s_dino_mc+0.2*s_probe, thr=0.636 严格>。
输出 vis_cache/crawl_tail_scores.json。判决不自动写 registry (治理)。"""
import json, os, re, sys
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
    reg = json.load(open(ROOT / "data/registry_v3.json"))
    reg_ids = {int(re.match(r"img_(\d+)", r["image_id"]).group(1)) for r in reg
               if re.match(r"img_(\d+)", r["image_id"])}

    # 画廊: registry_v3 正例 → 文件 (cNNNN_* 前缀匹配)
    files_by_num = {}
    for f in os.listdir(ROOT / "data/images"):
        m = re.match(r"c(\d{3,4})_", f)
        if m:
            files_by_num[int(m.group(1))] = f
    pos_rows, pos_missing = [], []
    for r in reg:
        if r["label"] != "positive":
            continue
        n = int(re.match(r"img_(\d+)", r["image_id"]).group(1))
        f = files_by_num.get(n)
        if f:
            pos_rows.append({"image_id": r["image_id"], "path": str(ROOT / "data/images" / f),
                             "label": "positive", "group": r.get("store_id") or r["image_id"]})
        else:
            pos_missing.append(r["image_id"])
    print(f"gallery positives: {len(pos_rows)} (no-file dropped: {pos_missing})", flush=True)

    tail = [{"image_id": f"tail_{n}", "path": str(ROOT / "data/images" / f),
             "label": "negative", "group": f"tail_{n}"}
            for n, f in sorted(files_by_num.items()) if n not in reg_ids]
    print(f"tail candidates: {len(tail)}", flush=True)

    rows = pos_rows + tail
    labels = np.array([r["label"] == "positive" for r in rows])

    # 全量活体嵌入 (CLIP full 用于 kNN; embed_all 附带 dino full/crop 供 probe)
    rf, ids_emb = e8.embed_all(rows, "tailall")
    id2i = {x: i for i, x in enumerate(ids_emb)}
    assert set(id2i) == {r["image_id"] for r in rows}, "embed id mismatch"
    s_clip = ef.knn_scores(rows, rf["clip_full"])

    s_dino_mc, _ = em.multicrop_dino_scores(rows)
    s_dino_mc = np.clip((s_dino_mc - 0.3) / 0.5, 0, 1)

    sr = e8.load_synth()
    sf, _ = e8.embed_all(sr, "synth", use_truth_crop=True)
    probe = e8.train_probes(sf, sr, seed=0)
    s_probe = probe["brand_clf"].predict_proba(probe["scaler"].transform(e8.concat_feats(rf)))[:, 1]

    s = 0.4 * s_clip + 0.4 * s_dino_mc + 0.2 * s_probe

    out = []
    for r, sc, clip_s in zip(rows, s, s_clip):
        if not r["image_id"].startswith("tail_"):
            continue
        out.append({"image_id": r["image_id"], "file": Path(r["path"]).name,
                    "score": round(float(sc), 4), "pred": bool(sc > THR),
                    "s_clip": round(float(clip_s), 4)})
    out.sort(key=lambda x: -x["score"])
    hits = [o for o in out if o["pred"]]
    json.dump({"thr": THR, "n_tail": len(out), "hits": len(hits), "ranked": out},
              open(CACHE / "crawl_tail_scores.json", "w"), ensure_ascii=False, indent=1)
    print(f"tail scored: {len(out)}, hits@0.636: {len(hits)}")
    for o in out[:20]:
        print(f"  {o['image_id']} {o['file']} score={o['score']} clip={o['s_clip']}")


if __name__ == "__main__":
    main()
