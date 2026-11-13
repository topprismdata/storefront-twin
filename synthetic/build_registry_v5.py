#!/usr/bin/env python3
"""registry v5: +12 张高德 POI 相册正例 (2026-11-13 v4 候选筛选, 双通道仲裁 conv 6af692a2 12/12 A)。
新行 image_id img_0580-0591, store_id=poi_<id>, sha 入 sha_file_index。
registry_v4 不动。"""
import json, hashlib, re
from pathlib import Path

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
ARB_NOTE = ("2026-11-13 双通道仲裁入正例: 本地全分辨率目检 + GPT盲裁"
            "(arb_ownbrand_v5.jpg, conv 6af692a2-e3fc-83e8-be8a-5024ad19c8e2, 12/12 裁A, 对照双正确); "
            "来源=高德POI相册 REST v5 采集(key 7cef...), E8G预筛@0.636 13命中12真(0335效果图否决)")

# file, poi_id, poi_name, city, score
NEW = [
    ("v4_0776_13dd0f7aba53.jpg", None, None, None, 0.760),
    ("v4_0629_847080f393bf.jpg", None, None, None, 0.7586),
    ("v4_0468_1fac922f2b27.jpg", None, None, None, 0.680),
    ("v4_0487_aac9ad30cc87.jpg", None, None, None, 0.680),
    ("v4_0656_de99c2f43193.jpg", None, None, None, 0.680),
    ("v4_0542_28cc02c52c4f.jpg", None, None, None, 0.6794),
    ("v4_0154_8e2cc1ac672d.jpg", None, None, None, 0.679),
    ("v4_0318_95cc8a2addb2.jpg", None, None, None, 0.6781),
    ("v4_0774_6d49d2de929d.jpg", None, None, None, 0.6701),
    ("v4_0796_3e52a6ed9541.jpg", None, None, None, 0.6584),
    ("v4_1369_1ca80ada4dca.jpg", None, None, None, 0.6578),
    ("v4_0587_230cc4d1963e.jpg", None, None, None, 0.6372),
]

def main():
    log = json.load(open(ROOT / "data/amap_v4_download_log.json"))
    byfile = {x["file"]: x for x in log if x.get("file")}
    reg = json.load(open(ROOT / "data/registry_v4.json"))
    sha_idx = json.load(open(ROOT / "data/sha_file_index.json"))
    reg_ids = {int(m.group(1)) for r in reg for m in [re.match(r"img_(\d+)", r["image_id"])] if m}
    next_id = max(reg_ids) + 1
    added = []
    for fname, _n, _pn, _c, score in NEW:
        meta = byfile.get(fname)
        assert meta, fname
        fp = ROOT / "data/images_v4" / fname
        sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        rel = f"data/images_v4/{fname}"
        sha_idx[sha] = rel
        iid = f"img_{next_id:04d}"
        next_id += 1
        reg.append({
            "image_id": iid, "label": "positive", "neg_type": "", "brand_state": "verified_own",
            "store_id": f"poi_{meta['poi']}", "src_url": meta["url"], "landing_page": "",
            "source_group": f"amap_poi_{meta['poi']}", "photo_family": "",
            "sha256": sha, "perceptual_hash": "", "w": 0, "h": 0,
            "rights_status": "pending_review", "source_license": "amap_poi_album",
            "capture_date": "", "geo_bucket": meta.get("city", ""), "franchise_type": None,
            "evidence_level": "dual_channel_arbitrated", "split": "unassigned",
            "features": {}, "is_representative": True, "dup_group_size": 1,
            "brand_verdict_fullres": "verified_own_brand_storefront",
            "sign_text_fullres": "左邻·右舍", "style_features_fullres": "",
            "fullres_confidence": "high", "fullres_note": ARB_NOTE + f" | E8G预筛分 {score}",
            "candidate_style": "",
        })
        added.append({"image_id": iid, "file": fname, "poi": meta["poi"], "poi_name": meta["poi_name"],
                      "city": meta.get("city", ""), "score": score})
    json.dump(reg, open(ROOT / "data/registry_v5.json", "w"), ensure_ascii=False, indent=1)
    json.dump(sha_idx, open(ROOT / "data/sha_file_index.json", "w"), ensure_ascii=False, indent=1)
    rep_pos = sum(1 for r in reg if r["label"] == "positive" and r.get("is_representative"))
    print(f"registry_v5: +{len(added)} 正例, 评测代表正例总数 {rep_pos}")
    for a in added:
        print(" ", a["image_id"], a["city"], a["poi_name"][:20], a["score"])

if __name__ == "__main__":
    main()
