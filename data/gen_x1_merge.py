#!/usr/bin/env python3
"""gen_x1 跨省扩展集后处理: 与 gen_full(1178店/2542图) sha 去重 → 独立编号 x1_* → 汇总。
用法: python3 gen_x1_merge.py
产物: gen_x1/merged_records.json (含 dedupe 统计); 不改动 gen_full 原有编号。"""
import json, os
from collections import Counter

DATA = os.path.dirname(os.path.abspath(__file__))
full = json.load(open(os.path.join(DATA, "gen_full", "gen_candidates.json")))
x1 = json.load(open(os.path.join(DATA, "gen_x1", "gen_candidates.json")))
full_sha = {r["sha256"] for r in full}

merged, dropped_sha, seen = [], 0, set()
for r in x1:
    if r["sha256"] in full_sha or r["sha256"] in seen:
        dropped_sha += 1
        continue
    seen.add(r["sha256"])
    r["image_id"] = "x1_" + r["image_id"].split("_")[-1].zfill(4)
    merged.append(r)

stats = {
    "new_raw": len(x1), "after_cross_dedupe": len(merged),
    "dropped_sha_vs_genfull_or_self": dropped_sha,
    "by_province": dict(Counter(r["province"] for r in merged)),
    "by_store": len({r["poi_id"] for r in merged}),
    "note": "与既有1178店候选经sha级去重;感知去重已在采集端完成(Hamming<=12);品牌核验待做",
}
json.dump({"stats": stats, "records": merged},
          open(os.path.join(DATA, "gen_x1", "merged_records.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
