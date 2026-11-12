#!/usr/bin/env python3
"""合并 v8 素材 VLM 四态批测 → stock_benchmark.json + 报告数据。
用法: python3 synthetic/merge_stock_vlm.py
读取 data/annotation_pack/stock_sheets/v8eval_stock_00-10.md,解析表格行,
徽章 S00121 → mid S0121(数字补零4位),写回 stock_benchmark.json 的 vlm 字段,
输出 stock_vlm_summary.json(VLM-alone + 误收清单)。
"""
import json, re, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
SHEET_DIR = ROOT / "data/annotation_pack/stock_sheets"
STATE_RE = re.compile(r"\b(MATCH|NON_MATCH|UNKNOWN|OUT_OF_SCOPE)\b", re.I)
ROW_RE = re.compile(r"^\s*\|?\s*(S\d{4,6})\s*\|", re.M)

def norm_mid(badge):
    return "S" + str(int(badge[1:])).zfill(4)

# 徽章后跟 | 或 : 再跟四态词(容忍粗体/空格/表格线),全局扫描兼容三种格式:
# 1) | S00121 | UNKNOWN | ...  2) | S00215 | OUT_OF_SCOPE | S00406 | UNKNOWN | (双列对)
# 3) S00234: OUT_OF_SCOPE (纯行)
PAIR_RE = re.compile(r"S\d{4,6}\s*[|:]\s*\**\s*(MATCH|NON_MATCH|UNKNOWN|OUT_OF_SCOPE)\b", re.I)

def parse_verdicts(txt):
    out = {}
    for m in PAIR_RE.finditer(txt):
        badge = re.match(r"S\d{4,6}", m.group(0)).group(0)
        out[norm_mid(badge)] = m.group(1).upper()
    return out

def main():
    bench_path = ROOT / "data/stock_benchmark.json"
    bench = json.load(open(bench_path))
    rows = {r["mid"]: r for r in bench["rows"]}
    verdicts, missing_files = {}, []
    for i in range(11):
        f = SHEET_DIR / f"v8eval_stock_{i:02d}.md"
        if not f.exists():
            missing_files.append(f.name)
            continue
        txt = f.read_text()
        for mid, v in parse_verdicts(txt).items():
            verdicts[mid] = v
    if missing_files:
        print(f"缺少批次文件: {missing_files} —— 先跑完 v8 再合并"); sys.exit(1)

    hit = sum(1 for mid in verdicts if mid in rows)
    unknown_mid = [m for m in verdicts if m not in rows]
    # 写回
    for r in bench["rows"]:
        v = verdicts.get(r["mid"])
        if v:
            r["vlm_verdict_v8"] = v
    bench["vlm_v8"] = {"n_verdicts": len(verdicts), "merged": hit, "version": "v8", "date": "2026-11-13"}
    json.dump(bench, open(bench_path, "w"), ensure_ascii=False, indent=1)

    # 汇总: VLM-alone(对 B 素材全负例域: 任何 MATCH=误收)
    subs = Counter()
    fp_list = []
    for mid, v in verdicts.items():
        subs[v] += 1
        if v == "MATCH":
            fp_list.append(mid)
    summary = {
        "n_verdicts": len(verdicts),
        "verdict_dist": dict(subs),
        "match_false_accepts": fp_list,
        "false_accept_rate": round(len(fp_list) / max(1, len(verdicts)), 4),
        "unmapped_ids": unknown_mid[:20],
    }
    json.dump(summary, open(ROOT / "synthetic/vis_cache/stock_vlm_summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
