#!/usr/bin/env python3
"""E10 全量盲测对账：解析 14 批裁决 vs 真值密钥。
计算：① VLM 单独四态 vs 真值；② E10 拒识层（作用于 E8-G 预测正例）；
③ 组合系统 E8-G→E10（MATCH保留/NON_MATCH+OUT_OF_SCOPE拒/UNKNOWN回退E8-G）。
产物 vis_cache/e10_full/reconciliation.json + conflicts.json。"""
import json, re, os
import numpy as np

ROOT = "/Users/mac/chatgpt-harvest/6af41562/synthetic/vis_cache/e10_full"
key = json.load(open(f"{ROOT}/truth_key.json"))

# 解析所有裁决
verdicts = {}
for fn in sorted(os.listdir(ROOT)):
    m = re.match(r"verdict_batch(\d+)\.md", fn)
    if not m:
        continue
    for line in open(f"{ROOT}/{fn}"):
        mm = re.match(r"\|\s*(B\d\d_\d\d)\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|", line)
        if mm:
            verdicts[mm.group(1)] = dict(verdict=mm.group(2), text=mm.group(3))

print(f"parsed verdicts: {len(verdicts)} / {len(key)}")
missing = [a for a in key if a not in verdicts]
if missing:
    print("MISSING:", missing)

# ① VLM 单独四态 vs 真值（MATCH→正，其余→负）
vlm_tp = vlm_fp = vlm_fn = vlm_tn = 0
# ② E10 拒识层：仅对 E8-G 预测正例
e10_reject_tp = e10_reject_fp = 0  # 被拒的 TP/FP
# ③ 组合系统
comb_tp = comb_fp = comb_fn = comb_tn = 0
conflicts = []
records = []
for anon, k in key.items():
    v = verdicts.get(anon, {})
    verd = v.get("verdict", "MISSING")
    text = v.get("text", "")
    truth = k["label"] == "positive"
    e8pos = k["e8g_pred"]
    # VLM 单独
    vlm_pos = verd == "MATCH"
    if truth and vlm_pos: vlm_tp += 1
    elif truth and not vlm_pos: vlm_fn += 1
    elif not truth and vlm_pos: vlm_fp += 1
    else: vlm_tn += 1
    # 组合：MATCH→正；NON_MATCH/OUT_OF_SCOPE→负；UNKNOWN/MISSING→回退E8-G
    if verd == "MATCH":
        final = True
    elif verd in ("NON_MATCH", "OUT_OF_SCOPE"):
        final = False
    else:
        final = e8pos
    if truth and final: comb_tp += 1
    elif truth and not final: comb_fn += 1
    elif not truth and final: comb_fp += 1
    else: comb_tn += 1
    # E10 拒识层对 E8-G 正例的作用
    if e8pos and verd in ("NON_MATCH", "OUT_OF_SCOPE"):
        if truth: e10_reject_tp += 1
        else: e10_reject_fp += 1
    # 冲突：VLM 裁决与真值不符（用于独立复核）
    if vlm_pos != truth:
        conflicts.append(dict(anon=anon, image_id=k["image_id"], path=k["path"],
                              truth=k["label"], neg_type=k["neg_type"], group=k["group"],
                              e8g_score=k["e8g_score"], e8g_pred=e8pos,
                              vlm_verdict=verd, vlm_text=text))
    records.append(dict(anon=anon, image_id=k["image_id"], truth=k["label"],
                        neg_type=k["neg_type"], e8g_pred=e8pos, e8g_score=k["e8g_score"],
                        vlm_verdict=verd, vlm_text=text, combined_final=bool(final)))

def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=round(p, 3), recall=round(r, 3),
                f1=round(2 * p * r / max(p + r, 1e-9), 3))

out = dict(
    n=len(key),
    vlm_alone=prf(vlm_tp, vlm_fp, vlm_fn),
    vlm_alone_tn=vlm_tn,
    combined_e8g_e10=prf(comb_tp, comb_fp, comb_fn),
    combined_tn=comb_tn,
    e10_rejection_on_e8g_pos=dict(rejected_tp=e10_reject_tp, rejected_fp=e10_reject_fp),
    verdict_distribution={v: sum(1 for r in records if r["vlm_verdict"] == v)
                          for v in ["MATCH", "NON_MATCH", "UNKNOWN", "OUT_OF_SCOPE", "MISSING"]},
    n_conflicts=len(conflicts))
json.dump(out, open(f"{ROOT}/reconciliation.json", "w"), ensure_ascii=False, indent=1)
json.dump(conflicts, open(f"{ROOT}/conflicts.json", "w"), ensure_ascii=False, indent=1)
json.dump(records, open(f"{ROOT}/records.json", "w"), ensure_ascii=False, indent=1)

print("\n=== VLM 单独（MATCH→正）===")
print(out["vlm_alone"], "TN", vlm_tn)
print("=== 组合 E8-G→E10（UNKNOWN回退）===")
print(out["combined_e8g_e10"], "TN", comb_tn)
print("=== E10 拒识层对 E8-G 正例 ===")
print(out["e10_rejection_on_e8g_pos"])
print("=== 裁决分布 ===")
print(out["verdict_distribution"])
print(f"\n冲突样本（VLM vs 真值）: {len(conflicts)}")
for c in conflicts:
    print(f"  {c['anon']} {c['image_id']} truth={c['truth']} neg={c['neg_type']} "
          f"e8g={c['e8g_score']} vlm={c['vlm_verdict']} | {c['vlm_text'][:40]}")