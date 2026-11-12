#!/usr/bin/env python3
"""E10 v2（加强指令）对账 + v1→v2 迁移分析。
解析 v2_verdict_batch*.md（已完成批次），独立输出 v2_*，不覆盖 v1。
重点：GPT 评审 D 项要求的迁移检查（NON_MATCH→UNKNOWN 是否让竞品回退漏拒）。
产物 v2_records.json / v2_reconciliation.json / v2_migration.json。"""
import json, re, os
import numpy as np

ROOT = "/Users/mac/chatgpt-harvest/6af41562/synthetic/vis_cache/e10_full"
key = json.load(open(f"{ROOT}/truth_key.json"))
v1rec = {r["anon"]: r for r in json.load(open(f"{ROOT}/records.json"))}

# 解析 v2 裁决
verdicts = {}
batches_done = []
for fn in sorted(os.listdir(ROOT)):
    m = re.match(r"v2_verdict_batch(\d+)\.md", fn)
    if not m:
        continue
    batches_done.append(int(m.group(1)))
    for line in open(f"{ROOT}/{fn}"):
        mm = re.match(r"\|\s*(B\d\d_\d\d)\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|", line)
        if mm:
            verdicts[mm.group(1)] = dict(verdict=mm.group(2), text=mm.group(3))

covered = [a for a in key if a in verdicts]
print(f"v2 batches done: {batches_done}")
print(f"v2 verdicts: {len(verdicts)} / {len(key)} (covered {len(covered)})")

REJ = {"NON_MATCH", "OUT_OF_SCOPE"}


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=round(p, 3), recall=round(r, 3),
                f1=round(2 * p * r / max(p + r, 1e-9), 3))


# 指标（仅在 v2 覆盖样本上）
vlm = [0, 0, 0, 0]; A = [0, 0, 0, 0]; B = [0, 0, 0, 0]
records = []; conflicts = []
for anon in covered:
    k = key[anon]; v = verdicts[anon]
    verd = v["verdict"]; text = v["text"]
    truth = k["label"] == "positive"; e8 = k["e8g_pred"]
    vpos = verd == "MATCH"
    # VLM alone
    idx = 0 if (truth and vpos) else 1 if (truth and not vpos) else 2 if (not truth and vpos) else 3
    vlm[idx] += 1
    # Mode A 拒识-only
    fa = e8 and verd not in REJ
    idx = 0 if (truth and fa) else 1 if (truth and not fa) else 2 if (not truth and fa) else 3
    A[idx] += 1
    # Mode B 拒识+救援
    fb = True if verd == "MATCH" else False if verd in REJ else e8
    idx = 0 if (truth and fb) else 1 if (truth and not fb) else 2 if (not truth and fb) else 3
    B[idx] += 1
    if vpos != truth:
        conflicts.append(dict(anon=anon, image_id=k["image_id"], truth=k["label"],
                              neg_type=k["neg_type"], e8g_score=k["e8g_score"],
                              vlm_verdict=verd, vlm_text=text))
    records.append(dict(anon=anon, image_id=k["image_id"], truth=k["label"],
                        neg_type=k["neg_type"], group=k["group"], e8g_pred=e8,
                        e8g_score=k["e8g_score"], vlm_verdict=verd, vlm_text=text,
                        modeA_final=bool(fa), modeB_final=bool(fb)))

# v1→v2 迁移矩阵（覆盖样本）
mig = {}; mig_detail = []
for anon in covered:
    v1v = v1rec[anon]["vlm_verdict"]; v2v = verdicts[anon]["verdict"]
    mig[f"{v1v}->{v2v}"] = mig.get(f"{v1v}->{v2v}", 0) + 1
    if v1v != v2v:
        k = key[anon]; e8 = k["e8g_pred"]; truth = k["label"] == "positive"
        # 迁移对 Mode B 的影响
        fb1 = True if v1v == "MATCH" else False if v1v in REJ else e8
        fb2 = True if v2v == "MATCH" else False if v2v in REJ else e8
        mig_detail.append(dict(anon=anon, image_id=k["image_id"], truth=k["label"],
                               neg_type=k["neg_type"], e8g_pred=e8, e8g_score=k["e8g_score"],
                               v1=v1v, v2=v2v, B_v1=bool(fb1), B_v2=bool(fb2),
                               flipped=fb1 != fb2))

out = dict(n_covered=len(covered), batches_done=batches_done,
           vlm_alone=prf(*vlm[:3]), modeA=prf(*A[:3]), modeB=prf(*B[:3]),
           vlm_tn=vlm[3], modeA_tn=A[3], modeB_tn=B[3],
           n_conflicts=len(conflicts))
json.dump(out, open(f"{ROOT}/v2_reconciliation.json", "w"), ensure_ascii=False, indent=1)
json.dump(records, open(f"{ROOT}/v2_records.json", "w"), ensure_ascii=False, indent=1)
json.dump(dict(matrix=mig, changes=mig_detail), open(f"{ROOT}/v2_migration.json", "w"), ensure_ascii=False, indent=1)

print("\n=== v2 VLM alone ===", out["vlm_alone"])
print("=== v2 Mode A ===", out["modeA"])
print("=== v2 Mode B ===", out["modeB"])
print(f"\n冲突: {len(conflicts)}")
for c in conflicts:
    print(f"  {c['image_id']} truth={c['truth']} neg={c['neg_type']} vlm={c['vlm_verdict']} | {c['vlm_text'][:30]}")
print("\n=== v1→v2 迁移矩阵 ===")
for kk, vv in sorted(mig.items(), key=lambda x: -x[1]):
    print(f"  {kk}: {vv}")
print(f"\n=== 改变 Mode B 结果的迁移 ({sum(1 for d in mig_detail if d['flipped'])}) ===")
for d in mig_detail:
    if d["flipped"]:
        print(f"  {d['image_id']} truth={d['truth']} neg={d['neg_type']} e8={d['e8g_pred']} {d['v1']}->{d['v2']} B:{d['B_v1']}->{d['B_v2']}")