#!/usr/bin/env python3
"""E10 第一层：OCR 品牌文字证据提取（GPT 裁决 A/P1）。
对 330 张评估图提取品牌文字证据：OWN（左邻右舍）/ COMPETITOR（竞品名）/ NONE。
纪律：OCR 无结果 ≠ 负例（img_0012 昏暗模糊会 OCR 失败）；仅作证据，不作硬门禁。
产物 vis_cache/e10_ocr_evidence.json。"""
import json, subprocess, re, os
from collections import Counter

ROOT = "/Users/mac/chatgpt-harvest/6af41562"
import sys; sys.path.insert(0, ROOT + "/synthetic")
import visual_baseline as vb
rows = vb.load_registry()
os.chdir(ROOT + "/data")
idx = json.load(open("sha_file_index.json"))

OWN = ["左邻右舍", "左邻·右舍", "左邻", "右舍", "good neighbors"]
COMPETITORS = {
    "美宜佳": ["美宜佳", "meiyijia", "meiyina"],
    "芙蓉兴盛": ["芙蓉兴盛", "芙蓉"],
    "邻几": ["邻几", "good spot"],
    "好邻居": ["好邻居"],
    "便利蜂": ["便利蜂"],
    "苏宁小店": ["苏宁"],
    "京东便利": ["京东"],
    "奥乐齐": ["奥乐齐", "aldi"],
    "7-11": ["7-eleven", "7-11", "seven eleven"],
    "全家": ["全家", "familymart"],
    "罗森": ["罗森", "lawson"],
    "快客": ["快客"],
    "十足": ["十足"],
    "天福": ["天福"],
    "邻儿": ["邻儿"],
    "凯烨": ["凯烨"],
    "佐邻佑邻": ["佐邻佑邻"],
    "美邻优": ["美邻优"],
    "甄选": ["甄选"],
}

def ocr(path):
    try:
        out = subprocess.run(["tesseract", path, "stdout", "-l", "chi_sim+eng", "--psm", "11"],
                             capture_output=True, text=True, timeout=30).stdout
        return re.sub(r"\s+", " ", out).strip()
    except Exception:
        return ""

results = []
cnt = Counter()
for i, r in enumerate(rows):
    p = r["path"]
    txt = ocr(p) if p and os.path.exists(p) else ""
    low = txt.lower()
    own_hit = [k for k in OWN if k in low]
    comp_hit = [b for b, kws in COMPETITORS.items() if any(k in low for k in kws)]
    if comp_hit:
        ev = "COMPETITOR"
    elif own_hit:
        ev = "OWN"
    else:
        ev = "NONE"
    cnt[ev] += 1
    results.append(dict(image_id=r["image_id"], label=r["label"],
                        neg_type=r.get("neg_type", ""), group=r["group"],
                        evidence=ev, own_text=own_hit, competitor=comp_hit,
                        ocr_text=txt[:150]))
    if (i + 1) % 60 == 0:
        print(f"  {i+1}/{len(rows)}", flush=True)

# 交叉表：证据 × 真实标签
xtab = {}
for ev in ["OWN", "COMPETITOR", "NONE"]:
    sub = [x for x in results if x["evidence"] == ev]
    xtab[ev] = dict(total=len(sub),
                    positive=sum(1 for x in sub if x["label"] == "positive"),
                    negative=sum(1 for x in sub if x["label"] == "negative"))
summary = dict(counts=dict(cnt), xtab=xtab,
               competitor_on_negative=[x["image_id"] for x in results
                                       if x["evidence"] == "COMPETITOR" and x["label"] == "negative"],
               own_on_negative=[x["image_id"] for x in results
                                if x["evidence"] == "OWN" and x["label"] == "negative"])
json.dump(dict(summary=summary, results=results),
          open(ROOT + "/synthetic/vis_cache/e10_ocr_evidence.json", "w"),
          ensure_ascii=False, indent=1)
print("SUMMARY:", json.dumps(summary, ensure_ascii=False))