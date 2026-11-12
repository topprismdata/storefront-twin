#!/usr/bin/env python3
"""量化 render_stock 负例标签噪声：OCR 招牌文字，匹配竞品/本品牌关键词。
产物 vis_cache/render_stock_audit.json。探索性，不改 registry。"""
import json, subprocess, re, os
from collections import Counter

ROOT = "/Users/mac/chatgpt-harvest/6af41562"
os.chdir(ROOT + "/data")
reg = json.load(open("registry_v3.json"))
rows = reg if isinstance(reg, list) else reg.get("rows", reg.get("images", []))
idx = json.load(open("sha_file_index.json"))
rs = [r for r in rows if r.get("neg_type") == "render_stock"]

# 竞品品牌关键词（含 OCR 常见变体）+ 本品牌
COMPETITORS = {
    "美宜佳": ["美宜佳", "MEIYIJIA", "MEIYINA", "美宜"],
    "芙蓉兴盛": ["芙蓉兴盛", "芙蓉", "兴盛"],
    "邻几": ["邻几", "GOOD SPOT", "GOODSPOT"],
    "好邻居": ["好邻居"],
    "便利蜂": ["便利蜂"],
    "苏宁小店": ["苏宁小店", "苏宁"],
    "京东便利": ["京东便利", "京东"],
    "奥乐齐": ["奥乐齐", "ALDI"],
    "7-11": ["7-11", "7-ELEVEN", "711", "SEVEN"],
    "全家": ["全家", "FamilyMart"],
    "罗森": ["罗森", "LAWSON"],
    "快客": ["快客"],
    "十足": ["十足"],
    "天福": ["天福"],
    "唐久": ["唐久"],
    "邻儿": ["邻儿"],
    "凯烨": ["凯烨"],
    "佐邻佑邻": ["佐邻佑邻"],
    "美邻优": ["美邻优"],
}
OWN = ["左邻右舍", "左邻", "右舍", "Good Neighbors", "GOOD NEIGHBORS"]

def ocr(path):
    try:
        out = subprocess.run(["tesseract", path, "stdout", "-l", "chi_sim+eng", "--psm", "11"],
                             capture_output=True, text=True, timeout=30).stdout
        return re.sub(r"\s+", " ", out).strip()
    except Exception:
        return ""

results = []
comp_hits = Counter()
own_hits = []
for i, r in enumerate(rs):
    p = idx.get(r["sha256"])
    if not p or not os.path.exists(p):
        results.append(dict(image_id=r["image_id"], text="", match="NO_PATH"))
        continue
    txt = ocr(p)
    match = "none"
    for brand, kws in COMPETITORS.items():
        if any(k.lower() in txt.lower() for k in kws):
            match = "competitor:" + brand
            comp_hits[brand] += 1
            break
    if match == "none" and any(k.lower() in txt.lower() for k in OWN):
        match = "OWN_BRAND"
        own_hits.append(r["image_id"])
    results.append(dict(image_id=r["image_id"], text=txt[:120], match=match))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(rs)}", flush=True)

summary = dict(
    total=len(rs),
    competitor_hits=dict(comp_hits),
    competitor_total=sum(comp_hits.values()),
    own_brand_hits=own_hits,
    own_brand_total=len(own_hits),
    no_text=sum(1 for x in results if not x["text"]),
)
json.dump(dict(summary=summary, results=results),
          open(ROOT + "/synthetic/vis_cache/render_stock_audit.json", "w"),
          ensure_ascii=False, indent=1)
print("SUMMARY:", json.dumps(summary, ensure_ascii=False))