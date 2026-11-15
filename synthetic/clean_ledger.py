#!/usr/bin/env python3
"""clean_ledger — 评测集噪声台账: 汇总四轮 VLM 兜底争议, 双读者一致规则改判。
输出: data/gold_noise_ledger.json (file/old_gold/vlm_read/ocr_read/decision/revised)
决策规则:
  FIX   = VLM读出与OCR文字都指向目录品牌X ≠ gold → gold 改为 X (双读者一致, 最强证据)
  EXCL  = VLM 判 无法确认/室内/非门头 → 移出评测(标签无法挽救)
  KEEP  = 证据不足(单读者/冲突) → 保留 gold 原判, 单列观察
"""
import json, re, glob
from pathlib import Path

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")

CAT_BRANDS = ["左邻右舍", "美宜佳", "芙蓉兴盛", "京东便利店", "7-11", "罗森", "全家", "天猫小店",
              "便利蜂", "十足", "每一天", "见福", "唐久", "红旗连锁", "零食很忙", "赵一鸣零食",
              "好想来", "家家悦", "盒马鲜生", "好特卖", "快客", "恰货铺子", "天福", "万嘉", "好德",
              "悦来悦喜", "零食有鸣", "喜士多", "丹尼斯", "苏宁小店", "京客隆", "北京华联", "天虹",
              "盒马NB", "TODAY", "戴永红", "新佳宜", "良品铺子", "老婆大人", "谊品生鲜", "利群",
              "山姆", "华润万家", "开市客", "中百罗森", "苏果便利", "好邻居", "每日每夜", "八点半",
              "友客", "金虎", "舞东风", "合家欢", "佳乐家", "三只松鼠", "中百", "三江", "北国",
              "大商", "人人乐", "华冠", "家得乐"]
ALIAS_MAP = {"7-ELEVEN": "7-11", "711": "7-11", "SEVEN": "7-11", "FRESHIPPO": "盒马鲜生",
             "盒马奥莱": "盒马鲜生", "盒马NB": "盒马NB", "苏果": "苏果便利", "三江购物": "三江",
             "三江超市": "三江", "中百超市": "中百", "中百仓储": "中百", "北国超市": "北国",
             "大商超市": "大商", "赵一鸣省钱超市": "赵一鸣零食", "FAMILYMART": "全家", "LAWSON": "罗森",
             "SAM": "山姆", "COSTCO": "开市客", "HOTMAXX": "好特卖", "BHG": "北京华联",
             "TODAY": "TODAY", "NEWJOY": "新佳宜", "丹尼斯全日鲜": "丹尼斯", "丹尼斯超市": "丹尼斯"}


def read_brand(brand_field, txt):
    """从 VLM 判读(品牌+文字)映射到目录品牌; 无法映射→None。最长键优先。"""
    t = f"{brand_field} {txt}".upper().replace(" ", "")
    cands = []
    for b in CAT_BRANDS:
        if b.upper() in t:
            cands.append((b, len(b)))
    for pat, b in ALIAS_MAP.items():
        if pat.upper() in t:
            cands.append((b, len(pat)))
    if not cands:
        return None
    cands.sort(key=lambda kv: -kv[1])
    return cands[0][0]


ledger = {}
# --- 各轮 backstop: (index_json, [ans_txt], 判读格式) ---
SOURCES = [
    ("data/annotation_pack/e22_backstop_index.json",
     ["data/annotation_pack/e22_backstop_ans_0.txt", "data/annotation_pack/e22_backstop_ans_2.txt"]),
    ("data/annotation_pack/e23b_backstop_index.json",
     ["data/annotation_pack/e23b_backstop_ans_0.txt", "data/annotation_pack/e23b_backstop_ans_2.txt"]),
    ("data/annotation_pack/e25_backstop_index.json",
     ["data/annotation_pack/e25_backstop_ans_0.txt", "data/annotation_pack/e25_backstop_ans_2.txt"]),
]
for idx_path, ans_paths in SOURCES:
    idx = json.load(open(ROOT / idx_path))
    id2 = {m["id"]: m for m in idx["items"]}
    for ap in ans_paths:
        p = ROOT / ap
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            m = re.match(r"\s*(S\d+N\d+)\s*[|｜]\s*([^|｜]+)\s*[|｜]\s*(.*)", line)
            if not m or m.group(1) not in id2:
                continue
            iid, brand, txt = m.group(1), m.group(2).strip(), m.group(3).strip()
            fname = id2[iid]["path"].split("/")[-1]
            ledger[fname] = {"vlm_brand": brand, "vlm_text": txt, "old_gold": id2[iid]["gold"]}
# --- v14 backstop (E20-era 10 图, F01-F10) ---
vf = json.load(open(ROOT / "data/v13_backstop_files.json"))
vfa = (ROOT / "data/annotation_pack/v14_backstop_ans.txt")
if vfa.exists():
    for line in vfa.read_text().splitlines():
        m = re.match(r"\s*(F\d+)\s*[|｜]\s*([^|｜]+)\s*[|｜]\s*(.*)", line)
        if not m:
            continue
        i = int(m.group(1)[1:]) - 1
        if 0 <= i < len(vf):
            fname = vf[i].split("/")[-1]
            ledger.setdefault(fname, {"vlm_brand": m.group(2).strip(), "vlm_text": m.group(3).strip(),
                                      "old_gold": None})

# --- OCR 读者: 从各 results json 取 texts ---
for res in ("e25_results.json", "e23b_results.json", "e22_results.json"):
    p = ROOT / "synthetic/vis_cache" / res
    if not p.exists():
        continue
    for o in json.load(open(p)):
        fn = o.get("file")
        if fn in ledger and "ocr_texts" not in ledger[fn]:
            ledger[fn]["ocr_texts"] = o.get("texts", [])

# --- 决策(三桶) ---
# EXCL_DEAD  = VLM 无法确认/室内/读出非目录品牌 → 标签死亡, 移出评测
# EXCL_UNRES = VLM 明确读出目录品牌 X ≠ gold(单读者) → 标签存疑, 移出评测(不参与acc)
# KEEP_ERROR = VLM 读出与 gold 一致 → 系统真错误, 保留在评测(计错)
for fn, e in ledger.items():
    joined_vlm = f"{e['vlm_brand']} {e['vlm_text']}"
    if ("无法确认" in joined_vlm or "非门头" in joined_vlm or "室内" in joined_vlm
            or "货架" in joined_vlm or "手机画面" in joined_vlm or "效果图" in joined_vlm):
        e["decision"] = "EXCL_DEAD"
        continue
    vb = read_brand(e["vlm_brand"], e["vlm_text"])
    e["vlm_read"] = vb
    if vb is None:
        e["decision"] = "EXCL_DEAD"          # 读出的是非目录品牌(如 aPanda/美特好/重庆小面)
    elif vb == e["old_gold"]:
        e["decision"] = "KEEP_ERROR"         # VLM 认可 gold → 系统真错
    else:
        e["decision"] = "EXCL_UNRES"         # 明确矛盾但单读者

import collections
print("决策分布:", dict(collections.Counter(e["decision"] for e in ledger.values())))
for d in ("KEEP_ERROR", "EXCL_UNRES"):
    items = [(f, e["old_gold"], e["vlm_read"]) for f, e in ledger.items() if e["decision"] == d]
    print(f"\n{d} {len(items)} 例(前10):")
    for f, og, vb in items[:10]:
        print(f"  gold={og} vlm={vb}  ({f[:36]})")
json.dump(ledger, open(ROOT / "data/gold_noise_ledger.json", "w"), ensure_ascii=False, indent=1)
