#!/usr/bin/env python3
"""registry v4: 2026-11-13 素材池分类战役产物。
1) 9 张 render_stock 误标翻正为 positive (双通道仲裁: 本地全分辨率 + GPT盲裁 conv 6af686b2, 9/9 A)
2) img_0488 本地文件=昵图网防盗链占位图 → is_representative=False 剔除评测
3) 3 张 rep=False 翻正图升级 rep=True 入评测
v4 eval = 338 行 (38 pos / 300 neg)。registry_v3 不动。"""
import json, shutil
from pathlib import Path

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
ARB_NOTE = ("2026-11-13 双通道仲裁翻正: 本地全分辨率目检(真实门店+品牌VI) + GPT盲裁"
            "(data/annotation_pack/gen_sheets/arb_ownbrand.jpg, conv 6af686b2-d1d0-83ea-a822-6f1e9080f687, 裁A); "
            "原 render_stock 标签系图像搜索素材波误标, 实为品牌宣传/新闻实拍图")

# (image_id, 新 store_group, 备注)
FLIPS = [
    ("img_0037", "tail_no0785",  "门店编号No.0785, 日间街拍"),
    ("img_0140", "tail_plus_night_a", "夜间Plus店, ai-znsj源"),
    ("img_0141", "tail_news_open", "潮新闻开业排队照"),
    ("img_0155", "tail_plus_night_b", "夜间Plus店, 知乎专栏源"),
    ("img_0182", "tail_plus_night_c", "夜间Plus店, 雪球源"),
    ("img_0217", "tail_video_frame", "宣传片转场帧, logosc源"),
    ("img_0404", "tail_no3047",  "门店编号No.3047, 白底红字变体"),
    ("img_0405", "tail_jishun",  "吉顺三村转角店"),
    ("img_0479", "tail_tuxi",    "兔喜生活共址店"),
]

reg = json.load(open(ROOT / "data/registry_v3.json"))
byid = {r["image_id"]: r for r in reg}
flipped = set()
for iid, grp, note in FLIPS:
    r = byid[iid]
    assert r["label"] == "negative" and r.get("neg_type") == "render_stock", iid
    r["label"] = "positive"
    r["neg_type"] = ""
    r["store_id"] = grp
    r["brand_state"] = "verified_own"
    r["fullres_note"] = (r.get("fullres_note") or "") + " | " + ARB_NOTE + " | " + note
    if not r.get("is_representative"):
        r["is_representative"] = True
    flipped.add(iid)

# img_0488: 占位图剔除
r = byid["img_0488"]
r["is_representative"] = False
r["fullres_note"] = (r.get("fullres_note") or "") + (" | 2026-11-13 数据缺陷: 本地文件 c0488_132f74f6c467.jpg 为昵图网防盗链占位图"
    "(400x400, ahash全库扫描仅此1张), 非门店照片; v3 中其 E8G 分数0.64系占位图嵌入产生, 不可信; 剔除评测, 待重下真图")

rep_pos = sum(1 for r in reg if r["label"] == "positive" and r.get("is_representative"))
rep_neg = sum(1 for r in reg if r["label"] == "negative" and r.get("is_representative"))
json.dump(reg, open(ROOT / "data/registry_v4.json", "w"), ensure_ascii=False, indent=1)
json.dump({"date": "2026-11-13", "flips": [f[0] for f in FLIPS], "drop_placeholder": "img_0488",
           "arb_conv": "6af686b2-d1d0-83ea-a822-6f1e9080f687", "arb_result": "9/9 A",
           "v4_eval": {"pos": rep_pos, "neg": rep_neg, "total": rep_pos + rep_neg}},
          open(ROOT / "data/registry_v4_changes.json", "w"), ensure_ascii=False, indent=1)
print(f"registry_v4.json: eval {rep_pos} pos / {rep_neg} neg / {rep_pos+rep_neg} total")
