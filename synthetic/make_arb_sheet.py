#!/usr/bin/env python3
"""仲裁接触表: 9 张 ownbrand 候选 + 2 张对照(已知正例/已知模板), 640px 大格红底编号。
盲测纪律: 发给 GPT 时只给规则不给真值; 输出仲裁表 arb_ownbrand.jpg + manifest。"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))

# 编号 A01..A12: 12 张 v4 POI 候选 + C1 已知正例对照 + C2 已知模板对照
CANDS = [
    ("v4_0776_13dd0f7aba53.jpg", "images_v4", "嘉兴 左邻右舍"),
    ("v4_0629_847080f393bf.jpg", "images_v4", "宁波 方前村店"),
    ("v4_0468_1fac922f2b27.jpg", "images_v4", "宁波 德居公寓店"),
    ("v4_0487_aac9ad30cc87.jpg", "images_v4", "宁波 大黄桥南路店"),
    ("v4_0656_de99c2f43193.jpg", "images_v4", "宁波 芳华景苑店"),
    ("v4_0542_28cc02c52c4f.jpg", "images_v4", "宁波 滨海锦苑店"),
    ("v4_0154_8e2cc1ac672d.jpg", "images_v4", "杭州 东湖北路店"),
    ("v4_0318_95cc8a2addb2.jpg", "images_v4", "杭州 临空SOHO店"),
    ("v4_0774_6d49d2de929d.jpg", "images_v4", "嘉兴 紫金路店"),
    ("v4_0796_3e52a6ed9541.jpg", "images_v4", "嘉兴 星河CocoCity店"),
    ("v4_1369_1ca80ada4dca.jpg", "images_v4", "绍兴 京新花苑店"),
    ("v4_0587_230cc4d1963e.jpg", "images_v4", "宁波 左邻右舍便利店"),
]
CONTROLS = [("img_0037", "images"), ("img_0165", "images")]  # 已知正例对照 / 已知模板对照


def find_file(image_id):
    import re, os
    n = int(re.match(r"img_(\d+)", image_id).group(1))
    for f in os.listdir(ROOT / "data/images"):
        m = re.match(r"c?(\d{3,4})_", f)
        if m and int(m.group(1)) == n:
            return f
    return None


def main():
    reg = {r["image_id"]: r for r in json.load(open(ROOT / "data/registry_v3.json"))}
    print("对照1 img_0488:", reg.get("img_0488", {}).get("label"), reg.get("img_0488", {}).get("neg_type"))
    print("对照2 img_0165:", reg.get("img_0165", {}).get("label"), reg.get("img_0165", {}).get("neg_type"))
    F = "/System/Library/Fonts/STHeiti Medium.ttc"
    fc = lambda s: ImageFont.truetype(F, s)
    CW, CH, BADGE = 640, 512, 44
    cells = [(f"A{i+1:02d}", f, d) for i, (f, d, _note) in enumerate(CANDS)]
    for j, (iid, d) in enumerate(CONTROLS):
        cells.append((f"C{j+1}", find_file(iid), d))
    cols, rows_n = 4, (len(cells) + 3) // 4
    sheet = Image.new("RGB", (CW * cols, CH * rows_n), (70, 74, 82))
    dr = ImageDraw.Draw(sheet)
    manifest = []
    for ci, (badge, f, d) in enumerate(cells):
        p = ROOT / "data" / d / f
        im = Image.open(p).convert("RGB")
        im.thumbnail((CW, CH - BADGE))
        x0, y0 = (ci % cols) * CW, (ci // cols) * CH
        sheet.paste(im, (x0 + (CW - im.width) // 2, y0 + BADGE + (CH - BADGE - im.height) // 2))
        dr.rectangle([x0, y0, x0 + 138, y0 + BADGE], fill=(196, 30, 30))
        dr.text((x0 + 10, y0 + 8), badge, font=fc(28), fill=(255, 255, 255))
        dr.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1], outline=(40, 42, 46), width=2)
        manifest.append({"badge": badge, "file": f, "dir": d})
    out = ROOT / "data/annotation_pack/gen_sheets/arb_ownbrand_v5.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    json.dump(manifest, open(out.with_suffix(".json"), "w"), ensure_ascii=False, indent=1)
    print(f"sheet: {out} ({len(cells)} 格)")
    for m in manifest:
        print(" ", m["badge"], m["file"], m["dir"])


if __name__ == "__main__":
    main()
