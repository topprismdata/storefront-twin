#!/usr/bin/env python3
"""仲裁接触表: 9 张 ownbrand 候选 + 2 张对照(已知正例/已知模板), 640px 大格红底编号。
盲测纪律: 发给 GPT 时只给规则不给真值; 输出仲裁表 arb_ownbrand.jpg + manifest。"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))

# 编号 A01..A11: 9 候选 + C1 已知正例对照 + C2 已知模板对照
CANDS = [
    ("img_0037", "c0037_28d555c7449b.png"), ("img_0140", "c0140_35751f12e7f2.png"),
    ("img_0141", "c0141_f87fa9277c29.jpg"), ("img_0155", "c0155_1fa4c5aa401c.jpg"),
    ("img_0182", "c0182_263504af01c3.jpg"), ("img_0217", "c0217_0c2b4604bbd6.jpg"),
    ("img_0404", "c0404_f987f8a1f804.png"), ("img_0405", "c0405_23abbad8944a.jpg"),
    ("img_0479", "c0479_60b879539ec3.png"),
]
CONTROLS = [("img_0488", None), ("img_0165", None)]  # 已知正例对照 / 已知模板对照


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
    cells = []
    for i, (iid, f) in enumerate(CANDS):
        cells.append((f"A{i+1:02d}", iid, f))
    for j, (iid, _) in enumerate(CONTROLS):
        cells.append((f"C{j+1}", iid, find_file(iid)))
    cols, rows_n = 4, (len(cells) + 3) // 4
    sheet = Image.new("RGB", (CW * cols, CH * rows_n), (70, 74, 82))
    dr = ImageDraw.Draw(sheet)
    manifest = []
    for ci, (badge, iid, f) in enumerate(cells):
        p = ROOT / "data/images" / f
        im = Image.open(p).convert("RGB")
        im.thumbnail((CW, CH - BADGE))
        x0, y0 = (ci % cols) * CW, (ci // cols) * CH
        sheet.paste(im, (x0 + (CW - im.width) // 2, y0 + BADGE + (CH - BADGE - im.height) // 2))
        dr.rectangle([x0, y0, x0 + 138, y0 + BADGE], fill=(196, 30, 30))
        dr.text((x0 + 10, y0 + 8), badge, font=fc(28), fill=(255, 255, 255))
        dr.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1], outline=(40, 42, 46), width=2)
        manifest.append({"badge": badge, "image_id": iid, "file": f})
    out = ROOT / "data/annotation_pack/gen_sheets/arb_ownbrand.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    json.dump(manifest, open(out.with_suffix(".json"), "w"), ensure_ascii=False, indent=1)
    print(f"sheet: {out} ({len(cells)} 格)")
    for m in manifest:
        print(" ", m["badge"], m["image_id"])


if __name__ == "__main__":
    main()
