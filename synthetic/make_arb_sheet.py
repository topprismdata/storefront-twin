#!/usr/bin/env python3
"""仲裁接触表: 9 张 ownbrand 候选 + 2 张对照(已知正例/已知模板), 640px 大格红底编号。
盲测纪律: 发给 GPT 时只给规则不给真值; 输出仲裁表 arb_ownbrand.jpg + manifest。"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))

# 编号 A01..A18: 18 张 v5 兄弟照候选 + C1 已知正例对照 + C2 已知模板对照
CANDS = [
    ("v4_1263_600effe0fcfa.jpg", "images_v4", "绍兴 望湖花园店"),
    ("v4_0478_92bf4e85c1eb.jpg", "images_v4", "宁波 石河路店"),
    ("v4_1225_06453e922d4c.jpg", "images_v4", "绍兴 柯笛花园店"),
    ("v4_1186_542d8f8b5d0f.jpg", "images_v4", "绍兴 永福路店"),
    ("v4_0929_d54844ec888f.jpg", "images_v4", "湖州 风雅阁店"),
    ("v4_0538_2e96656dc2b1.jpg", "images_v4", "宁波 孙塘南路店A"),
    ("v4_0788_e23f8f54286a.jpg", "images_v4", "嘉兴 濮院毛衫店"),
    ("v4_0537_518e2fe71aa3.jpg", "images_v4", "宁波 孙塘南路店B"),
    ("v4_1304_5304de0a1568.jpg", "images_v4", "绍兴 文象府店"),
    ("v4_1339_6fec0b36be1d.jpg", "images_v4", "绍兴 翡翠公馆店"),
    ("v4_1283_6b8f5cf0fc20.jpg", "images_v4", "绍兴 元垄大都会店"),
    ("v4_1699_7f74af7ebd33.jpg", "images_v4", "金华 乐乐烟酒超市?"),
    ("v4_0926_52d96b90fe14.jpg", "images_v4", "湖州 诚宜路"),
    ("v4_2179_b51754e5ae4a.jpg", "images_v4", "衢州 建乙巷店"),
    ("v4_1380_6df7beb2959e.jpg", "images_v4", "绍兴 江南名苑店"),
    ("v4_0942_9bb9ba27a930.jpg", "images_v4", "湖州 中国丝绸小镇店"),
    ("v4_1240_136a292a9d27.jpg", "images_v4", "绍兴 群商路店"),
    ("v4_0828_0ca10cecf348.jpg", "images_v4", "嘉兴 荷园店"),
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
    out = ROOT / "data/annotation_pack/gen_sheets/arb_ownbrand_v6.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    json.dump(manifest, open(out.with_suffix(".json"), "w"), ensure_ascii=False, indent=1)
    print(f"sheet: {out} ({len(cells)} 格)")
    for m in manifest:
        print(" ", m["badge"], m["file"], m["dir"])


if __name__ == "__main__":
    main()
