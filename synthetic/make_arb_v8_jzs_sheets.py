#!/usr/bin/env python3
"""v7 候选 67 张 → 3 张 4×6 接触表 (每张含 C1 正例 + C2 模板对照)。输出 gen_sheets/arb_v8_jzs_s{1,2,3}.jpg"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
OUT = ROOT / "data/annotation_pack/gen_sheets"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
scored = json.load(open(ROOT / "synthetic/vis_cache/v8_jzs_candidate_scores.json"))
HITS = [o for o in scored["ranked"] if o["pred"]]
assert len(HITS) == 58, len(HITS)


def main():
    fc = lambda s: ImageFont.truetype(FONT, s)
    CW, CH, BADGE = 640, 520, 44
    log = json.load(open(ROOT / "data/amap_v5_jzs_download_log.json"))
    byfile = {x["file"]: x for x in log.values() if x.get("file")}
    controls = [(f"C1", "images", "c0037_28d555c7449b.png"), ("C2", "images", "c0165_52347b971b14.jpg")]
    sheets = [HITS[i * 22:(i + 1) * 22] for i in range(3)]
    while sum(len(s) for s in sheets) < len(HITS):
        sheets[-1].append(HITS[-1])
    manifest_all = []
    for si, chunk in enumerate(sheets, 1):
        cells = [(f"A{i+1:02d}", "images_v4", o["file"], o) for i, o in enumerate(chunk)] + [(c[0], c[1], c[2], None) for c in controls]
        cols, rows_n = 4, (len(cells) + 3) // 4
        sheet = Image.new("RGB", (CW * cols, CH * rows_n), (70, 74, 82))
        dr = ImageDraw.Draw(sheet)
        manifest = []
        for ci, (badge, d, f, o) in enumerate(cells):
            p = ROOT / "data" / d / f
            im = Image.open(p).convert("RGB")
            im.thumbnail((CW, CH - BADGE))
            x0, y0 = (ci % cols) * CW, (ci // cols) * CH
            sheet.paste(im, (x0 + (CW - im.width) // 2, y0 + BADGE + (CH - BADGE - im.height) // 2))
            dr.rectangle([x0, y0, x0 + 118, y0 + BADGE], fill=(200, 16, 18))
            dr.text((x0 + 10, y0 + 8), badge, font=fc(28), fill=(255, 255, 255))
            dr.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1], outline=(40, 42, 46), width=2)
            m = {"badge": badge, "file": f, "dir": d}
            if o:
                m.update({"score": o["score"], "poi_name": o["poi_name"], "city": o["city"]})
            manifest.append(m)
        fp = OUT / f"arb_v8_jzs_s{si}.jpg"
        sheet.save(fp, quality=88)
        json.dump(manifest, open(fp.with_suffix(".json"), "w"), ensure_ascii=False, indent=1)
        manifest_all.append({"sheet": str(fp), "n": len(cells)})
        print(f"{fp} ({len(cells)} 格)")
    json.dump(manifest_all, open(ROOT / "synthetic/vis_cache/arb_v8_jzs_manifest.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
