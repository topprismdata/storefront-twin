#!/usr/bin/env python3
"""重建 render_stock eval 接触表(2026-11-13)。
归档会话生成的 eval_00-10.jpg 全是灰占位(图片加载失败未断言)。
本脚本从 stock_benchmark.json 的 path 直读真实图片,加载失败即报错。
布局: 每张 20 格(4×5), 徽章 S+mid 数字补零5位(与原版一致), cell 640×512。
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ImageFile = __import__("PIL.ImageFile", fromlist=["ImageFile"])
ImageFile.LOAD_TRUNCATED_IMAGES = True  # bing 下载偶有尾部截断,容忍渲染已解码部分

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/annotation_pack/stock_sheets"

def main():
    b = json.load(open(ROOT / "data/stock_benchmark.json"))
    rows = b["rows"]
    assert len(rows) == 220, len(rows)
    F = "/System/Library/Fonts/STHeiti Medium.ttc"
    fc = lambda s: ImageFont.truetype(F, s)
    CW, CH, BADGE = 640, 512, 44
    n_load_fail = 0
    for si in range(11):
        chunk = rows[si * 20:(si + 1) * 20]
        sheet = Image.new("RGB", (CW * 4, CH * 5), (70, 74, 82))
        dr = ImageDraw.Draw(sheet)
        for ci, r in enumerate(chunk):
            mid = r["mid"]
            badge = "S" + mid[1:].zfill(5)
            p = ROOT / r["path"]
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((CW, CH - BADGE))
            except Exception as e:
                n_load_fail += 1
                raise RuntimeError(f"LOAD FAIL {mid}: {p} ({e})")
            x0, y0 = (ci % 4) * CW, (ci // 4) * CH
            sheet.paste(im, (x0 + (CW - im.width) // 2, y0 + BADGE + (CH - BADGE - im.height) // 2))
            dr.rectangle([x0, y0, x0 + 138, y0 + BADGE], fill=(196, 30, 30))
            dr.text((x0 + 10, y0 + 8), badge, font=fc(28), fill=(255, 255, 255))
            dr.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1], outline=(40, 42, 46), width=2)
        sheet.save(OUT / f"eval_{si:02d}.jpg", quality=85)
        print(f"eval_{si:02d}.jpg rebuilt ({len(chunk)} cells, mids {chunk[0]['mid']}..{chunk[-1]['mid']})")
    print(f"done, load_failures={n_load_fail}")

if __name__ == "__main__":
    main()
