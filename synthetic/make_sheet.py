#!/usr/bin/env python3
"""make_sheet — 联表生成器(色彩保真版)。
修复"油腻色": ①带 ICC 配置的照片先转 sRGB 再贴图(否则 PIL 忽略配置→色相漂移);
②quality=95 + subsampling=0 消色度渗色。"""
import json, sys
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageCms

SRGB = ImageCms.createProfile("sRGB")


def load_rgb(pth):
    im = Image.open(pth)
    icc = im.info.get("icc_profile")
    im = im.convert("RGB")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(BytesIO(icc))
            im = ImageCms.profileToProfile(im, src, SRGB)
        except Exception:
            pass
    return im


def build_sheets(log_path, img_dir, out_dir, title, cols=6, rows=4, qw=320, qh=240):
    ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
    log = json.load(open(ROOT / log_path))
    items = sorted(log.values(), key=lambda v: (v.get("brand_cat") or v.get("brand_query"), v["file"]))
    W, H = qw, qh
    sheets = []
    outp = ROOT / out_dir
    outp.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), cols * rows):
        chunk = items[i:i + cols * rows]
        si = len(sheets)
        im = Image.new("RGB", (W * cols, H * rows + 40), "white")
        d = ImageDraw.Draw(im)
        d.text((8, 8), f"SHEET {si:02d}  {title}  共{len(items)}图", fill="black")
        for j, it in enumerate(chunk):
            r, c = divmod(j, cols)
            x0, y0 = c * W, 40 + r * H
            try:
                p = load_rgb(ROOT / img_dir / it["file"])
                p.thumbnail((W, H - 24))
                im.paste(p, (x0, y0 + 24))
            except Exception:
                pass
            no = it["file"].split("_")[2].replace(".jpg", "")
            d.rectangle([x0, y0, x0 + 66, y0 + 22], fill=(180, 0, 0))
            d.text((x0 + 4, y0 + 3), no, fill="white")
            d.text((x0 + 30, y0 + 3), (it.get("brand_cat") or it.get("brand_query"))[:9], fill="white")
            d.text((x0 + 2, y0 + 24), (it.get("poi_name") or "")[:18], fill="blue")
        fp = outp / f"sheet_{si:02d}.jpg"
        im.save(fp, quality=95, subsampling=0)
        sheets.append({"sheet": fp.name, "n": len(chunk)})
    json.dump(sheets, open(outp / "index.json", "w"), ensure_ascii=False, indent=1)
    print(f"{len(sheets)} 张联表 → {out_dir}")


if __name__ == "__main__":
    build_sheets(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
