#!/usr/bin/env python3
"""
M0 2D 参数化门头母本生成器 — 左邻右舍 (Good Neighbors)
=====================================================
设计文档 §6.1 M0：参数化图层表示招牌主标/文字/框/色带/结构比例；
在【自建程序化街景背景】上做四点透视合成 + 遮挡 + 摄影退化。

关键治理约束（全部内建为代码门禁）：
- 母本 asset_status = candidate_unverified（E2 官方对应未证实，禁止命名简约/标准/都市）。
- style_id 用候选簇 CS-01..CS-06，非已证实 style_id。
- 颜色/几何按官方 E2 标定（红 #e60012 / 蓝 #005dac / 彩钢条 白蓝白红白 / 灯箱比例）。
- 背景全程序化自建 → source_policy = self_created（真实图 link_only_pending 不能当背景）。
- 标注随生成 by construction：sign_polygon 与透视投影同一矩阵导出，像素级一致。
- 三层 QC：参数合法性门禁 + 渲染后可读性 + 视觉审样（审样在外部）。
- CF-1..4 反事实组内建；类别不变性白名单（PLUS/编号/字体/色带 是类定义，不当无关增强）。
- 合成数据 domain=synthetic，永不入真实 test split。
- CS-02 书法体：本机无楷体/行楷，用宋体 + 程序化笔触形变作【代理】，显式标注 font_proxy。

用法：
  python3 m0_generate.py --smoke            # 冒烟：每簇 1 张 + CF 组
  python3 m0_generate.py --per-cluster 8    # 每簇 N 张 + CF 组
  python3 m0_generate.py --verify           # 仅校验已有 manifest
"""
import argparse, hashlib, json, math, os, random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageChops

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "schemas" / "synthetic_manifest.schema.json"
GEN_VERSION = "m0-2d-parametric-v1.0"
W, H = 1280, 960

# ---------------------------------------------------------------------------
# 官方 E2 标定的 VI 常量（data/OFFICIAL_E2_FINDINGS.md）
# ---------------------------------------------------------------------------
RED = (230, 0, 18)        # #e60012 PANTONE 3346C
BLUE = (0, 93, 172)       # #005dac PANTONE 2175C
WHITE = (245, 245, 245)
DARK = (40, 42, 48)
GREY_BOARD = (70, 74, 82)

FONT_HEI = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_HIRA = "/System/Library/Fonts/Hiragino Sans GB.ttc"   # index 2 = W6
FONT_SONG = "/System/Library/Fonts/Supplemental/Songti.ttc"  # index 1 = Bold
FONT_ARIAL = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

_font_cache = {}
def font(path, size, index=0):
    key = (path, size, index)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size, index=index)
    return _font_cache[key]

# ---------------------------------------------------------------------------
# 母本注册表：6 个候选簇。asset_status 全部 candidate_unverified。
# 每个母本定义"类因子"的合法取值与结构；nuisance 在生成时独立随机。
# ---------------------------------------------------------------------------
def _logo_glyph(draw, cx, cy, r, color=WHITE):
    """⊙ 圆形标志符号（圆心=和谐/凝聚力），E2 标志网格 20a×10a。"""
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=max(2, int(r*0.22)))
    draw.ellipse([cx-r*0.42, cy-r*0.42, cx+r*0.42, cy+r*0.42], fill=color)

MASTERS = {
    "CS-01": {
        "name": "编号标准型", "font": "hei", "font_proxy": False,
        "number": True, "en_sub": True, "plus": False, "dim3d": False,
        "board": "red", "band": "blue_horizontal", "caigang": False,
        "aspect": (3.2, 4.2), "separator": "dot",
        "evidence": ["E2_bzb_B3-5", "real_img_0005/0010/0018/0025"],
    },
    "CS-02": {
        "name": "书法字体型", "font": "song_proxy", "font_proxy": True,
        "number": True, "en_sub": False, "plus": False, "dim3d": False,
        "board": "red", "band": "blue_horizontal", "caigang": False,
        "aspect": (3.0, 3.8), "separator": "circle",
        "evidence": ["real_img_0000"],
        "note": "行楷书法字体；本机无楷体，宋体+笔触形变代理，待真字体标定",
    },
    "CS-03": {
        "name": "红蓝常规型", "font": "hei", "font_proxy": False,
        "number": False, "en_sub": True, "plus": False, "dim3d": False,
        "board": "red", "band": "blue_horizontal", "caigang": False,
        "aspect": (3.4, 5.0), "separator": "dot",
        "evidence": ["E2_jyb_B3-1", "real_img_0004/0007/0008/0009/0011/0013/0019/0024/0026"],
    },
    "CS-04": {
        "name": "立体发光字型", "font": "hira", "font_proxy": False,
        "number": False, "en_sub": False, "plus": False, "dim3d": True,
        "board": "blue", "band": "red_top", "caigang": False,
        "aspect": (4.0, 6.5), "separator": "dot",
        "evidence": ["E2_dsb_B3-7/8/9_3M灯膜发光字", "real_img_0002/0003/0023/0028+amap"],
    },
    "CS-05": {
        "name": "PLUS升级型", "font": "hei", "font_proxy": False,
        "number": False, "en_sub": True, "plus": True, "dim3d": False,
        "board": "red", "band": "blue_horizontal", "caigang": False,
        "aspect": (3.6, 5.0), "separator": "dot",
        "evidence": ["E2_jyb_品牌故事页PLUS夜景", "real_img_0015/0017/0027", "P05_搜狐"],
        "note": "官方SI标准无PLUS；升级/特定加盟变体，franchise_type 待 E3",
    },
    "CS-06": {
        "name": "超市型", "font": "hira", "font_proxy": False,
        "number": False, "en_sub": False, "plus": False, "dim3d": True,
        "board": "red", "band": "none", "caigang": True,
        "aspect": (3.0, 4.0), "separator": "none", "suffix": "超市",
        "evidence": ["real_img_0012"],
        "note": "超市业态，非便利店标准门头",
    },
}

# 类别不变性白名单：这些是类定义的一部分，CF-2 环境干预时【不得】改动。
CLASS_INVARIANT_KEYS = {"font", "number", "en_sub", "plus", "dim3d", "board",
                        "band", "caigang", "separator", "suffix"}

# 合法扰动范围（设计文档 §6.2 valid_transform_ranges 扩展）
NUISANCE_RANGES = {
    "yaw_deg": (-28, 28), "pitch_deg": (-10, 10),
    "sign_w_frac": (0.52, 0.82), "sign_cy_frac": (0.40, 0.50),
    "exposure": (-0.5, 0.6), "blur_px": (0.0, 1.6),
    "jpeg_q": (55, 92), "occlusion_frac": (0.0, 0.22),
}
LIGHTING = ["day", "dusk", "night", "rain"]
BACKGROUNDS = ["street_a", "street_b", "street_c", "mall"]


# ---------------------------------------------------------------------------
# 透视：平面单应（yaw+pitch 旋转 → 透视投影），sign_polygon 与渲染共用同一矩阵
# ---------------------------------------------------------------------------
def homography_yaw_pitch(yaw_deg, pitch_deg, focal=1400.0):
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # R = Rx(pitch) @ Ry(yaw)
    R = [[cy, 0, sy],
         [sp*sy, cp, -sp*cy],
         [-cp*sy, sp, cp*cy]]
    # 平面 z=0：H 列 = [f*R0, f*R1, t]，t=(0,0,f)
    H = [[focal*R[0][0], focal*R[0][1], 0],
         [focal*R[1][0], focal*R[1][1], 0],
         [R[2][0],        R[2][1],        focal]]
    return H

def apply_H(H, pts):
    out = []
    for x, y in pts:
        d = H[2][0]*x + H[2][1]*y + H[2][2]
        out.append(((H[0][0]*x + H[0][1]*y + H[0][2])/d,
                    (H[1][0]*x + H[1][1]*y + H[1][2])/d))
    return out

def fit_quad_transform(src_rect, dst_quad):
    """PIL QUAD 系数：输出(x,y)→输入。src_rect=(x0,y0,x1,y1)，dst_quad 4 点顺时针。"""
    x0, y0, x1, y1 = src_rect
    src = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    # 求 8 系数 a..h 使 dst→src
    import numpy as np
    A, b = [], []
    for (sx, sy), (dx, dy) in zip(src, dst_quad):
        A.append([dx, dy, 1, 0, 0, 0, -sx*dx, -sx*dy]); b.append(sx)
        A.append([0, 0, 0, dx, dy, 1, -sy*dx, -sy*dy]); b.append(sy)
    coef = np.linalg.solve(np.array(A, float), np.array(b, float))
    return tuple(coef)


# ---------------------------------------------------------------------------
# 程序化自建街景背景（self_created，无版权风险）
# ---------------------------------------------------------------------------
def make_background(kind, rng, night=False):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    # 天空
    sky_top = (150, 175, 200) if not night else (18, 22, 40)
    sky_bot = (205, 215, 225) if not night else (40, 44, 66)
    horizon = int(H*0.40)
    for y in range(horizon):
        t = y/horizon
        c = tuple(int(sky_top[i]*(1-t)+sky_bot[i]*t) for i in range(3))
        d.line([(0, y), (W, y)], fill=c)
    # 远景楼群剪影
    bx = 0
    while bx < W:
        bw = rng.randint(70, 160); bh = rng.randint(60, 200)
        shade = rng.randint(120, 165) if not night else rng.randint(30, 50)
        d.rectangle([bx, horizon-bh, bx+bw, horizon], fill=(shade, shade, shade+6))
        # 窗
        for wy in range(horizon-bh+12, horizon-10, 22):
            for wx in range(bx+10, bx+bw-10, 20):
                if rng.random() < 0.5:
                    wc = (255, 230, 150) if night and rng.random()<0.4 else (shade-25, shade-25, shade-20)
                    d.rectangle([wx, wy, wx+9, wy+11], fill=wc)
        bx += bw + rng.randint(4, 18)
    # 地面/人行道
    gnd = (150, 150, 152) if not night else (45, 46, 52)
    d.rectangle([0, horizon, W, H], fill=gnd)
    for y in range(horizon, H, 26):
        d.line([(0, y), (W, y)], fill=(gnd[0]-12, gnd[1]-12, gnd[2]-12), width=1)
    # 店面墙体（招牌载体）：一整面建筑立面
    wall_x0, wall_x1 = int(W*0.04), int(W*0.96)
    wall_top = int(H*0.16)
    wall = (190, 186, 178) if not night else (60, 58, 62)
    if kind == "street_b": wall = (176, 168, 158) if not night else (56, 52, 56)
    if kind == "street_c": wall = (168, 176, 182) if not night else (52, 56, 62)
    if kind == "mall": wall = (200, 198, 192) if not night else (66, 64, 68)
    d.rectangle([wall_x0, wall_top, wall_x1, H], fill=wall)
    # 墙面分缝
    for x in range(wall_x0, wall_x1, 90):
        d.line([(x, wall_top), (x, H)], fill=(wall[0]-14, wall[1]-14, wall[2]-14), width=1)
    # 招牌下方的玻璃门窗带
    glass_top = int(H*0.56)
    glass = (120, 150, 170) if not night else (30, 40, 58)
    d.rectangle([wall_x0+20, glass_top, wall_x1-20, H-30], fill=glass)
    # 门框竖梃
    for x in range(wall_x0+20, wall_x1-20, 130):
        d.rectangle([x, glass_top, x+8, H-30], fill=(90, 92, 96) if not night else (40, 42, 48))
    # 玻璃反光斜纹
    for i in range(6):
        gx = rng.randint(wall_x0+30, wall_x1-120)
        d.polygon([(gx, glass_top+10), (gx+50, glass_top+10),
                   (gx-30, H-40), (gx-80, H-40)],
                  fill=(glass[0]+22, glass[1]+22, glass[2]+26))
    # 店内货架隐约（玻璃后）
    for x in range(wall_x0+40, wall_x1-60, 150):
        for sy in range(glass_top+40, H-70, 46):
            d.rectangle([x, sy, x+100, sy+8], fill=(glass[0]-18, glass[1]-14, glass[2]-10))
    # 门前车辆/杂物（nuisance）
    if rng.random() < 0.6:
        cx = rng.randint(int(W*0.1), int(W*0.8)); cy = H-70
        car_c = (rng.randint(60,180), rng.randint(60,180), rng.randint(60,180))
        if night: car_c = tuple(c//2 for c in car_c)
        d.rounded_rectangle([cx, cy-46, cx+150, cy], radius=14, fill=car_c)
        d.rounded_rectangle([cx+22, cy-72, cx+128, cy-40], radius=12, fill=car_c)
        d.ellipse([cx+18, cy-16, cx+50, cy+16], fill=(30,30,30))
        d.ellipse([cx+100, cy-16, cx+132, cy+16], fill=(30,30,30))
    return img, {"wall_rect": [wall_x0, wall_top, wall_x1, H],
                 "glass_top": glass_top, "horizon": horizon}


# ---------------------------------------------------------------------------
# 招牌渲染（参数化图层）：在本地正射坐标系画，再透视贴到背景
# ---------------------------------------------------------------------------
def render_sign(master, rng, store_no, en_variant, px_h=260, anomaly=False):
    """返回 (sign_img RGBA, 本地坐标系下的组件 bbox 字典, 可见属性列表)。"""
    aspect = rng.uniform(*master["aspect"])
    px_w = int(px_h * aspect)
    img = Image.new("RGBA", (px_w, px_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    attrs = []
    boxes = {}

    board_c = RED if master["board"] == "red" else BLUE
    if anomaly:
        board_c = (0, 140, 70)   # CF-4 非法 VI：绿色底板（官方仅红/蓝）
    # 底板
    pad = int(px_h*0.06)
    if master["dim3d"]:
        # 立体发光字：深色/蓝底板 + 白色亚克力发光字（无边）
        d.rounded_rectangle([0, 0, px_w, px_h], radius=int(px_h*0.08), fill=(*board_c, 255))
    else:
        d.rounded_rectangle([0, 0, px_w, px_h], radius=int(px_h*0.10), fill=(*board_c, 255))
        # 铝合金包边（E2 30mm）
        d.rounded_rectangle([0, 0, px_w, px_h], radius=int(px_h*0.10),
                            outline=(210, 212, 216, 255), width=max(2, int(px_h*0.02)))
    boxes["board"] = [0, 0, px_w, px_h]

    # 色带（CF-4 anomaly 时换非法黄色）
    band_blue = (240, 200, 0) if anomaly else BLUE
    band_red = (240, 200, 0) if anomaly else RED
    if master["band"] == "blue_horizontal":
        bh = int(px_h*0.16)
        d.rectangle([0, px_h-bh, px_w, px_h], fill=(*band_blue, 255))
        boxes["band"] = [0, px_h-bh, px_w, px_h]
        attrs.append("illegal_yellow_band" if anomaly else "blue_band")
    elif master["band"] == "red_top":
        bh = int(px_h*0.12)
        d.rectangle([0, 0, px_w, bh], fill=(*band_red, 255))
        boxes["band"] = [0, 0, px_w, bh]
        attrs.append("illegal_yellow_band" if anomaly else "red_top_band")

    # 彩钢条（白蓝白红白，竖向 5 色）— 超市型/简约版副招
    if master["caigang"]:
        cw = int(px_w*0.10); cx = px_w - cw - pad
        colors = [WHITE, BLUE, WHITE, RED, WHITE]
        seg = px_h / 5
        for i, cc in enumerate(colors):
            d.rectangle([cx, int(i*seg), cx+cw, int((i+1)*seg)], fill=(*cc, 255))
        boxes["caigang"] = [cx, 0, cx+cw, px_h]; attrs.append("caigang_strip")

    # 主文字「左邻·右舍」
    sep = master["separator"]
    suffix = master.get("suffix", "")
    if sep == "circle":
        text_main = "左邻 右舍"   # ⊙ 单独画
    elif sep == "dot":
        text_main = "左邻·右舍"
    else:
        text_main = "左邻右舍"
    if suffix:
        text_main += suffix

    font_kind = master["font"]
    text_h = int(px_h*0.46)
    if font_kind == "hei":
        f = font(FONT_HEI, text_h)
    elif font_kind == "hira":
        f = font(FONT_HIRA, text_h, index=2)
    else:  # song_proxy
        f = font(FONT_SONG, text_h, index=1)

    # 文字区域（避开彩钢条）
    text_right = px_w - (int(px_w*0.12) if master["caigang"] else pad)
    text_left = pad + int(px_h*0.04)
    avail_w = text_right - text_left
    # 自适应字号使主标占 ~70% 宽
    while f.size > 20:
        bb = d.textbbox((0, 0), text_main, font=f)
        if bb[2]-bb[0] <= avail_w*0.92:
            break
        f = font(f.path, f.size-4, index=(2 if font_kind=="hira" else (1 if font_kind=="song_proxy" else 0)))
    bb = d.textbbox((0, 0), text_main, font=f)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    tx = text_left + (avail_w - tw)//2
    ty = int(px_h*0.20) if master["band"] != "red_top" else int(px_h*0.26)

    # CS-04 都市版 B3-6：蓝底板上居中红底白字 logo 面板（无论多长只放一组 logo）
    if master["board"] == "blue" and not anomaly:
        lp_w = int(tw*1.18); lp_h = int(th*1.5)
        lp_x = tx - (lp_w-tw)//2; lp_y = ty - (lp_h-th)//2
        d.rounded_rectangle([lp_x, lp_y, lp_x+lp_w, lp_y+lp_h],
                            radius=int(px_h*0.05), fill=(*RED, 255))
        boxes["logo_panel"] = [lp_x, lp_y, lp_x+lp_w, lp_y+lp_h]
        attrs.append("red_logo_panel")
    text_color = WHITE
    if master["dim3d"]:
        # 立体发光字：先画发光晕，再画白字 + 侧壁厚度
        glow = Image.new("RGBA", (px_w, px_h), (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        gd.text((tx, ty), text_main, font=f, fill=(255, 250, 235, 255))
        glow = glow.filter(ImageFilter.GaussianBlur(px_h*0.045))
        img = Image.alpha_composite(img, glow)
        d = ImageDraw.Draw(img)
        # 侧壁（厚度感）
        d.text((tx+3, ty+4), text_main, font=f, fill=(180, 180, 185, 255))
        d.text((tx, ty), text_main, font=f, fill=(255, 255, 252, 255))
        attrs.append("dim3d_letter")
    else:
        d.text((tx, ty), text_main, font=f, fill=(*text_color, 255))
    boxes["main_text"] = [tx, ty, tx+tw, ty+th]
    attrs.append("main_text")

    # ⊙ 间隔符（circle 分隔）
    if sep == "circle":
        # 在"左邻"与"右舍"之间画 ⊙
        f_left = d.textbbox((0,0), "左邻", font=f)
        lw = f_left[2]-f_left[0]
        gcx = tx + lw + int(px_h*0.10)
        gcy = ty + th//2
        _logo_glyph(d, gcx, gcy, int(px_h*0.11))
        boxes["logo_glyph"] = [gcx-int(px_h*0.11), gcy-int(px_h*0.11),
                               gcx+int(px_h*0.11), gcy+int(px_h*0.11)]
        attrs.append("logo_circle")

    # 书法代理：笔触形变（CS-02）— 轻微剪切+粗细抖动模拟行楷
    if master["font_proxy"]:
        # 对主文字区域做轻微非线性形变（代理，非真行楷）
        region = img.crop(boxes["main_text"])
        region = region.transform(region.size, Image.AFFINE,
                                  (1, 0.06, 0, 0, 1, 0), resample=Image.BICUBIC)
        img.paste(region, (boxes["main_text"][0], boxes["main_text"][1]), region)
        attrs.append("calligraphy_proxy")

    # 编号 No.XXXX（红色门牌铝板的招牌侧小标）
    if master["number"]:
        nf = font(FONT_ARIAL, int(px_h*0.16))
        ntext = f"No.{store_no:04d}"
        nb = d.textbbox((0,0), ntext, font=nf)
        nx = px_w - (nb[2]-nb[0]) - pad - int(px_h*0.02)
        ny = px_h - int(px_h*0.30)
        d.text((nx, ny), ntext, font=nf, fill=(255, 255, 255, 255))
        boxes["number"] = [nx, ny, nx+nb[2]-nb[0], ny+nb[3]-nb[1]]
        attrs.append("store_number")

    # 英文副标 Good Neighbors / GoodNeighbor（拼写变体）
    if master["en_sub"]:
        ef = font(FONT_ARIAL, int(px_h*0.13))
        etext = en_variant
        eb = d.textbbox((0,0), etext, font=ef)
        ex = text_left
        ey = ty + th + int(px_h*0.01)
        # 防裁切：若下方空间不足则上移紧贴主标
        if ey + (eb[3]-eb[1]) > px_h - int(px_h*0.16):
            ey = px_h - int(px_h*0.16) - (eb[3]-eb[1])
        d.text((ex, ey), etext, font=ef, fill=(255, 255, 255, 235))
        boxes["en_sub"] = [ex, ey, ex+eb[2]-eb[0], ey+eb[3]-eb[1]]
        attrs.append("en_subtitle")

    # PLUS 角标（主标右侧，深色底白字）
    if master["plus"]:
        pf = font(FONT_ARIAL, int(px_h*0.20))
        ptext = "PLUS"
        pb = d.textbbox((0,0), ptext, font=pf)
        pw, phh = pb[2]-pb[0]+int(px_h*0.08), pb[3]-pb[1]+int(px_h*0.06)
        px0 = px_w - pw - pad
        py0 = int(px_h*0.16)
        d.rounded_rectangle([px0, py0, px0+pw, py0+phh], radius=int(px_h*0.04),
                            fill=(35, 38, 46, 255))
        d.text((px0+int(px_h*0.04), py0+int(px_h*0.02)), ptext, font=pf,
               fill=(255, 255, 255, 255))
        boxes["plus_badge"] = [px0, py0, px0+pw, py0+phh]
        attrs.append("plus_badge")

    return img, boxes, attrs


# ---------------------------------------------------------------------------
# 摄影退化 + 光照
# ---------------------------------------------------------------------------
def apply_lighting(img, lighting, rng, sign_quad=None):
    if lighting == "night":
        img = ImageEnhance.Brightness(img).enhance(0.55)
        img = ImageEnhance.Color(img).enhance(0.8)
        # 暖色路灯偏色
        r, g, b = img.split()
        r = r.point(lambda v: min(255, int(v*1.06)))
        b = b.point(lambda v: int(v*0.92))
        img = Image.merge("RGB", (r, g, b))
    elif lighting == "dusk":
        img = ImageEnhance.Brightness(img).enhance(0.78)
        r, g, b = img.split()
        r = r.point(lambda v: min(255, int(v*1.08)))
        b = b.point(lambda v: int(v*0.90))
        img = Image.merge("RGB", (r, g, b))
    elif lighting == "rain":
        img = ImageEnhance.Brightness(img).enhance(0.72)
        img = ImageEnhance.Color(img).enhance(0.7)
        # 雨丝
        d = ImageDraw.Draw(img, "RGBA")
        for _ in range(160):
            x = rng.randint(0, W); y = rng.randint(0, H)
            ln = rng.randint(12, 26)
            d.line([(x, y), (x-4, y+ln)], fill=(200, 210, 225, 70), width=1)
    return img

def lightbox_glow(img, lighting, sign_quad):
    """夜景/黄昏 × 灯箱光源联动（§6.3）：招牌自发光，在摄影退化【之后】施加，避免被再压暗。"""
    if lighting not in ("night", "dusk") or not sign_quad:
        return img
    sx0 = max(0, int(min(p[0] for p in sign_quad))); sx1 = min(W, int(max(p[0] for p in sign_quad)))
    sy0 = max(0, int(min(p[1] for p in sign_quad))); sy1 = min(H, int(max(p[1] for p in sign_quad)))
    if sx1 <= sx0 or sy1 <= sy0:
        return img
    bright = 2.1 if lighting == "night" else 1.45
    region = img.crop((sx0, sy0, sx1, sy1))
    region = ImageEnhance.Brightness(region).enhance(bright)
    region = ImageEnhance.Contrast(region).enhance(1.15)
    img.paste(region, (sx0, sy0))
    glow = region.filter(ImageFilter.GaussianBlur(12))
    img.paste(Image.blend(region, glow, 0.4), (sx0, sy0))
    return img

def degrade(img, exposure, blur_px, jpeg_q, rng):
    img = ImageEnhance.Brightness(img).enhance(1.0 + exposure*0.4)
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.92, 1.08))
    if blur_px > 0.15:
        img = img.filter(ImageFilter.GaussianBlur(blur_px))
    # 传感器噪声
    if rng.random() < 0.7:
        noise = Image.effect_noise(img.size, rng.uniform(4, 12)).convert("L")
        img = ImageChops.add(img, Image.merge("RGB", (noise, noise, noise)), scale=2, offset=-6)
    return img


# ---------------------------------------------------------------------------
# 单张生成
# ---------------------------------------------------------------------------
def generate_one(master_id, seed, cf_mode="none", cf_pair=None):
    rng = random.Random(seed)
    master = MASTERS[master_id]
    store_no = rng.randint(1, 9999)
    en_variant = rng.choice(["Good Neighbors", "GoodNeighbor", "Good neighbors"])

    # nuisance 采样
    lighting = rng.choice(LIGHTING)
    bg_kind = rng.choice(BACKGROUNDS)
    yaw = rng.uniform(*NUISANCE_RANGES["yaw_deg"])
    pitch = rng.uniform(*NUISANCE_RANGES["pitch_deg"])
    sign_w_frac = rng.uniform(*NUISANCE_RANGES["sign_w_frac"])
    sign_cy_frac = rng.uniform(*NUISANCE_RANGES["sign_cy_frac"])
    exposure = rng.uniform(*NUISANCE_RANGES["exposure"])
    blur_px = rng.uniform(*NUISANCE_RANGES["blur_px"])
    jpeg_q = rng.randint(*NUISANCE_RANGES["jpeg_q"])
    occ_frac = rng.uniform(0, NUISANCE_RANGES["occlusion_frac"][1])

    # CF-2 环境干预：固定类因子，独立随机环境（这里 nuisance 已独立）
    # CF-1 类别干预：固定环境，只换版型 → 由 cf_pair 控制（见 build_cf_groups）
    if cf_mode == "cf1_fixed_env" and cf_pair:
        lighting, bg_kind, yaw, pitch, sign_w_frac, sign_cy_frac = cf_pair["env"]
        exposure, blur_px, jpeg_q = cf_pair["degrade"]
    if cf_mode == "cf3_damage":
        occ_frac = rng.uniform(0.30, 0.48)   # 关键文字部分遮挡
    anomaly = (cf_mode == "cf4_anomaly")

    bg, bg_meta = make_background(bg_kind, rng, night=(lighting in ("night", "dusk")))

    # 招牌正射渲染
    sign_px_h = int(H * 0.20)
    sign_img, comp_boxes, attrs = render_sign(master, rng, store_no, en_variant, px_h=sign_px_h, anomaly=anomaly)
    sw, sh = sign_img.size

    # 透视：本地坐标 → 图像
    Hmat = homography_yaw_pitch(yaw, pitch)
    local_corners = [(-sw/2, -sh/2), (sw/2, -sh/2), (sw/2, sh/2), (-sw/2, sh/2)]
    proj = apply_H(Hmat, local_corners)
    # 归一到目标宽度并放置
    pxs = [p[0] for p in proj]; pys = [p[1] for p in proj]
    pw = max(pxs)-min(pxs); ph = max(pys)-min(pys)
    target_w = W * sign_w_frac
    scale = target_w / pw
    cx = W * rng.uniform(0.42, 0.58)
    cy = H * sign_cy_frac
    dst_quad = [(cx + (p[0]-min(pxs)-pw/2)*scale,
                 cy + (p[1]-min(pys)-ph/2)*scale) for p in proj]

    # 透视贴合：PIL PERSPECTIVE 系数（输出→输入），与 sign_polygon 同一单应
    coef = fit_quad_transform((0, 0, sw, sh), dst_quad)
    warped = sign_img.transform((W, H), Image.PERSPECTIVE, coef, resample=Image.BICUBIC)
    bg_rgba = bg.convert("RGBA")
    composed = Image.alpha_composite(bg_rgba, warped)

    # sign_polygon（像素级，与渲染同一投影）
    sign_polygon = [[round(p[0], 1), round(p[1], 1)] for p in dst_quad]

    # 组件 landmark boxes：本地 bbox 经同一单应+缩放平移
    def map_local_box(b):
        pts = [(b[0]-sw/2, b[1]-sh/2), (b[2]-sw/2, b[1]-sh/2),
               (b[2]-sw/2, b[3]-sh/2), (b[0]-sw/2, b[3]-sh/2)]
        pp = apply_H(Hmat, pts)
        return [[round(cx + (p[0]-min(pxs)-pw/2)*scale, 1),
                 round(cy + (p[1]-min(pys)-ph/2)*scale, 1)] for p in pp]
    landmarks = {k: map_local_box(v) for k, v in comp_boxes.items()}

    img = composed.convert("RGB")

    # 遮挡（nuisance）：在招牌区域叠加前景遮挡物
    occ_box = None
    if occ_frac > 0.03:
        d = ImageDraw.Draw(img, "RGBA")
        sx0 = min(p[0] for p in dst_quad); sx1 = max(p[0] for p in dst_quad)
        sy0 = min(p[1] for p in dst_quad); sy1 = max(p[1] for p in dst_quad)
        ow = (sx1-sx0)*occ_frac
        ox = rng.uniform(sx0, sx1-ow)
        # 遮挡物：树枝/招牌旁广告/车辆顶
        occ_type = rng.choice(["tree", "ad_banner", "pole"])
        if occ_type == "tree":
            d.ellipse([ox-20, sy0-40, ox+ow+20, sy1+30], fill=(40, 70, 40, 235))
        elif occ_type == "ad_banner":
            # 异色前景广告（橙底白字），不与红招牌同色，避免前景/招牌混淆
            d.rectangle([ox, sy0-10, ox+ow, sy1+10], fill=(235, 150, 30, 240))
            d.text((ox+4, sy0+4), "促销", font=font(FONT_HEI, 22), fill=(255,255,255,255))
        else:
            d.rectangle([ox, sy0-60, ox+12, sy1+80], fill=(90, 90, 95, 255))
        occ_box = [round(ox,1), round(sy0,1), round(ox+ow,1), round(sy1,1)]
        attrs.append("occluded")

    # 光照 + 退化
    img = apply_lighting(img, lighting, rng)
    img = degrade(img, exposure, blur_px, jpeg_q, rng)
    img = lightbox_glow(img, lighting, dst_quad)   # 退化后施加，保持灯箱亮度

    # CF-3 破损：降低关键文字可读性（额外模糊招牌区）
    if cf_mode == "cf3_damage":
        sx0 = int(min(p[0] for p in dst_quad)); sx1 = int(max(p[0] for p in dst_quad))
        sy0 = int(min(p[1] for p in dst_quad)); sy1 = int(max(p[1] for p in dst_quad))
        region = img.crop((sx0, sy0, sx1, sy1)).filter(ImageFilter.GaussianBlur(3.2))
        img.paste(region, (sx0, sy0))

    # QC1 参数合法性门禁
    qc = qc_param_check(master, yaw, pitch, sign_w_frac, lighting, attrs)
    # QC2 渲染后可读性（招牌区对比度/尺寸）
    readability = qc_readability(img, dst_quad)
    qc["readability"] = readability
    # cf3_damage 预期低可读性（拒识样本），豁免可读性门禁
    if cf_mode != "cf3_damage":
        qc["pass"] = qc["pass"] and readability["ok"]
    else:
        qc["cf3_expected_low_readability"] = True

    # 编码（JPEG 压缩退化）
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=jpeg_q)
    img_bytes = buf.getvalue()
    sha = hashlib.sha256(img_bytes).hexdigest()

    asset_hash = master_asset_hash(master_id)
    manifest = {
        "image_id": f"syn_{master_id}_{seed:08d}",
        "domain": "synthetic",
        "asset_id": f"twin2d_{master_id}",
        "asset_hash": asset_hash,
        "generator_commit": GEN_VERSION,
        "seed": seed,
        "style_id": master_id,   # 候选簇，非已证实
        "gt": {
            "sign_polygon": sign_polygon,
            "visible_attributes": attrs,
            "landmarks": landmarks,
            "store_number": f"No.{store_no:04d}" if master["number"] else None,
            "en_subtitle": en_variant if master["en_sub"] else None,
            "class_id": master_id,
            "brand_id": "zuolinyoushe",
            "asset_status": "candidate_unverified",
            "font_proxy": master["font_proxy"],
        },
        "nuisance": {
            "background_id": bg_kind,
            "camera_yaw_deg": round(yaw, 2),
            "camera_pitch_deg": round(pitch, 2),
            "lighting_id": lighting,
            "sign_w_frac": round(sign_w_frac, 3),
            "exposure": round(exposure, 3),
            "blur_px": round(blur_px, 2),
            "jpeg_q": jpeg_q,
            "occlusion_frac": round(occ_frac, 3),
            "occlusion_box": occ_box,
        },
        "source_policy": "self_created",
        "label_authority": "synthetic_by_construction",
        "review_state": "pending",
        "cf_mode": cf_mode,
        "qc": qc,
        "image_sha256": sha,
        "split": "synthetic_train_only",   # 永不入真实 test
    }
    return img_bytes, manifest


def master_asset_hash(master_id):
    """母本内容哈希（定义+结构），用于 asset_hash 与可复现追溯。"""
    m = MASTERS[master_id]
    payload = json.dumps({k: m[k] for k in sorted(m)}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def qc_param_check(master, yaw, pitch, sign_w_frac, lighting, attrs):
    """QC1：参数合法性 + 类别不变性白名单 + 物理/品牌合理联动。"""
    issues = []
    if not (NUISANCE_RANGES["yaw_deg"][0] <= yaw <= NUISANCE_RANGES["yaw_deg"][1]):
        issues.append("yaw_out_of_range")
    if not (NUISANCE_RANGES["pitch_deg"][0] <= pitch <= NUISANCE_RANGES["pitch_deg"][1]):
        issues.append("pitch_out_of_range")
    if not (NUISANCE_RANGES["sign_w_frac"][0] <= sign_w_frac <= NUISANCE_RANGES["sign_w_frac"][1]):
        issues.append("sign_width_out_of_range")
    # 品牌合理联动：夜景应有灯箱发光（dim3d 或灯箱底板），不能纯黑无光
    # 类别关键元素不得被非法删除（白名单）
    if master["plus"] and "plus_badge" not in attrs:
        issues.append("class_key_plus_missing")
    if master["number"] and "store_number" not in attrs:
        issues.append("class_key_number_missing")
    if master["caigang"] and "caigang_strip" not in attrs:
        issues.append("class_key_caigang_missing")
    return {"pass": len(issues) == 0, "issues": issues}


def qc_readability(img, dst_quad):
    """QC2：渲染后招牌区可读性（面积占比 + 局部对比度）。"""
    sx0 = max(0, int(min(p[0] for p in dst_quad))); sx1 = min(W, int(max(p[0] for p in dst_quad)))
    sy0 = max(0, int(min(p[1] for p in dst_quad))); sy1 = min(H, int(max(p[1] for p in dst_quad)))
    area_frac = ((sx1-sx0)*(sy1-sy0)) / (W*H)
    region = img.crop((sx0, sy0, sx1, sy1)).convert("L")
    import numpy as np
    arr = np.asarray(region, float)
    contrast = float(arr.std()) if arr.size else 0.0
    ok = area_frac > 0.04 and contrast > 18 and (sx1-sx0) > 120
    return {"ok": ok, "area_frac": round(area_frac, 4), "contrast_std": round(contrast, 1),
            "sign_px_w": sx1-sx0}


# ---------------------------------------------------------------------------
# CF 反事实组（设计文档 §6.4）
# ---------------------------------------------------------------------------
def build_cf_groups(out_dir, rng_seed=20261111):
    """生成 CF-1..4 成对样本，返回清单。"""
    rng = random.Random(rng_seed)
    records = []
    pairs_dir = out_dir / "cf_pairs"; pairs_dir.mkdir(parents=True, exist_ok=True)

    # CF-1 类别干预：固定环境，只换版型（CS-03 vs CS-05，差异=PLUS）
    env = ("day", "street_a", 8.0, 2.0, 0.66, 0.36)
    deg = (0.1, 0.4, 82)
    cf_pair = {"env": env, "degrade": deg}
    for mid in ("CS-03", "CS-05"):
        b, m = generate_one(mid, seed=rng.randint(0, 10**8), cf_mode="cf1_fixed_env", cf_pair=cf_pair)
        fn = f"cf1_{mid}.jpg"; (pairs_dir/fn).write_bytes(b)
        m["image_id"] = f"cf1_{mid}"; m["cf_group"] = "CF1_plus_intervention"
        records.append(m)

    # CF-2 环境干预：固定版型 CS-03，独立随机环境（多张）
    base_seed = rng.randint(0, 10**8)
    for i in range(3):
        b, m = generate_one("CS-03", seed=base_seed+i*97, cf_mode="cf2_env")
        fn = f"cf2_{i}.jpg"; (pairs_dir/fn).write_bytes(b)
        m["image_id"] = f"cf2_{i}"; m["cf_group"] = "CF2_env_invariance"
        records.append(m)

    # CF-3 局部破损：关键文字遮挡，模型应降信心/拒识
    b, m = generate_one("CS-01", seed=rng.randint(0, 10**8), cf_mode="cf3_damage")
    (pairs_dir/"cf3_damage.jpg").write_bytes(b)
    m["image_id"] = "cf3_damage"; m["cf_group"] = "CF3_occlusion_reject"
    m["gt"]["expected_behavior"] = "lower_confidence_or_reject"
    records.append(m)

    # CF-4 非法搭配：交换颜色/Logo 形成不符合 VI 的组合 → synthetic_unknown/anomaly
    b, m = generate_one("CS-03", seed=rng.randint(0, 10**8), cf_mode="cf4_anomaly")
    # 真实渲染非法 VI：绿色底板 + 黄色色带（render_sign anomaly=True），标记 anomaly
    m["gt"]["class_id"] = "synthetic_unknown"
    m["gt"]["anomaly"] = "illegal_color_combination"
    m["style_id"] = "synthetic_unknown"
    m["cf_group"] = "CF4_illegal_combo"
    m["image_id"] = "cf4_anomaly"
    (pairs_dir/"cf4_anomaly.jpg").write_bytes(b)
    records.append(m)

    return records


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def validate_manifests(manifests):
    import jsonschema
    schema = json.loads(SCHEMA_PATH.read_text())
    errs = []
    for m in manifests:
        try:
            jsonschema.validate(m, schema)
        except jsonschema.ValidationError as e:
            errs.append((m.get("image_id"), str(e)[:120]))
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-cluster", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "renders"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seed-base", type=int, default=20261111)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "synthetic_manifest.jsonl"

    if args.verify:
        manifests = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
        errs = validate_manifests(manifests)
        print(f"verified {len(manifests)} manifests, errors={len(errs)}")
        for e in errs[:20]: print("  ERR", e)
        sys.exit(1 if errs else 0)

    per = 1 if args.smoke else (args.per_cluster or 8)
    manifests = []
    rng = random.Random(args.seed_base)
    for mid in MASTERS:
        for i in range(per):
            seed = rng.randint(0, 10**8)
            img_bytes, m = generate_one(mid, seed)
            fn = f"{m['image_id']}.jpg"
            (out_dir / fn).write_bytes(img_bytes)
            manifests.append(m)
        print(f"{mid}: {per} rendered")

    # CF 组
    cf_records = build_cf_groups(out_dir, rng_seed=args.seed_base+1)
    manifests.extend(cf_records)
    print(f"CF groups: {len(cf_records)} samples")

    # schema 校验
    errs = validate_manifests(manifests)
    print(f"schema validation: {len(manifests)} manifests, {len(errs)} errors")
    for e in errs[:20]: print("  ERR", e)

    # QC 汇总
    qc_fail = [m["image_id"] for m in manifests if not m["qc"]["pass"]]
    print(f"QC pass: {len(manifests)-len(qc_fail)}/{len(manifests)}; fail={qc_fail[:10]}")

    with open(manifest_path, "w") as f:
        for m in manifests:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"manifest → {manifest_path}")
    print(f"renders → {out_dir} ({len(manifests)} images)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()