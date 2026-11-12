#!/usr/bin/env python3
"""
M1-3D 参数化数字孪生母本 — Blender 渲染（左邻右舍门头）
=========================================================
设计文档 §6 M1/M2：招牌厚度、灯箱侧壁、门框、玻璃反射、夜景照明、遮挡层。
类因子与 M0 (m0_generate.py MASTERS) 完全对齐；VI 常量来自 data/OFFICIAL_E2_FINDINGS.md。
用户指令 2026-11-11：以 3D 孪生合成数据做识别训练（真实图权利未审，仅合成可训练）。

治理（硬约束）：
- style_id = 候选簇 CS-01..06，asset_status=candidate_unverified（E2 对应未证实）。
- 全部几何/材质/字体程序化自建 → source_policy=self_created, split=synthetic_train_only。
- 不使用任何真实照片作背景/贴图（权利 pending_review）。
- franchise_type=null（E3 待品牌确认）。CS-02 宋体代理 font_proxy=True。
- 负例含：异品牌门头、无招牌立面、违规 VI（真品牌名+绿板黄带 → brand=other/anomaly）。

确定性：每渲染一个 seed，场景从零重建；同 seed → 同场景参数。
用法（Blender 内运行）：
  blender -b -P m1_3d_twin.py -- --smoke
  blender -b -P m1_3d_twin.py -- --per-cluster 60 --neg 160 --out renders_3d
"""
import bpy, json, math, os, random, sys
from mathutils import Vector, Matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(ROOT, "renders_3d")
FONT_DIR = os.path.join(ROOT, "fonts3d")
GEN_VERSION = "m1-3d-blender-v1.0"
W, H = 960, 720

RED = (230/255, 0/255, 18/255)
BLUE = (0/255, 93/255, 172/255)
WHITE = (0.93, 0.93, 0.93)

FONTS = {
    "hei": os.path.join(FONT_DIR, "hei.ttf"),
    "hira": os.path.join(FONT_DIR, "hira_w6.ttf"),
    "song_proxy": os.path.join(FONT_DIR, "song_bold.ttf"),
    "arial": os.path.join(FONT_DIR, "arial_bold.ttf"),
}

# 与 M0 MASTERS 对齐的类因子（candidate_unverified）
MASTERS = {
    "CS-01": dict(font="hei", number=True, en_sub=True, plus=False, dim3d=False,
                  board="red", band="blue_horizontal", caigang=False,
                  aspect=(3.2, 4.2), separator="dot", font_proxy=False),
    "CS-02": dict(font="song_proxy", number=True, en_sub=False, plus=False, dim3d=False,
                  board="red", band="blue_horizontal", caigang=False,
                  aspect=(3.0, 3.8), separator="circle", font_proxy=True),
    "CS-03": dict(font="hei", number=False, en_sub=True, plus=False, dim3d=False,
                  board="red", band="blue_horizontal", caigang=False,
                  aspect=(3.4, 5.0), separator="dot", font_proxy=False),
    "CS-04": dict(font="hira", number=False, en_sub=False, plus=False, dim3d=True,
                  board="blue", band="red_top", caigang=False,
                  aspect=(4.0, 6.5), separator="dot", font_proxy=False),
    "CS-05": dict(font="hei", number=False, en_sub=True, plus=True, dim3d=False,
                  board="red", band="blue_horizontal", caigang=False,
                  aspect=(3.6, 5.0), separator="dot", font_proxy=False),
    "CS-06": dict(font="hira", number=False, en_sub=False, plus=False, dim3d=True,
                  board="red", band="none", caigang=True,
                  aspect=(3.0, 4.0), separator="none", suffix="超市", font_proxy=False),
}

OTHER_BRANDS = [
    ("美宜佳", (0.0, 0.55, 0.25), WHITE), ("天福便利", (0.9, 0.6, 0.0), (0.1, 0.1, 0.1)),
    ("芙蓉兴盛", (0.85, 0.1, 0.1), WHITE), ("Today今天", (0.0, 0.45, 0.35), WHITE),
    ("见福便利", (0.95, 0.95, 0.95), (0.7, 0.1, 0.1)), ("红旗连锁", (0.8, 0.05, 0.1), WHITE),
    ("唐久便利", (0.0, 0.35, 0.65), WHITE), ("十足便利", (0.95, 0.45, 0.0), WHITE),
    # 错误归因新增：真实 FP 中的同色系/含"邻"竞品（render_stock 误报源）
    ("便利蜂", (0.85, 0.15, 0.1), (1.0, 0.85, 0.0)), ("邻几", (0.8, 0.05, 0.1), WHITE),
    ("好邻居", (0.0, 0.5, 0.25), WHITE), ("奥乐齐", (0.8, 0.05, 0.1), WHITE),
    ("苏宁小店", (0.95, 0.45, 0.0), WHITE), ("京东便利", (0.8, 0.05, 0.1), WHITE),
]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def srgb2lin(c):
    return tuple((v/12.92 if v <= 0.04045 else ((v+0.055)/1.055)**2.4) for v in c)


def mat(name, color, rough=0.6, metal=0.0, emit=None, emit_strength=0.0,
        transmission=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*srgb2lin(color), 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emit is not None:
        bsdf.inputs["Emission Color"].default_value = (*srgb2lin(emit), 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit_strength
    if transmission > 0:
        bsdf.inputs["Transmission Weight"].default_value = transmission
        bsdf.inputs["IOR"].default_value = 1.45
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
    return m


def box(name, size, loc, material, rot=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = (size[0], size[1], size[2])
    if rot:
        o.rotation_euler = rot
    o.data.materials.append(material)
    return o


def text_obj(name, body, font_key, size, loc, material, extrude=0.0,
             align="CENTER", shear=0.0):
    cu = bpy.data.curves.new(name, type="FONT")
    cu.body = body
    cu.font = bpy.data.fonts.load(FONTS[font_key], check_existing=True)
    cu.size = size
    cu.extrude = extrude
    cu.align_x = align
    cu.align_y = "CENTER"
    o = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(o)
    # 文字默认在 XY 平面朝 +Z；旋转使朝 -Y（面向街道），文字向上=世界 +Z
    R = Matrix.Rotation(math.radians(90), 4, "X")
    S = Matrix.Identity(4)
    if shear:
        S[0][1] = shear          # 书法代理：x 随 y 剪切（M0 AFFINE 同参 0.06）
    o.matrix_world = Matrix.Translation(Vector(loc)) @ R @ S
    o.data.materials.append(material)
    return o


def text_width(o):
    bpy.context.view_layer.update()
    return o.dimensions.x


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_render(exposure):
    sc = bpy.context.scene
    try:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x = W
    sc.render.resolution_y = H
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    try:
        sc.eevee.taa_render_samples = 48
        sc.eevee.use_raytracing = True
    except AttributeError:
        pass
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.exposure = exposure
    sc.render.image_settings.file_format = "PNG"


def setup_world(lighting, rng):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("world")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    if lighting == "day":
        col, strength = (0.55, 0.65, 0.8), 1.0
    elif lighting == "dusk":
        col, strength = (0.9, 0.5, 0.3), 0.35
    elif lighting == "rain":
        col, strength = (0.5, 0.53, 0.58), 0.55
    else:  # night
        col, strength = (0.05, 0.07, 0.13), 0.18
    bg.inputs[0].default_value = (*col, 1.0)
    bg.inputs[1].default_value = strength
    # 太阳/主光
    sun_data = bpy.data.lights.new("sun", type="SUN")
    if lighting == "day":
        sun_data.energy = 3.0
        sun_data.color = (1.0, 0.98, 0.92)
        elev = math.radians(rng.uniform(40, 65))
    elif lighting == "dusk":
        sun_data.energy = 1.3
        sun_data.color = (1.0, 0.6, 0.35)
        elev = math.radians(rng.uniform(8, 18))
    elif lighting == "rain":
        sun_data.energy = 1.1
        sun_data.color = (0.9, 0.92, 0.95)
        elev = math.radians(rng.uniform(30, 55))
    else:
        sun_data.energy = 0.05
        sun_data.color = (0.6, 0.7, 1.0)
        elev = math.radians(60)
    az = math.radians(rng.uniform(-60, 60))
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(90) - elev, 0, az)
    if lighting == "night":
        # 路灯（暖）+ 门面筒灯
        for i, (lx, ly, lz, e, c) in enumerate([
            (3.5, -4.0, 4.5, 800, (1.0, 0.75, 0.45)),
            (-3.5, -4.5, 4.8, 600, (1.0, 0.78, 0.5)),
            (0.0, -1.2, 2.6, 250, (1.0, 0.9, 0.7)),
        ]):
            ld = bpy.data.lights.new(f"lamp{i}", type="POINT")
            ld.energy = e
            ld.color = c
            ld.shadow_soft_size = 0.3
            lo = bpy.data.objects.new(f"lamp{i}", ld)
            bpy.context.collection.objects.link(lo)
            lo.location = (lx, ly, lz)


# ---------------------------------------------------------------------------
# 场景：立面 + 环境
# ---------------------------------------------------------------------------
def build_facade(rng, lighting, sign_cz):
    wall_col = rng.choice([(0.72, 0.7, 0.66), (0.62, 0.6, 0.58), (0.78, 0.75, 0.7),
                           (0.55, 0.55, 0.57), (0.7, 0.62, 0.55)])
    wall = box("wall", (12, 0.3, 6), (0, 0.15, 3), mat("wall", wall_col, rough=0.85))
    ground_col = (0.25, 0.25, 0.27) if lighting != "rain" else (0.18, 0.19, 0.22)
    box("ground", (24, 14, 0.1), (0, -7, -0.05),
        mat("ground", ground_col, rough=0.4 if lighting == "rain" else 0.8))
    # 玻璃橱窗 + 门（招牌下方）
    glass_m = mat("glass", (0.6, 0.7, 0.75), rough=0.05, transmission=1.0)
    frame_m = mat("frame", (0.2, 0.2, 0.22), rough=0.4, metal=0.6)
    glass_top = sign_cz - 0.75
    n_panels = rng.choice([2, 3])
    pw = 1.6
    total = n_panels * pw + 1.0
    x0 = -total / 2
    for i in range(n_panels):
        cx = x0 + i * (pw + 0.1) + pw / 2
        box(f"glass{i}", (pw, 0.04, glass_top - 0.15), (cx, -0.02, (glass_top + 0.15) / 2), glass_m)
        box(f"frame{i}", (pw + 0.08, 0.06, 0.08), (cx, -0.03, glass_top), frame_m)
    # 门
    box("door", (0.95, 0.05, glass_top - 0.15), (x0 + total - 0.55, -0.04, (glass_top + 0.15) / 2), glass_m)
    # 室内（玻璃后暗盒 + 夜间货架灯带）
    box("interior", (total + 1, 2.5, glass_top), (0, 1.4, glass_top / 2),
        mat("interior", (0.12, 0.11, 0.1), rough=0.9))
    if lighting in ("night", "dusk"):
        for i in range(3):
            box(f"shelf{i}", (total * 0.8, 0.05, 0.06),
                (0, 0.6, 0.6 + i * 0.7),
                mat(f"shelf{i}", (1, 0.95, 0.85), emit=(1, 0.95, 0.85), emit_strength=3.0))
    # 雨棚（招牌下沿）
    box("canopy", (total + 1.2, 0.7, 0.08), (0, -0.35, sign_cz - 0.62),
        mat("canopy", (0.35, 0.35, 0.38), rough=0.5, metal=0.3))
    # 邻栋
    for sx in (-1, 1):
        nc = rng.choice([(0.6, 0.58, 0.55), (0.66, 0.6, 0.52), (0.5, 0.52, 0.55)])
        box(f"neighbor{sx}", (5, 3, rng.uniform(5, 8)), (sx * 8.5, 1.5, 3),
            mat(f"neighbor{sx}", nc, rough=0.9))
        # 邻店小招牌（语境，非品牌）
        oc = rng.choice([(0.2, 0.4, 0.7), (0.7, 0.5, 0.1), (0.4, 0.6, 0.3)])
        box(f"nsign{sx}", (3.2, 0.1, 0.6), (sx * 8.5, -0.05, 3.1),
            mat(f"nsign{sx}", oc, rough=0.6))


def build_occluder(rng, sign_cz, sign_w):
    kind = rng.choice(["tree", "pole", "van", "none", "none"])
    if kind == "none":
        return
    side = rng.choice([-1, 1])
    x = side * (sign_w / 2 * rng.uniform(0.55, 0.85))
    if kind == "tree":
        trunk = mat("trunk", (0.35, 0.25, 0.18), rough=0.9)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.09, depth=2.6, location=(x, -2.2, 1.3))
        bpy.context.active_object.data.materials.append(trunk)
        leaf = mat("leaf", (0.15, 0.4, 0.18), rough=0.9)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=rng.uniform(0.9, 1.4),
                                              location=(x, -2.2, 3.0))
        bpy.context.active_object.data.materials.append(leaf)
    elif kind == "pole":
        bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=5.5, location=(x, -1.8, 2.75))
        bpy.context.active_object.data.materials.append(mat("pole", (0.3, 0.3, 0.32), metal=0.7, rough=0.4))
    else:  # van
        vc = rng.choice([(0.8, 0.8, 0.82), (0.25, 0.3, 0.5), (0.6, 0.2, 0.15)])
        box("van", (1.8, 4.2, 1.9), (x * 1.3, -3.5, 0.95), mat("van", vc, rough=0.3, metal=0.4))
        box("vanwin", (1.7, 1.2, 0.6), (x * 1.3, -4.9, 1.5), mat("vanwin", (0.1, 0.12, 0.15), rough=0.1))


# ---------------------------------------------------------------------------
# 招牌（类因子驱动）
# ---------------------------------------------------------------------------
def build_sign(master, rng, store_no, lighting, anomaly=False, hard=False):
    """返回 (组件 dict, sign 几何参数)。招牌前面 y=-0.16。"""
    h = 1.0
    w = h * rng.uniform(*master["aspect"])
    cz = rng.uniform(3.0, 3.5)
    depth = 0.12
    comps = {}
    glow = {"day": 0.0, "rain": 0.1, "dusk": 0.7, "night": 1.1}[lighting]
    letter_glow = {"day": 0.25, "rain": 0.4, "dusk": 1.4, "night": 2.2}[lighting]

    if anomaly:
        board_col, band_col = (0.05, 0.5, 0.2), (0.95, 0.8, 0.0)   # 违规 VI：绿板黄带
    else:
        board_col = RED if master["board"] == "red" else BLUE
        band_col = BLUE if master["board"] == "red" else RED
    board_m = mat("board", board_col, rough=0.35,
                  emit=board_col, emit_strength=glow)
    board = box("sign_board", (w, depth, h), (0, -0.10, cz), board_m)
    comps["board"] = board
    # 铝合金包边（E2 B3-3：30mm 包边）
    rim_m = mat("rim", (0.75, 0.76, 0.78), rough=0.3, metal=0.8)
    box("rim_top", (w + 0.06, depth + 0.02, 0.035), (0, -0.10, cz + h/2 + 0.015), rim_m)
    box("rim_bot", (w + 0.06, depth + 0.02, 0.035), (0, -0.10, cz - h/2 - 0.015), rim_m)
    # 色带
    if master["band"] == "blue_horizontal" and not anomaly:
        box("band_low", (w, 0.02, 0.10), (0, -0.165, cz - h/2 + 0.06),
            mat("band", BLUE, rough=0.4, emit=BLUE, emit_strength=glow))
    elif master["band"] == "blue_horizontal" and anomaly:
        box("band_low", (w, 0.02, 0.10), (0, -0.165, cz - h/2 + 0.06),
            mat("band", band_col, rough=0.4, emit=band_col, emit_strength=glow))
    elif master["band"] == "red_top":
        box("band_top", (w, 0.02, 0.07), (0, -0.165, cz + h/2 - 0.05),
            mat("band", RED, rough=0.4, emit=RED, emit_strength=glow))
    # 彩钢条（CS-06，E2 白蓝白红白 5 色竖条，占招牌右 12%）
    text_right = w/2
    if master["caigang"]:
        strip_w = w * 0.12 / 5
        cols = [WHITE, BLUE, WHITE, RED, WHITE]
        for i, c in enumerate(cols):
            sx = w/2 - w*0.12 + strip_w*(i+0.5)
            box(f"caigang{i}", (strip_w*0.92, 0.03, h*0.96), (sx, -0.17, cz),
                mat(f"caigang{i}", c, rough=0.45, emit=c, emit_strength=glow*0.6))
        text_right = w/2 - w*0.12

    # 主标
    suffix = master.get("suffix", "")
    sep = master["separator"]
    if hard:
        # 难负例：保留红/蓝板结构，主标换异品牌名（教模型辨文字而非颜色）
        main_body = rng.choice([b[0] for b in OTHER_BRANDS])
        sep = "dot"
    elif sep == "circle":
        main_body = None            # 分两段 + ⊙
    elif sep == "dot":
        main_body = "左邻·右舍" + suffix
    else:
        main_body = "左邻右舍" + suffix
    text_h = 0.46 * h
    avail = (text_right + w/2) * 0.86
    white_m = mat("letters", WHITE, rough=0.4,
                  emit=(1, 0.98, 0.92), emit_strength=letter_glow if master["dim3d"] else glow*0.35)
    extrude = 0.05 if master["dim3d"] else 0.004
    text_cz = cz + (0.06*h if master["band"] == "red_top" else 0.10*h)
    text_cx = (-w/2 + text_right) / 2
    shear = 0.06 if master["font_proxy"] else 0.0

    if sep == "circle":
        tl = text_obj("t_left", "左邻", master["font"], text_h, (0, -0.17, text_cz), white_m,
                      extrude=extrude, align="CENTER", shear=shear)
        tr = text_obj("t_right", "右舍", master["font"], text_h, (0, -0.17, text_cz), white_m,
                      extrude=extrude, align="CENTER", shear=shear)
        wl, wr = text_width(tl), text_width(tr)
        gap = 0.30 * h
        total = wl + gap + wr
        lx = text_cx - total/2 + wl/2
        rx = text_cx + total/2 - wr/2
        tl.location.x = lx
        tr.location.x = rx
        # ⊙ 圆环 + 圆心
        gcx = text_cx
        torus_m = white_m
        bpy.ops.mesh.primitive_torus_add(major_radius=0.11*h, minor_radius=0.024*h,
                                         location=(gcx, -0.18, text_cz),
                                         rotation=(math.radians(90), 0, 0))
        bpy.context.active_object.name = "logo_ring"
        bpy.context.active_object.data.materials.append(torus_m)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.045*h, depth=0.02,
                                            location=(gcx, -0.18, text_cz),
                                            rotation=(math.radians(90), 0, 0))
        bpy.context.active_object.name = "logo_dot"
        bpy.context.active_object.data.materials.append(torus_m)
        comps["main_text"] = tl
        main_w = total
    else:
        t = text_obj("main_text", main_body, master["font"], text_h,
                     (text_cx, -0.17, text_cz), white_m, extrude=extrude, shear=shear)
        mw = text_width(t)
        if mw > avail:              # 自适应字号（M0 同逻辑）
            scale = avail / mw
            t.scale = (scale, scale, scale)
            mw = avail
        comps["main_text"] = t
        main_w = mw

    # CS-04：蓝底板上居中红底白字 logo 面板（E2 B3-6，无论多长只放一组）
    if master["board"] == "blue" and not anomaly:
        lp_w, lp_h = main_w * 1.18, text_h * 1.55
        box("logo_panel", (lp_w, 0.035, lp_h), (text_cx, -0.185, text_cz),
            mat("logo_panel", RED, rough=0.35, emit=RED, emit_strength=glow))
        # 文字前移到面板前
        comps["main_text"].location.y = -0.21

    # 编号牌（E2 B3-7 红色铝板 300×300 NO.XXXX，招牌右下）
    if master["number"]:
        plate = box("num_plate", (0.32, 0.025, 0.32), (w/2 - 0.25, -0.06, cz - h/2 - 0.35),
                    mat("num_plate", RED, rough=0.4, metal=0.3, emit=RED, emit_strength=glow*0.5))
        nt = text_obj("num_text", f"No.{store_no:04d}", "arial", 0.085,
                      (w/2 - 0.25, -0.08, cz - h/2 - 0.35), white_m)
        comps["number"] = plate

    # 英文副标
    if master["en_sub"]:
        en = rng.choice(["Good Neighbors", "GoodNeighbor", "GOOD NEIGHBORS"])
        et = text_obj("en_sub", en, "arial", 0.13*h,
                      (text_cx - main_w*0.30, -0.17, text_cz - 0.34*h), white_m, align="CENTER")
        comps["en_sub"] = et

    # PLUS 角标
    if master["plus"]:
        px = w/2 - 0.42
        box("plus_plate", (0.52, 0.03, 0.30), (px, -0.18, cz + 0.18*h),
            mat("plus_plate", (0.14, 0.15, 0.18), rough=0.4))
        pt = text_obj("plus_text", "PLUS", "arial", 0.17, (px, -0.20, cz + 0.18*h), white_m)
        comps["plus"] = pt

    return comps, dict(w=w, h=h, cz=cz, depth=depth)


def build_other_sign(rng, lighting):
    name, col, tcol = rng.choice(OTHER_BRANDS)
    h = rng.uniform(0.8, 1.2)
    w = h * rng.uniform(2.8, 5.0)
    cz = rng.uniform(3.0, 3.5)
    glow = {"day": 0.0, "rain": 0.1, "dusk": 0.6, "night": 1.0}[lighting]
    board = box("sign_board", (w, 0.12, h), (0, -0.10, cz),
                mat("ob_board", col, rough=0.4, emit=col, emit_strength=glow))
    t = text_obj("ob_text", name, "hei", 0.5*h, (0, -0.17, cz),
                 mat("ob_letters", tcol, rough=0.4, emit=tcol, emit_strength=glow*0.4))
    if text_width(t) > w*0.85:
        s = w*0.85/text_width(t)
        t.scale = (s, s, s)
    return {"board": board, "main_text": t}, dict(w=w, h=h, cz=cz, depth=0.12)


# ---------------------------------------------------------------------------
# 相机 + 真值投影
# ---------------------------------------------------------------------------
def setup_camera(rng, geom):
    yaw = math.radians(rng.uniform(-30, 30))
    pitch = math.radians(rng.uniform(-8, 12))
    dist = rng.uniform(5.5, 11.0)
    target = Vector((0, -0.1, geom["cz"] - rng.uniform(0.0, 0.5)))
    cx = target.x + dist * math.cos(pitch) * math.sin(yaw)
    cy = target.y - dist * math.cos(pitch) * math.cos(yaw)
    cz = target.z + dist * math.sin(pitch)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = rng.uniform(28, 45)
    cam_data.sensor_width = 36
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (cx, cy, cz)
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam, dict(yaw_deg=math.degrees(yaw), pitch_deg=math.degrees(pitch), dist=dist)


def project_box(o, cam):
    sc = bpy.context.scene
    pts = []
    for corner in o.bound_box:
        wp = o.matrix_world @ Vector(corner)
        co = bpy_extras.object_utils.world_to_camera_view(sc, cam, wp)
        pts.append([round(co.x, 5), round(1.0 - co.y, 5)])   # 归一化，y 自上而下
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)], pts


import bpy_extras.object_utils


FORCE_OCC = False  # v2 车辆轴置 True：跳过 40% 掷币，强制建遮挡物


def render_one(seed, kind, style=None, lighting=None, out_dir=None):
    """kind: positive / other_brand / no_sign / anomaly / hard_negative。返回 manifest dict。"""
    rng = random.Random(seed)
    clear_scene()
    if lighting is None:
        lighting = rng.choices(["day", "dusk", "night", "rain"], weights=[45, 20, 25, 10])[0]
    exposure = rng.uniform(-0.4, 0.5)
    setup_render(exposure)
    setup_world(lighting, rng)

    anomaly = kind == "anomaly"
    hard = kind == "hard_negative"
    if kind in ("positive", "anomaly", "hard_negative"):
        master = MASTERS[style]
        cz0 = 3.2
        build_facade(rng, lighting, cz0)
        comps, geom = build_sign(master, rng, rng.randint(1, 9999), lighting, anomaly=anomaly, hard=hard)
        brand = "left_right" if kind == "positive" else "other_brand"
    elif kind == "other_brand":
        build_facade(rng, lighting, 3.2)
        comps, geom = build_other_sign(rng, lighting)
        brand = "other_brand"
    else:  # no_sign
        build_facade(rng, lighting, 3.2)
        comps, geom = {}, dict(w=0, h=0, cz=3.2, depth=0)
        brand = "other_brand"
    if rng.random() < 0.4 or FORCE_OCC:
        build_occluder(rng, geom["cz"], max(geom["w"], 2.0))

    cam, cam_params = setup_camera(rng, geom)
    bpy.context.view_layer.update()

    # 真值：招牌 polygon + 组件框
    sign_poly, comp_boxes = [], {}
    if "board" in comps:
        b = comps["board"]
        corners = []
        for corner in b.bound_box:
            wp = b.matrix_world @ Vector(corner)
            co = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, cam, wp)
            corners.append([round(co.x, 5), round(1.0 - co.y, 5)])
        sign_poly = corners
        for cname in ("main_text", "number", "en_sub", "plus"):
            if cname in comps:
                bb, _ = project_box(comps[cname], cam)
                comp_boxes[cname] = bb

    img_name = f"{kind}_{style or 'na'}_{seed:08d}.png"
    fp = os.path.join(out_dir, img_name)
    bpy.context.scene.render.filepath = fp
    bpy.ops.render.render(write_still=True)

    return {
        "image": img_name, "seed": seed, "gen_version": GEN_VERSION,
        "brand": brand, "kind": kind,
        "style_id": style if kind == "positive" else None,
        "anomaly": anomaly, "lighting": lighting,
        "asset_status": "candidate_unverified",
        "source_policy": "self_created", "split": "synthetic_train_only",
        "franchise_type": None,
        "font_proxy": bool(style and MASTERS[style]["font_proxy"]),
        "camera": cam_params, "sign_polygon": sign_poly, "components": comp_boxes,
        "resolution": [W, H],
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-cluster", type=int, default=60)
    ap.add_argument("--neg", type=int, default=160)
    ap.add_argument("--hard-neg", type=int, default=0,
                    help="难负例数：红/蓝板结构+异品牌文字，独立追加到 manifest")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--seed-base", type=int, default=20261111)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest_3d.jsonl")
    mode = "a" if args.hard_neg and not args.smoke and args.per_cluster == 0 else "w"

    jobs = []
    if args.smoke:
        for i, s in enumerate(MASTERS):
            jobs.append((args.seed_base + i, "positive", s, "day"))
        jobs.append((args.seed_base + 100, "positive", "CS-03", "night"))
        jobs.append((args.seed_base + 101, "positive", "CS-04", "night"))
        jobs.append((args.seed_base + 102, "other_brand", None, "day"))
        jobs.append((args.seed_base + 103, "anomaly", "CS-03", "day"))
        jobs.append((args.seed_base + 104, "no_sign", None, "dusk"))
    elif args.hard_neg and args.per_cluster == 0:
        seed = args.seed_base
        for i in range(args.hard_neg):
            s = list(MASTERS)[i % len(MASTERS)]
            jobs.append((seed, "hard_negative", s, None)); seed += 1
    else:
        seed = args.seed_base
        for s in MASTERS:
            for i in range(args.per_cluster):
                jobs.append((seed, "positive", s, None)); seed += 1
        n_other = int(args.neg * 0.6)
        n_nosign = int(args.neg * 0.25)
        n_anom = args.neg - n_other - n_nosign
        for i in range(n_other):
            jobs.append((seed, "other_brand", None, None)); seed += 1
        for i in range(n_nosign):
            jobs.append((seed, "no_sign", None, None)); seed += 1
        for i in range(n_anom):
            jobs.append((seed, "anomaly", "CS-03", None)); seed += 1

    with open(manifest_path, mode) as mf:
        for k, (seed, kind, style, lighting) in enumerate(jobs):
            rec = render_one(seed, kind, style, lighting, args.out)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            print(f"[{k+1}/{len(jobs)}] {rec['image']} {kind}/{style} {lighting}", flush=True)
    print(f"DONE {len(jobs)} renders → {args.out}", flush=True)


if __name__ == "__main__":
    main()