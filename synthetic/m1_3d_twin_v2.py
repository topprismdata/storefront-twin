#!/usr/bin/env python3
"""M1-3D 孪生 v2：高质量 + 宽域随机化（文献 Tobin 2017 域随机化 + 混合训练）。
相对 v1 的升级：
  1. 程序化纹理：砖墙/招牌磨损/沥青地面/玻璃反射（Noise+Voronoi+ColorRamp，非纯色方块）
  2. 真实室内：玻璃后可见货架+商品色块（不再暗盒）
  3. 宽域相机：yaw ±75° / pitch -20~35° / dist 3~22m / lens 24~70mm（v1 仅 ±30°）
  4. 宽域光照：太阳方位全向 + 强度/色温随机 + 随机补光
  5. 传感器噪声后处理：高斯模糊/噪点/暗角/色偏/JPEG压缩（模拟真实相机，缩 sim2real 域差）
治理：self_created/synthetic_train_only；程序化纹理非真实照片贴图。
用法：blender -b -P m1_3d_twin_v2.py -- --per-cluster 60 --neg 160 --out renders_3d_v2
"""
import bpy, json, math, os, random, sys
from mathutils import Vector, Matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = os.path.join(ROOT, "renders_3d_v2")
FONT_DIR = os.path.join(ROOT, "fonts3d")
GEN_VERSION = "m1-3d-blender-v2.0"
W, H = 960, 720

RED = (230/255, 0/255, 18/255)
BLUE = (0/255, 93/255, 172/255)
WHITE = (0.93, 0.93, 0.93)

FONTS = {
    "hei": os.path.join(FONT_DIR, "hei.ttf"),
    "song": os.path.join(FONT_DIR, "song_bold.ttf"),
    "hira": os.path.join(FONT_DIR, "hira_w6.ttf"),
    "arial": os.path.join(FONT_DIR, "arial_bold.ttf"),
}

MASTERS = {
    "CS-01": dict(aspect=(3.2, 4.2), board="red", band="blue_horizontal", caigang=False,
                  font="hei", separator="dot", dim3d=False, number=True, en_sub=False, plus=False, font_proxy=False),
    "CS-02": dict(aspect=(3.0, 3.8), board="red", band="blue_horizontal", caigang=False,
                  font="song", separator="dot", dim3d=False, number=True, en_sub=False, plus=False, font_proxy=True),
    "CS-03": dict(aspect=(3.4, 4.4), board="red", band="blue_horizontal", caigang=False,
                  font="hei", separator="circle", dim3d=False, number=True, en_sub=True, plus=False, font_proxy=False),
    "CS-04": dict(aspect=(3.2, 4.0), board="blue", band="red_top", caigang=False,
                  font="hei", separator="circle", dim3d=True, number=True, en_sub=True, plus=False, font_proxy=False),
    "CS-05": dict(aspect=(3.4, 4.2), board="red", band="blue_horizontal", caigang=False,
                  font="hei", separator="dot", dim3d=False, number=True, en_sub=True, plus=True, font_proxy=False),
    "CS-06": dict(aspect=(4.0, 5.0), board="red", band="blue_horizontal", caigang=True,
                  font="hei", separator="dot", dim3d=False, number=True, en_sub=False, plus=False, font_proxy=False),
}

OTHER_BRANDS = [
    ("美宜佳", (0.85, 0.1, 0.1), WHITE), ("天福便利", (0.8, 0.15, 0.1), WHITE),
    ("芙蓉兴盛", (0.9, 0.45, 0.05), WHITE), ("邻几", (0.0, 0.5, 0.3), WHITE),
    ("好邻居", (0.1, 0.35, 0.7), WHITE), ("便利蜂", (0.95, 0.75, 0.0), (0.1, 0.1, 0.1)),
    ("苏宁小店", (0.95, 0.6, 0.0), (0.1, 0.1, 0.1)), ("十足", (0.0, 0.45, 0.75), WHITE),
    ("罗森", (0.0, 0.3, 0.65), WHITE), ("全家", (0.0, 0.55, 0.35), WHITE),
]


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


def _noise_node(nt, scale=8.0, detail=4.0, rough=0.6):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    return n


def mat_brick(name, base_col, rng):
    """程序化砖墙：Voronoi 砖块 + Noise 灰缝 + 颜色变化。"""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.75, 0.95)
    tex = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (rng.uniform(3, 5), rng.uniform(6, 9), 1)
    vor = nt.nodes.new("ShaderNodeTexVoronoi"); vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 1.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[1].position = rng.uniform(0.04, 0.09)
    noise = _noise_node(nt, scale=rng.uniform(15, 30), detail=3)
    mixc = nt.nodes.new("ShaderNodeMixRGB"); mixc.blend_type = "MULTIPLY"
    mixc.inputs["Fac"].default_value = 0.35
    c1 = nt.nodes.new("ShaderNodeRGB"); c1.outputs[0].default_value = (*srgb2lin(base_col), 1)
    c2 = nt.nodes.new("ShaderNodeRGB")
    var = tuple(max(0, min(1, v * rng.uniform(0.8, 1.1))) for v in base_col)
    c2.outputs[0].default_value = (*srgb2lin(var), 1)
    nt.links.new(tex.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], vor.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(vor.outputs["Distance"], ramp.inputs["Fac"])
    nt.links.new(c1.outputs[0], mixc.inputs["Color1"])
    nt.links.new(c2.outputs[0], mixc.inputs["Color2"])
    nt.links.new(noise.outputs["Fac"], mixc.inputs["Fac"])
    # 砖缝暗化：ramp 输出乘到颜色
    mul = nt.nodes.new("ShaderNodeMixRGB"); mul.blend_type = "MULTIPLY"; mul.inputs["Fac"].default_value = 0.6
    nt.links.new(mixc.outputs["Color"], mul.inputs["Color1"])
    nt.links.new(ramp.outputs["Color"], mul.inputs["Color2"])
    nt.links.new(mul.outputs["Color"], bsdf.inputs["Base Color"])
    # 凹凸
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.25
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def mat_worn_sign(name, color, rng, glow=0.0):
    """招牌铝板：Noise 磨损 + 粗糙度变化 + 轻微污渍。"""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Metallic"].default_value = 0.15
    noise = _noise_node(nt, scale=rng.uniform(3, 7), detail=5, rough=0.7)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.75
    base = nt.nodes.new("ShaderNodeRGB"); base.outputs[0].default_value = (*srgb2lin(color), 1)
    dark = tuple(v * 0.82 for v in color)
    dk = nt.nodes.new("ShaderNodeRGB"); dk.outputs[0].default_value = (*srgb2lin(dark), 1)
    mix = nt.nodes.new("ShaderNodeMixRGB"); mix.blend_type = "MIX"
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(dk.outputs[0], mix.inputs["Color1"])
    nt.links.new(base.outputs[0], mix.inputs["Color2"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["From Min"].default_value = 0; rr.inputs["From Max"].default_value = 1
    rr.inputs["To Min"].default_value = 0.25; rr.inputs["To Max"].default_value = 0.55
    nt.links.new(noise.outputs["Fac"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], bsdf.inputs["Roughness"])
    if glow > 0:
        bsdf.inputs["Emission Color"].default_value = (*srgb2lin(color), 1)
        bsdf.inputs["Emission Strength"].default_value = glow
    return m


def mat_asphalt(name, rng, wet=False):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes.get("Principled BSDF")
    noise = _noise_node(nt, scale=rng.uniform(40, 80), detail=6, rough=0.8)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.04, 0.04, 0.045, 1)
    ramp.color_ramp.elements[1].color = (0.13, 0.13, 0.14, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.25 if wet else 0.85
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.15
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
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


def text_obj(name, body, font_key, size, loc, material, extrude=0.0, align="CENTER", shear=0.0):
    cu = bpy.data.curves.new(name, type="FONT")
    cu.body = body
    cu.font = bpy.data.fonts.load(FONTS[font_key], check_existing=True)
    cu.size = size
    cu.extrude = extrude
    cu.align_x = align
    cu.align_y = "CENTER"
    o = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(o)
    R = Matrix.Rotation(math.radians(90), 4, "X")
    S = Matrix.Identity(4)
    if shear:
        S[0][1] = shear
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
        sc.eevee.taa_render_samples = 64
        sc.eevee.use_raytracing = True
    except AttributeError:
        pass
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.exposure = exposure
    sc.render.image_settings.file_format = "PNG"


def setup_world(lighting, rng):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("world"); sc.world = w; w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    if lighting == "day":
        col, strength = (0.55, 0.65, 0.8), rng.uniform(0.8, 1.3)
    elif lighting == "dusk":
        col, strength = (0.9, 0.5, 0.3), rng.uniform(0.25, 0.5)
    elif lighting == "rain":
        col, strength = (0.5, 0.53, 0.58), rng.uniform(0.4, 0.7)
    else:
        col, strength = (0.05, 0.07, 0.13), rng.uniform(0.12, 0.25)
    bg.inputs[0].default_value = (*col, 1.0)
    bg.inputs[1].default_value = strength
    sun_data = bpy.data.lights.new("sun", type="SUN")
    if lighting == "day":
        sun_data.energy = rng.uniform(2.5, 4.0); sun_data.color = (1.0, 0.98, 0.92)
        elev = math.radians(rng.uniform(25, 70))
    elif lighting == "dusk":
        sun_data.energy = rng.uniform(1.0, 1.8); sun_data.color = (1.0, 0.6, 0.35)
        elev = math.radians(rng.uniform(5, 22))
    elif lighting == "rain":
        sun_data.energy = rng.uniform(0.8, 1.4); sun_data.color = (0.9, 0.92, 0.95)
        elev = math.radians(rng.uniform(25, 60))
    else:
        sun_data.energy = 0.05; sun_data.color = (0.6, 0.7, 1.0); elev = math.radians(60)
    az = math.radians(rng.uniform(-180, 180))  # 宽域：全向方位（v1 仅 ±60）
    sun = bpy.data.objects.new("sun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(90) - elev, 0, az)
    if lighting == "night":
        for i, (lx, ly, lz, e, c) in enumerate([
            (3.5, -4.0, 4.5, 800, (1.0, 0.75, 0.45)),
            (-3.5, -4.5, 4.8, 600, (1.0, 0.78, 0.5)),
            (0.0, -1.2, 2.6, 250, (1.0, 0.9, 0.7)),
        ]):
            ld = bpy.data.lights.new(f"lamp{i}", type="POINT")
            ld.energy = e * rng.uniform(0.7, 1.3); ld.color = c; ld.shadow_soft_size = 0.3
            lo = bpy.data.objects.new(f"lamp{i}", ld)
            bpy.context.collection.objects.link(lo); lo.location = (lx, ly, lz)


def build_facade(rng, lighting, sign_cz):
    wall_col = rng.choice([(0.72, 0.7, 0.66), (0.62, 0.6, 0.58), (0.78, 0.75, 0.7),
                           (0.55, 0.55, 0.57), (0.7, 0.62, 0.55), (0.68, 0.64, 0.6)])
    wall_style = rng.choice(["brick", "plaster", "tile"])
    if wall_style == "brick":
        wall_m = mat_brick("wall", wall_col, rng)
    else:
        wall_m = mat_worn_sign("wall", wall_col, rng)
    box("wall", (12, 0.3, 6), (0, 0.15, 3), wall_m)
    ground_m = mat_asphalt("ground", rng, wet=(lighting == "rain"))
    box("ground", (24, 14, 0.1), (0, -7, -0.05), ground_m)
    # 人行道
    box("sidewalk", (14, 2.5, 0.12), (0, -2.2, 0.06),
        mat_worn_sign("sidewalk", (0.5, 0.5, 0.5), rng))
    glass_m = mat("glass", (0.6, 0.7, 0.75), rough=0.05, transmission=1.0)
    frame_m = mat("frame", (0.2, 0.2, 0.22), rough=0.4, metal=0.6)
    glass_top = sign_cz - 0.75
    n_panels = rng.choice([2, 3, 4])
    pw = 1.6
    total = n_panels * pw + 1.0
    x0 = -total / 2
    for i in range(n_panels):
        cx = x0 + i * (pw + 0.1) + pw / 2
        box(f"glass{i}", (pw, 0.04, glass_top - 0.15), (cx, -0.02, (glass_top + 0.15) / 2), glass_m)
        box(f"frame{i}", (pw + 0.08, 0.06, 0.08), (cx, -0.03, glass_top), frame_m)
    box("door", (0.95, 0.05, glass_top - 0.15), (x0 + total - 0.55, -0.04, (glass_top + 0.15) / 2), glass_m)
    # 真实室内：货架 + 商品色块（玻璃后可见，不再暗盒）
    box("interior_back", (total + 1, 0.2, glass_top), (0, 2.4, glass_top / 2),
        mat("interior_back", (0.35, 0.33, 0.3), rough=0.9))
    box("interior_floor", (total + 1, 2.5, 0.05), (0, 1.4, 0.1),
        mat("interior_floor", (0.4, 0.38, 0.36), rough=0.7))
    shelf_m = mat("shelf", (0.55, 0.55, 0.58), rough=0.5, metal=0.3)
    prod_cols = [(0.8, 0.2, 0.15), (0.2, 0.5, 0.8), (0.9, 0.75, 0.1), (0.3, 0.7, 0.3),
                 (0.9, 0.9, 0.9), (0.6, 0.3, 0.6), (0.95, 0.5, 0.2)]
    for si in range(3):
        sz = 0.5 + si * 0.65
        box(f"shelf{si}", (total * 0.85, 0.4, 0.04), (0, 1.0, sz), shelf_m)
        # 商品色块
        for pi in range(rng.randint(4, 7)):
            px = -total * 0.4 + pi * (total * 0.8 / 6)
            pc = rng.choice(prod_cols)
            box(f"prod{si}_{pi}", (0.18, 0.15, rng.uniform(0.2, 0.35)),
                (px, 1.0, sz + 0.15), mat(f"prod{si}_{pi}", pc, rough=0.6))
    if lighting in ("night", "dusk"):
        box("ceiling_light", (total * 0.8, 0.3, 0.05), (0, 1.2, glass_top - 0.1),
            mat("clight", (1, 0.95, 0.85), emit=(1, 0.95, 0.85), emit_strength=4.0))
    box("canopy", (total + 1.2, 0.7, 0.08), (0, -0.35, sign_cz - 0.62),
        mat_worn_sign("canopy", (0.35, 0.35, 0.38), rng))
    for sx in (-1, 1):
        nc = rng.choice([(0.6, 0.58, 0.55), (0.66, 0.6, 0.52), (0.5, 0.52, 0.55)])
        nm = mat_brick(f"neighbor{sx}", nc, rng) if rng.random() < 0.5 else mat_worn_sign(f"neighbor{sx}", nc, rng)
        box(f"neighbor{sx}", (5, 3, rng.uniform(5, 8)), (sx * 8.5, 1.5, 3), nm)
        oc = rng.choice([(0.2, 0.4, 0.7), (0.7, 0.5, 0.1), (0.4, 0.6, 0.3)])
        box(f"nsign{sx}", (3.2, 0.1, 0.6), (sx * 8.5, -0.05, 3.1), mat_worn_sign(f"nsign{sx}", oc, rng))


def build_occluder(rng, sign_cz, sign_w):
    kind = rng.choice(["tree", "pole", "van", "none", "none"])
    if kind == "none":
        return
    x = rng.choice([-1, 1]) * rng.uniform(2.5, 4.5)
    if kind == "tree":
        box("trunk", (0.25, 0.25, 2.5), (x, -3.5, 1.25), mat("trunk", (0.3, 0.22, 0.15), rough=0.9))
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=rng.uniform(1.2, 1.8), location=(x, -3.5, 3.2))
        bpy.context.active_object.name = "foliage"
        bpy.context.active_object.data.materials.append(mat("foliage", (0.15, 0.35, 0.12), rough=0.9))
    elif kind == "pole":
        box("pole", (0.12, 0.12, 5), (x, -3.0, 2.5), mat("pole", (0.3, 0.3, 0.32), rough=0.4, metal=0.7))
    elif kind == "van":
        box("vanbody", (3.5, 1.8, 1.8), (x * 1.3, -4.9, 1.0), mat("van", (0.7, 0.7, 0.72), rough=0.3, metal=0.4))
        box("vanwin", (1.7, 1.2, 0.6), (x * 1.3, -4.9, 1.5), mat("vanwin", (0.1, 0.12, 0.15), rough=0.1))


# ---------------------------------------------------------------------------
# 招牌组件（门头组件，v2 磨损材质）
# ---------------------------------------------------------------------------
def build_sign(master, rng, store_no, lighting, anomaly=False, hard=False):
    h = 1.0
    w = h * rng.uniform(*master["aspect"])
    cz = rng.uniform(3.0, 3.5)
    depth = 0.12
    comps = {}
    glow = {"day": 0.0, "rain": 0.1, "dusk": 0.7, "night": 1.1}[lighting]
    letter_glow = {"day": 0.25, "rain": 0.4, "dusk": 1.4, "night": 2.2}[lighting]
    if anomaly:
        board_col, band_col = (0.05, 0.5, 0.2), (0.95, 0.8, 0.0)
    else:
        board_col = RED if master["board"] == "red" else BLUE
        band_col = BLUE if master["board"] == "red" else RED
    board_m = mat_worn_sign("board", board_col, rng, glow=glow)
    board = box("sign_board", (w, depth, h), (0, -0.10, cz), board_m)
    comps["board"] = board
    rim_m = mat("rim", (0.75, 0.76, 0.78), rough=0.3, metal=0.8)
    box("rim_top", (w + 0.06, depth + 0.02, 0.035), (0, -0.10, cz + h/2 + 0.015), rim_m)
    box("rim_bot", (w + 0.06, depth + 0.02, 0.035), (0, -0.10, cz - h/2 - 0.015), rim_m)
    if master["band"] == "blue_horizontal":
        bc = band_col if anomaly else BLUE
        box("band_low", (w, 0.02, 0.10), (0, -0.165, cz - h/2 + 0.06),
            mat_worn_sign("band", bc, rng, glow=glow))
    elif master["band"] == "red_top":
        box("band_top", (w, 0.02, 0.07), (0, -0.165, cz + h/2 - 0.05),
            mat_worn_sign("band", RED, rng, glow=glow))
    text_right = w/2
    if master["caigang"]:
        strip_w = w * 0.12 / 5
        cols = [WHITE, BLUE, WHITE, RED, WHITE]
        for i, c in enumerate(cols):
            sx = w/2 - w*0.12 + strip_w*(i+0.5)
            box(f"caigang{i}", (strip_w*0.92, 0.03, h*0.96), (sx, -0.17, cz),
                mat(f"caigang{i}", c, rough=0.45, emit=c, emit_strength=glow*0.6))
        text_right = w/2 - w*0.12
    suffix = master.get("suffix", "")
    sep = master["separator"]
    if hard:
        main_body = rng.choice([b[0] for b in OTHER_BRANDS]); sep = "dot"
    elif sep == "circle":
        main_body = None
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
        tl = text_obj("t_left", "左邻", master["font"], text_h, (0, -0.17, text_cz), white_m, extrude=extrude, shear=shear)
        tr = text_obj("t_right", "右舍", master["font"], text_h, (0, -0.17, text_cz), white_m, extrude=extrude, shear=shear)
        wl, wr = text_width(tl), text_width(tr)
        gap = 0.30 * h
        total = wl + gap + wr
        tl.location.x = text_cx - total/2 + wl/2
        tr.location.x = text_cx + total/2 - wr/2
        gcx = text_cx
        bpy.ops.mesh.primitive_torus_add(major_radius=0.11*h, minor_radius=0.024*h,
                                         location=(gcx, -0.18, text_cz), rotation=(math.radians(90), 0, 0))
        bpy.context.active_object.name = "logo_ring"
        bpy.context.active_object.data.materials.append(white_m)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.045*h, depth=0.02,
                                            location=(gcx, -0.18, text_cz), rotation=(math.radians(90), 0, 0))
        bpy.context.active_object.name = "logo_dot"
        bpy.context.active_object.data.materials.append(white_m)
        comps["main_text"] = tl
        main_w = total
    else:
        t = text_obj("main_text", main_body, master["font"], text_h,
                     (text_cx, -0.17, text_cz), white_m, extrude=extrude, shear=shear)
        mw = text_width(t)
        if mw > avail:
            scale = avail / mw
            t.scale = (scale, scale, scale); mw = avail
        comps["main_text"] = t
        main_w = mw
    if master["board"] == "blue" and not anomaly:
        lp_w, lp_h = main_w * 1.18, text_h * 1.55
        box("logo_panel", (lp_w, 0.035, lp_h), (text_cx, -0.185, text_cz),
            mat_worn_sign("logo_panel", RED, rng, glow=glow))
        comps["main_text"].location.y = -0.21
    if master["number"]:
        plate = box("num_plate", (0.32, 0.025, 0.32), (w/2 - 0.25, -0.06, cz - h/2 - 0.35),
                    mat_worn_sign("num_plate", RED, rng, glow=glow*0.5))
        text_obj("num_text", f"No.{store_no:04d}", "arial", 0.085,
                 (w/2 - 0.25, -0.08, cz - h/2 - 0.35), white_m)
        comps["number"] = plate
    if master["en_sub"]:
        en = rng.choice(["Good Neighbors", "GoodNeighbor", "GOOD NEIGHBORS"])
        text_obj("en_sub", en, "arial", 0.13*h,
                 (text_cx - main_w*0.30, -0.17, text_cz - 0.34*h), white_m)
        comps["en_sub"] = comps["main_text"]
    if master["plus"]:
        px = w/2 - 0.42
        box("plus_plate", (0.52, 0.03, 0.30), (px, -0.18, cz + 0.18*h), mat("plus_plate", (0.14, 0.15, 0.18), rough=0.4))
        pt = text_obj("plus_text", "PLUS", "arial", 0.17, (px, -0.20, cz + 0.18*h), white_m)
        comps["plus"] = pt
    return comps, dict(w=w, h=h, cz=cz, depth=depth)


def build_other_sign(rng, lighting):
    name, col, tcol = rng.choice(OTHER_BRANDS)
    w = rng.uniform(3.0, 4.5); h = 1.0; cz = rng.uniform(3.0, 3.5)
    board = box("sign_board", (w, 0.12, h), (0, -0.10, cz), mat_worn_sign("board", col, rng))
    t = text_obj("main_text", name, "hei", 0.46*h, (0, -0.17, cz + 0.1*h),
                 mat("letters", tcol, rough=0.4))
    mw = text_width(t)
    if mw > w * 0.86:
        s = w * 0.86 / mw; t.scale = (s, s, s)
    return {"board": board, "main_text": t}, dict(w=w, h=h, cz=cz, depth=0.12)


# ---------------------------------------------------------------------------
# 宽域相机（v2: yaw ±75° / pitch -20~35° / dist 3~22m）
# ---------------------------------------------------------------------------
def setup_camera(rng, geom):
    yaw = math.radians(rng.uniform(-75, 75))
    pitch = math.radians(rng.uniform(-20, 35))
    dist = rng.uniform(3.0, 22.0)
    target = Vector((0, -0.1, geom["cz"] - rng.uniform(0.0, 0.5)))
    cx = target.x + dist * math.cos(pitch) * math.sin(yaw)
    cy = target.y - dist * math.cos(pitch) * math.cos(yaw)
    cz = target.z + dist * math.sin(pitch)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = rng.uniform(24, 70)
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
    xs, ys = [], []
    for corner in o.bound_box:
        wp = o.matrix_world @ Vector(corner)
        co = bpy_extras.object_utils.world_to_camera_view(sc, cam, wp)
        xs.append(co.x); ys.append(1.0 - co.y)
    return [min(xs), min(ys), max(xs), max(ys)], None


import bpy_extras.object_utils


# ---------------------------------------------------------------------------
# 传感器噪声后处理（缩 sim2real 域差：模糊/噪点/暗角/色偏/JPEG）
# ---------------------------------------------------------------------------
def sensor_postprocess(fp, rng):
    import numpy as np
    try:
        import cv2
    except ImportError:
        return
    img = cv2.imread(fp)
    if img is None:
        return
    img = img.astype(np.float32)
    # 高斯模糊（镜头/运动）
    if rng.random() < 0.7:
        k = rng.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), rng.uniform(0.5, 1.5))
    # 传感器噪点
    if rng.random() < 0.8:
        sigma = rng.uniform(2, 9)
        img += np.random.normal(0, sigma, img.shape).astype(np.float32)
    # 色偏（白平衡漂移）
    if rng.random() < 0.6:
        gains = np.array([rng.uniform(0.92, 1.08) for _ in range(3)], np.float32)
        img *= gains
    # 暗角
    if rng.random() < 0.7:
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cy, cx = h/2, w/2
        r = np.sqrt((X-cx)**2 + (Y-cy)**2) / np.sqrt(cx**2 + cy**2)
        vig = 1.0 - rng.uniform(0.15, 0.4) * (r ** 2)
        img *= vig[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    # JPEG 压缩伪影
    q = rng.randint(55, 92)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    cv2.imwrite(fp, img)


def render_one(seed, kind, style=None, lighting=None, out_dir=None):
    rng = random.Random(seed)
    clear_scene()
    if lighting is None:
        lighting = rng.choices(["day", "dusk", "night", "rain"], weights=[45, 20, 25, 10])[0]
    exposure = rng.uniform(-0.6, 0.7)
    setup_render(exposure)
    setup_world(lighting, rng)
    anomaly = kind == "anomaly"
    hard = kind == "hard_negative"
    if kind in ("positive", "anomaly", "hard_negative"):
        master = MASTERS[style]
        build_facade(rng, lighting, 3.2)
        comps, geom = build_sign(master, rng, rng.randint(1, 9999), lighting, anomaly=anomaly, hard=hard)
        brand = "left_right" if kind == "positive" else "other_brand"
    elif kind == "other_brand":
        build_facade(rng, lighting, 3.2)
        comps, geom = build_other_sign(rng, lighting)
        brand = "other_brand"
    else:
        build_facade(rng, lighting, 3.2)
        comps, geom = {}, dict(w=0, h=0, cz=3.2, depth=0)
        brand = "other_brand"
    if rng.random() < 0.5:
        build_occluder(rng, geom["cz"], max(geom["w"], 2.0))
    cam, cam_params = setup_camera(rng, geom)
    bpy.context.view_layer.update()
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
    sensor_postprocess(fp, rng)
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
    ap.add_argument("--hard-neg", type=int, default=0)
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
        n_other = int(args.neg * 0.6); n_nosign = int(args.neg * 0.25)
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
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n"); mf.flush()
            print(f"[{k+1}/{len(jobs)}] {rec['image']} {kind}/{style} {lighting}", flush=True)
    print(f"DONE {len(jobs)} renders → {args.out}", flush=True)


if __name__ == "__main__":
    main()