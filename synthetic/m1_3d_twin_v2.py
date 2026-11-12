#!/usr/bin/env python3
"""
M1-3D 孪生 v2 — 按 E11 真实困难轴定向扩渲染（在 Blender 内运行）
=================================================================
blender -b -P m1_3d_twin_v2.py -- --per-axis 110 --out ../synthetic/renders_3d_v2

四轴（E11_STYLE_AUDIT 实证的真实缺口，VI 变体轴待品牌确认暂不渲染）：
  far              超远全景：dist 14-24m，广角 24-35mm，招牌 <5% 画面
  vehicle_occl     车辆级遮挡：货车强制停在招牌前方重叠区
  night            夜景霓虹：强制 night 光照（发光字/灯箱）
  vertical_corner  非常规形态：竖挂侧招 / 转角招（附加几何+真值）
机制：import m1_3d_twin 为模块，猴子补丁 setup_camera/build_occluder/clear_scene，
复用其 facade/sign/真值/渲染全链；manifest 加 axis 字段，v2 独立目录与 seed 段。
"""
import bpy, math, os, sys, json, random
from mathutils import Vector
import bpy_extras

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--per-axis", type=int, default=110)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders_3d_v2"))
ap.add_argument("--seed-base", type=int, default=20261112)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args(argv)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_3d_twin as m1

GEN_VERSION = "m1-3d-blender-v2.0"
STATE = {"axis": None, "extra_objs": [], "cam": None}


_orig_clear = m1.clear_scene


def clear_scene_keep():
    _orig_clear()


m1.clear_scene = clear_scene_keep  # render_one 内不清（wrapper 已清），保住附加几何


def setup_camera_v2(rng, geom):
    if STATE["axis"] == "far":
        yaw = math.radians(rng.uniform(-35, 35))
        pitch = math.radians(rng.uniform(-4, 10))
        dist = rng.uniform(14.0, 24.0)
    else:
        yaw = math.radians(rng.uniform(-30, 30))
        pitch = math.radians(rng.uniform(-8, 12))
        dist = rng.uniform(5.5, 11.0)
    target = Vector((0, -0.1, geom["cz"] - rng.uniform(0.0, 0.5)))
    cx = target.x + dist * math.cos(pitch) * math.sin(yaw)
    cy = target.y - dist * math.cos(pitch) * math.cos(yaw)
    cz = target.z + dist * math.sin(pitch)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = rng.uniform(24, 32) if STATE["axis"] == "far" else rng.uniform(28, 45)
    cam_data.sensor_width = 36
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (cx, cy, cz)
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    STATE["cam"] = cam
    return cam, dict(yaw_deg=math.degrees(yaw), pitch_deg=math.degrees(pitch), dist=dist)


m1.setup_camera = setup_camera_v2


_orig_occluder = m1.build_occluder


def build_occluder_v2(rng, sign_cz, sign_w):
    STATE["extra_objs"] = []
    if STATE["axis"] == "vehicle_occl":
        # 货车/三轮 强制停在招牌前方重叠区（保证真实遮挡）
        vc = rng.choice([(0.8, 0.8, 0.82), (0.25, 0.3, 0.5), (0.6, 0.2, 0.15), (0.9, 0.85, 0.3)])
        w = rng.uniform(1.8, 2.4)
        x = rng.uniform(-0.35, 0.35) * sign_w
        m1.box("truck", (w, 4.4, 2.1), (x, -2.6, 1.05), m1.mat("truck", vc, rough=0.3, metal=0.4))
        m1.box("trucktop", (w - 0.15, 1.6, 0.7), (x, -3.6, 1.75), m1.mat("trucktop", (0.1, 0.12, 0.15), rough=0.1))
        return
    if STATE["axis"] == "vertical_corner":
        side = rng.choice([-1, 1])
        if rng.random() < 0.6:
            # 竖挂侧招：窄长板 + 逐字竖排
            bw, bh = 0.45, rng.uniform(2.2, 3.0)
            x = side * (3.1 + rng.uniform(0.1, 0.5))
            board = m1.box("vsign", (bw, 0.12, bh), (x, -0.28, sign_cz - 0.3),
                           m1.mat("vboard", m1.RED, rough=0.5, emit=m1.RED, emit_strength=0.0))
            body = rng.choice(["左邻右舍", "便利店"])
            nch = len(body)
            ztop = sign_cz - 0.3 + bh / 2 - 0.35
            step = min(0.42, (bh - 0.7) / max(1, nch - 1))
            for ci, ch in enumerate(body):
                m1.text_obj(f"vtext{ci}", ch, rng.choice(["hei", "song_proxy"]), 0.34,
                            (x, -0.36, ztop - ci * step),
                            m1.mat(f"vtextm{ci}", (1, 1, 1), rough=0.4, emit=(1, 1, 1), emit_strength=0.35))
            STATE["extra_objs"].append(board)
        else:
            # 转角招：45° 斜面板 + 同角度文字
            from mathutils import Matrix
            bw = rng.uniform(1.6, 2.4)
            x = side * (3.4 + rng.uniform(0.0, 0.4))
            czs = sign_cz + rng.uniform(-0.4, 0.2)
            board = m1.box("csign", (bw, 0.1, 0.8), (x, -0.6, czs),
                           m1.mat("cboard", m1.RED, rough=0.5, emit=m1.RED, emit_strength=0.0),
                           rot=(0, math.radians(45 * side), 0))
            t = m1.text_obj("ctext", "左邻·右舍", rng.choice(["hei", "song_proxy"]), 0.26,
                            (0, 0, czs),
                            m1.mat("ctextm", (1, 1, 1), rough=0.4, emit=(1, 1, 1), emit_strength=0.35))
            t.matrix_world = (Matrix.Translation(Vector((x, -0.78, czs)))
                              @ Matrix.Rotation(math.radians(45 * side), 4, "Z")
                              @ Matrix.Rotation(math.radians(90), 4, "X"))
            STATE["extra_objs"].append(board)
        if rng.random() < 0.3:
            _orig_occluder(rng, sign_cz, sign_w)
        return
    _orig_occluder(rng, sign_cz, sign_w)


m1.build_occluder = build_occluder_v2


def project_extra(cam):
    """附加几何真值：相机空间投影框。"""
    out = []
    for o in STATE["extra_objs"]:
        if o is None:
            continue
        corners = []
        for c in o.bound_box:
            wp = o.matrix_world @ Vector(c)
            co = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, cam, wp)
            corners.append([round(co.x, 5), round(1.0 - co.y, 5)])
        out.append(corners)
    return out


def main():
    os.makedirs(args.out, exist_ok=True)
    mf = open(os.path.join(args.out, "manifest_v2.jsonl"), "a")
    styles = list(m1.MASTERS.keys())
    axes = ["far", "vehicle_occl", "night", "vertical_corner"]
    n = 2 if args.smoke else args.per_axis
    seed = args.seed_base
    done = 0
    for axis in axes:
        for i in range(n):
            seed += 1
            STATE["axis"] = axis
            m1.clear_scene()
            lighting = "night" if axis == "night" else None
            kind = "positive"
            style = rng_choice = random.Random(seed).choice(styles)
            r = m1.render_one(seed, kind, style=style, lighting=lighting, out_dir=args.out)
            r["gen_version"] = GEN_VERSION
            r["axis"] = axis
            r["extra_sign_polygons"] = project_extra(STATE["cam"]) if STATE["axis"] == "vertical_corner" else []
            mf.write(json.dumps(r, ensure_ascii=False) + "\n")
            mf.flush()
            done += 1
            if done % 20 == 0:
                print(f"[v2] {axis} {i+1}/{n} total={done}", flush=True)
    mf.close()
    print(f"[v2] DONE {done} renders -> {args.out}", flush=True)


main()
