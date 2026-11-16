#!/usr/bin/env python3
"""
M1-3D 孪生 v3 — 目录外 lookalike 硬负例定向扩产(在 Blender 内运行)
=================================================================
blender -b -P m1_3d_twin_v3_lookalike.py -- --n 320 --out renders_3d_v3

动机: S2 负例探针残留 6% 错判集中在咖啡/茶饮/快餐门头(目录只有便利店/零食/超市),
负类空间缺新品类。按 OTHER_BRANDS 既定机制(品牌名+板色+字色, 程序化几何/字体,
无真实 logo)扩展 14 个非便利店品类范式。seed 段 20270101+ 与 v1/v2 不重叠。
治理: source_policy=self_created, split=synthetic_train_only, brand=other_brand。
"""
import bpy, os, sys, json

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=320)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders_3d_v3"))
ap.add_argument("--seed-base", type=int, default=20270101)
args = ap.parse_args(argv)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_3d_twin as m1

WHITE = m1.WHITE
# 咖啡/茶饮/快餐/生鲜/烘焙范式(板色, 字色) — 品类级配色家族, 非官方 logo 复刻
LOOKALIKE = [
    ("瑞幸咖啡", (0.05, 0.15, 0.45), WHITE),
    ("库迪咖啡", (0.85, 0.20, 0.05), WHITE),
    ("蜜雪冰城", (0.95, 0.95, 0.95), (0.85, 0.10, 0.10)),
    ("茶百道", (0.72, 0.55, 0.35), WHITE),
    ("星巴克咖啡", (0.00, 0.30, 0.18), WHITE),
    ("奈雪的茶", (0.95, 0.95, 0.95), (0.30, 0.30, 0.32)),
    ("古茗", (0.50, 0.10, 0.10), WHITE),
    ("沪上阿姨", (0.10, 0.25, 0.50), WHITE),
    ("华莱士", (0.90, 0.10, 0.10), (0.95, 0.85, 0.10)),
    ("正新鸡排", (0.85, 0.10, 0.10), WHITE),
    ("老乡鸡", (0.90, 0.75, 0.15), (0.15, 0.15, 0.15)),
    ("钱大妈", (0.50, 0.05, 0.30), WHITE),
    ("百果园", (0.10, 0.50, 0.25), WHITE),
    ("巴比馒头", (0.90, 0.50, 0.10), WHITE),
]
m1.OTHER_BRANDS = m1.OTHER_BRANDS + LOOKALIKE  # 进程内扩展, 其他品牌按原概率混采

os.makedirs(args.out, exist_ok=True)
manifestf = os.path.join(args.out, "manifest_3d_v3.jsonl")
seed = args.seed_base
n_ok = 0
with open(manifestf, "w") as mf:
    for i in range(args.n):
        try:
            rec = m1.render_one(seed, "other_brand", None, None, out_dir=args.out)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            n_ok += 1
        except Exception as e:
            print(f"seed {seed} FAIL {e}", flush=True)
        seed += 1
        if (i + 1) % 20 == 0:
            print(f"...{i+1}/{args.n}", flush=True)
print(f"v3 完成: {n_ok}/{args.n} → {args.out}", flush=True)
