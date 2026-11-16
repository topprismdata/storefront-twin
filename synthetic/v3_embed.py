#!/usr/bin/env python3
"""嵌入 v3 lookalike 渲染 → v3_feats.npz (供 S2 负例池扩展)。"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import visual_baseline as vb
from PIL import Image

ROOT = "/Users/mac/chatgpt-harvest/6af41562"
recs = [json.loads(l) for l in open(f"{ROOT}/synthetic/renders_3d_v3/manifest_3d_v3.jsonl")]
paths = [f"{ROOT}/synthetic/renders_3d_v3/{r['image']}" for r in recs]
print(f"v3 渲染 {len(paths)} 张, 嵌入中…", flush=True)
ECC, EDD = vb.Embedder("clip"), vb.Embedder("dinov2")
imgs = [Image.open(p).convert("RGB") for p in paths]
Vc = ECC.embed_pil(imgs)
Vd = EDD.embed_pil(imgs)
np.savez(f"{ROOT}/synthetic/vis_cache/v3_feats.npz", Vc=Vc, Vd=Vd,
         brands=np.array([r["brand"] for r in recs]))
print(f"完成 → vis_cache/v3_feats.npz ({Vc.shape[0]}×{Vc.shape[1]}+{Vd.shape[1]})", flush=True)
