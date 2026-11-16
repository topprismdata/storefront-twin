#!/usr/bin/env python3
"""嵌入 v19 真实负例 → v19_feats.npz (供 S2 负例池扩展)。"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import visual_baseline as vb
from PIL import Image

ROOT = "/Users/mac/chatgpt-harvest/6af41562"
log = json.load(open(f"{ROOT}/data/amap_v19_teafood_log.json"))
items = [v for v in (log.values() if isinstance(log, dict) else log)
         if isinstance(v, dict) and v.get("sha256")]
paths = [f"{ROOT}/data/images_v19_teafood/{v['file']}" for v in items]
print(f"v19 真实负例 {len(paths)} 张, 嵌入中…", flush=True)
ECC, EDD = vb.Embedder("clip"), vb.Embedder("dinov2")
imgs = [Image.open(p).convert("RGB") for p in paths]
Vc = ECC.embed_pil(imgs)
Vd = EDD.embed_pil(imgs)
np.savez(f"{ROOT}/synthetic/vis_cache/v19_feats.npz", Vc=Vc, Vd=Vd,
         brands=np.array([v["brand_query"] for v in items]))
print(f"完成 → vis_cache/v19_feats.npz ({Vc.shape[0]} 张)", flush=True)
