#!/usr/bin/env python3
import json, sys
import numpy as np, cv2
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import visual_baseline as vb
import e8_train as e8

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
src = sorted((ROOT / "data/images_v12_confusable").glob("*.jpg"))
z = np.load(ROOT / "synthetic/vis_cache/e14_sign/final_probe.npz", allow_pickle=True)
reg = json.load(open(ROOT / "data/registry_v11.json"))
sha = json.load(open(ROOT / "data/sha_file_index.json"))
gal_ids, gal_rows = [], []
for r in reg:
    if r["label"] == "positive" and r.get("is_representative"):
        p = sha.get(r["sha256"])
        if p: gal_rows.append(str(ROOT / "data" / p))
rf, _ = e8.embed_all([{"image_id": f"g{i}", "path": p, "label": "positive"} for i, p in enumerate(gal_rows)], "v4_real")
gal_clip = e8.l2(rf["clip_full"])
dino_ref = e8.l2(np.vstack([rf["dinov2_full"], rf["dinov2_crop"]]))
ec, ed = vb.Embedder("clip"), vb.Embedder("dinov2")
out = []
for i, fp in enumerate(src):
    try:
        pil = Image.open(fp).convert("RGB")
        bgr = cv2.imread(str(fp))
        c = e8.l2(ec.embed_pil([pil]))
        s_clip = float(np.sort(c @ gal_clip.T, axis=1)[:, -5:].mean())
        dfull = e8.l2(ed.embed_pil([pil]))
        box = vb.sign_crop_box(bgr)
        h, w = bgr.shape[:2]
        cc = bgr[box[1]:box[3], box[0]:box[2]]
        dcrop = e8.l2(ed.embed_pil([Image.fromarray(cv2.cvtColor(cc, cv2.COLOR_BGR2RGB))])) if cc.size else dfull
        Q = np.vstack([dcrop, dfull])
        s_dino = float(np.clip((Q @ dino_ref.T).max() - 0.3, 0, None) / 0.5)
        x = np.concatenate([c, dfull, dcrop], axis=1)
        s_probe = float(1 / (1 + np.exp(-(x @ z["coef"].T + z["intercept"]))[0][0]))
        s = 0.4 * np.clip(s_clip, 0, 1) + 0.4 * s_dino + 0.2 * s_probe
        out.append({"file": fp.name, "s_clip": round(s_clip, 3), "s_dino": round(s_dino, 3),
                    "s_probe": round(s_probe, 3), "fusion": round(float(s), 3)})
    except Exception as e:
        out.append({"file": fp.name, "error": str(e)[:60]})
    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(src)}", flush=True)
json.dump(out, open(ROOT / "data/v12_scores.json", "w"), ensure_ascii=False, indent=1)
s = np.array([o["fusion"] for o in out if "fusion" in o])
print(f"n={len(s)} mean={s.mean():.3f} >0.636: {(s>0.636).sum()} ({(s>0.636).mean():.1%})  >0.80: {(s>0.80).sum()} ({(s>0.80).mean():.1%})", flush=True)
