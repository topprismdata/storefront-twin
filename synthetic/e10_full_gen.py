#!/usr/bin/env python3
"""E10 全量盲测材料生成：330 张匿名分批网格 + 真值密钥（密钥不发给 VLM）。
每批 24 图（两张 3×4 网格 g1/g2，512px/图），共 14 批（最后一批 18 图）。
全局乱序 + 匿名编号 B{batch:02d}_{idx:02d}。产物 vis_cache/e10_full/。"""
import json, os
import numpy as np
import cv2
import visual_baseline as vb

ROOT = "/Users/mac/chatgpt-harvest/6af41562"
OUT = ROOT + "/synthetic/vis_cache/e10_full"
os.makedirs(OUT, exist_ok=True)
rows = vb.load_registry()
rng = np.random.default_rng(2026)
order = list(range(len(rows)))
rng.shuffle(order)

BATCH = 24
batches = [order[i:i+BATCH] for i in range(0, len(order), BATCH)]
key = {}          # anon_id -> {image_id, path, label, neg_type, group, e8g_score, e8g_pred}
z = np.load(ROOT + "/synthetic/vis_cache/e8g_frozen_scores.npz", allow_pickle=True)
sc = {iid: float(s) for iid, s in zip(z["ids"], z["scores"])}
lab = {iid: int(l) for iid, l in zip(z["ids"], z["labels"])}

def make_cell(img, tag):
    if img is None:
        img = np.zeros((384, 512, 3), np.uint8)
    ih, iw = img.shape[:2]
    nw = 512; nh = int(ih * nw / iw)
    img = cv2.resize(img, (nw, min(nh, 420)))
    canvas = np.full((450, 512, 3), 40, np.uint8)
    canvas[:img.shape[0], :] = img
    cv2.rectangle(canvas, (0, 0), (512, 30), (0, 0, 0), -1)
    cv2.putText(canvas, tag, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return canvas

manifest = []
for bi, batch in enumerate(batches):
    cells = []
    for j, ri in enumerate(batch):
        r = rows[ri]
        anon = f"B{bi+1:02d}_{j+1:02d}"
        img = cv2.imread(r["path"])
        cells.append(make_cell(img, anon))
        key[anon] = dict(image_id=r["image_id"], path=r["path"], label=r["label"],
                         neg_type=r.get("neg_type", ""), group=r["group"],
                         e8g_score=round(sc.get(r["image_id"], 0), 3),
                         e8g_pred=bool(sc.get(r["image_id"], 0) > 0.636))
    # 拆两张 3×4（或末批不足）
    half = (len(cells) + 1) // 2
    g1cells = cells[:half]; g2cells = cells[half:]
    def grid(cs):
        if not cs:
            return None
        ncol = 4
        nrow = (len(cs) + ncol - 1) // ncol
        while len(cs) < nrow * ncol:
            cs.append(np.full((450, 512, 3), 40, np.uint8))
        return np.vstack([np.hstack(cs[r*ncol:(r+1)*ncol]) for r in range(nrow)])
    g1 = grid(g1cells); g2 = grid(g2cells)
    p1 = f"{OUT}/batch{bi+1:02d}_g1.jpg"
    cv2.imwrite(p1, g1)
    entry = dict(batch=bi+1, n=len(batch), g1=p1,
                 anon_first=f"B{bi+1:02d}_01", anon_last=f"B{bi+1:02d}_{len(batch):02d}")
    if g2 is not None:
        p2 = f"{OUT}/batch{bi+1:02d}_g2.jpg"
        cv2.imwrite(p2, g2)
        entry["g2"] = p2
    manifest.append(entry)

json.dump(key, open(f"{OUT}/truth_key.json", "w"), ensure_ascii=False, indent=1)
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1)
print(f"batches: {len(batches)}, total images: {len(order)}")
print(f"key entries: {len(key)}")
for m in manifest:
    print(f"  batch{m['batch']:02d}: n={m['n']} g1={'g2' in m and '2 sheets' or '1 sheet'}")