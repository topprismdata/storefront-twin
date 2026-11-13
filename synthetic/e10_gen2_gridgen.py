#!/usr/bin/env python3
"""E10 第二批独立泛化盲测网格生成 (e10_gen2): v10 赣闽 82 张新正例 + 18 张分层负例。
7 批 (6×15 + 1×10), 批内匿名号 K01.., seed 固定。冻结 v2 指令 (e10_prompt_v2_sealed.md)。
输出 vis_cache/e10_gen2/。"""
import json, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
OUT = ROOT / "synthetic/vis_cache/e10_gen2"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

POS_RANGE = range(774, 856)  # img_0774..img_0855 = v10 赣闽鲁 82 新正例
NEG_PLAN = [("same_name_unrelated", 5), ("confusable_brand", 5), ("render_stock", 3),
            ("indoor", 2), ("logo", 1), ("competitor", 1), ("uncertain", 1)]


def main():
    d = np.load(ROOT / "synthetic/vis_cache/e8g_v11_scores.npz", allow_pickle=True)
    e8g = {iid: {"score": float(sc), "pred": bool(sc > 0.636)} for iid, sc in zip(list(d["ids"]), list(d["scores"]))}

    reg = {r["image_id"]: r for r in json.load(open(ROOT / "data/registry_v11.json"))}
    sha_idx = json.load(open(ROOT / "data/sha_file_index.json"))

    def path_of(iid):
        return ROOT / "data" / sha_idx[reg[iid]["sha256"]]

    pos_ids = [f"img_{i:04d}" for i in POS_RANGE if f"img_{i:04d}" in reg]
    assert len(pos_ids) == 82, len(pos_ids)
    assert all(reg[i]["evidence_level"] == "dual_channel_arbitrated" for i in pos_ids)

    negs_by_type = {}
    for iid, r in reg.items():
        if r["label"] == "negative" and r.get("is_representative") and r.get("neg_type"):
            negs_by_type.setdefault(r["neg_type"], []).append(iid)
    # 负例只取 v11 池中不属于 v10 正例来源轮的 (两湖+安徽+历史), 与正例(赣闽)店组天然隔离
    neg_ids = []
    for t, k in NEG_PLAN:
        pool = sorted(set(negs_by_type.get(t, [])))
        random.Random(2026).shuffle(pool)
        neg_ids += pool[:k]
    assert len(neg_ids) == 18, len(neg_ids)

    all_ids = pos_ids + neg_ids
    random.Random(20261113).shuffle(all_ids)
    sizes = [15] * 6 + [10]
    batches, cur = [], []
    for iid in all_ids:
        cur.append(iid)
        if len(cur) == sizes[len(batches)]:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)
    assert len(batches) == 7 and sum(len(b) for b in batches) == 100

    fc = lambda s: ImageFont.truetype(FONT, s)
    CW, CH, BADGE = 620, 520, 44
    manifest, key = [], {}
    for b, batch in enumerate(batches, 1):
        cols, rows_n = 5, 3
        sheet = Image.new("RGB", (CW * cols, CH * rows_n), (70, 74, 82))
        dr = ImageDraw.Draw(sheet)
        key[str(b)] = {}
        for ci, iid in enumerate(batch):
            badge = f"K{ci + 1:02d}"
            r = reg[iid]
            p = path_of(iid)
            im = Image.open(p).convert("RGB")
            im.thumbnail((CW, CH - BADGE))
            x0, y0 = (ci % cols) * CW, (ci // cols) * CH
            sheet.paste(im, (x0 + (CW - im.width) // 2, y0 + BADGE + (CH - BADGE - im.height) // 2))
            dr.rectangle([x0, y0, x0 + 118, y0 + BADGE], fill=(200, 16, 18))
            dr.text((x0 + 10, y0 + 8), badge, font=fc(28), fill=(255, 255, 255))
            dr.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1], outline=(40, 42, 46), width=2)
            key[str(b)][badge] = {"image_id": iid, "label": r["label"], "neg_type": r.get("neg_type", ""),
                                  "store_id": r.get("store_id", ""), "city": r.get("geo_bucket", ""),
                                  "e8g_score": e8g.get(iid, {}).get("score"), "e8g_pred": e8g.get(iid, {}).get("pred"),
                                  "file": str(p)}
        fp = OUT / f"grid_b{b:02d}.jpg"
        sheet.save(fp, quality=88)
        manifest.append({"batch": b, "grid": str(fp), "n": len(batch)})
        print(f"batch {b}: {fp.name} ({len(batch)} 格)")

    json.dump(manifest, open(OUT / "manifest.json", "w"), ensure_ascii=False, indent=1)
    json.dump(key, open(OUT / "truth_key.json", "w"), ensure_ascii=False, indent=1)
    npos = sum(1 for b in key.values() for v in b.values() if v["label"] == "positive")
    print(f"key: 100 格 ({npos} 正 / {100 - npos} 负)")


if __name__ == "__main__":
    main()
