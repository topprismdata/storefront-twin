#!/usr/bin/env python3
"""E10 新正例盲测网格生成: v5+v6 的 30 张新正例 + 15 张分层负例对照。
3 批 × 15 格, 全局匿名号 N01..N45 (批内 N01..N15), seed 固定。输出 vis_cache/e10_newpos/。"""
import json, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
OUT = ROOT / "synthetic/vis_cache/e10_newpos"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

NEW_POS = [f"img_{i:04d}" for i in range(601, 631)]  # 30 张 v5+v6 新正例
NEG_PLAN = [("render_stock", 6), ("competitor", 4), ("same_name_unrelated", 3), ("indoor", 1), ("logo", 1)]


def main():
    d = np.load(ROOT / "synthetic/vis_cache/e8g_v6_scores.npz", allow_pickle=True)
    ids_npz, scores_npz = list(d["ids"]), list(d["scores"])
    e8g = {iid: {"score": float(sc), "pred": bool(sc > 0.636)} for iid, sc in zip(ids_npz, scores_npz)}

    reg = {r["image_id"]: r for r in json.load(open(ROOT / "data/registry_v6.json"))}
    sha_idx = json.load(open(ROOT / "data/sha_file_index.json"))

    def path_of(iid):
        return ROOT / "data" / sha_idx[reg[iid]["sha256"]]

    pos_ids = [i for i in NEW_POS if i in reg and reg[i].get("is_representative") and reg[i]["label"] == "positive"]
    assert len(pos_ids) == 30, pos_ids

    negs_by_type = {}
    for iid, r in reg.items():
        if r["label"] == "negative" and r.get("is_representative") and r.get("neg_type"):
            negs_by_type.setdefault(r["neg_type"], []).append(iid)
    neg_ids = []
    for t, k in NEG_PLAN:
        pool = sorted(negs_by_type.get(t, []))
        assert len(pool) >= k, (t, len(pool))
        random.Random(2026).shuffle(pool)
        neg_ids += pool[:k]
    assert len(neg_ids) == 15

    all_ids = pos_ids + neg_ids
    random.Random(20261113).shuffle(all_ids)
    batches = [all_ids[i * 15:(i + 1) * 15] for i in range(3)]

    fc = lambda s: ImageFont.truetype(FONT, s)
    CW, CH, BADGE = 620, 520, 44
    manifest, key = [], {"1": {}, "2": {}, "3": {}}
    for b, batch in enumerate(batches, 1):
        cols, rows_n = 5, 3
        sheet = Image.new("RGB", (CW * cols, CH * rows_n), (70, 74, 82))
        dr = ImageDraw.Draw(sheet)
        for ci, iid in enumerate(batch):
            badge = f"N{ci + 1:02d}"
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
        print(f"batch {b}: {fp} ({len(batch)} 格)")

    json.dump(manifest, open(OUT / "manifest.json", "w"), ensure_ascii=False, indent=1)
    json.dump(key, open(OUT / "truth_key.json", "w"), ensure_ascii=False, indent=1)
    npos = sum(1 for b in key.values() for v in b.values() if v["label"] == "positive")
    tot = sum(len(b) for b in key.values())
    print(f"key: {tot} 格 ({npos} 正 / {tot - npos} 负)")


if __name__ == "__main__":
    main()
