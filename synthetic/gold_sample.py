#!/usr/bin/env python3
"""P0-A 金标准 400 张边界加权抽样(全库覆盖式)。

设计(2026-11-13, GPT 评审 P0-A 决策):
- 代表语料全采: 330 (30 正 / 300 负), 这是实际 eval 语料, 全量人工覆盖
- 非代表补充 70: 全部 6 张正例兄弟照 + 全部 25 张非素材负例(竞品6/同名5/混淆2/logo2/地图1/室内9)
  + 39 张随机非代表 render_stock
- 素材域(render_stock 295=256 rep+39 nonrep)按现状全采: 目的是给整个 eval 语料人工真值
- 盲标: seed 2026 全局洗牌 → A0001..A0400, 正例完全打散; key 单独放 keys/ 目录
输出: data/annotation_pack_v2/gold/{images/, sheets/, keys/, gold_manifest.csv, SAMPLE_SUMMARY.md}
"""
import json, random, shutil, csv, collections
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "annotation_pack_v2" / "gold"
SEED = 2026

def main():
    random.seed(SEED)
    reg = json.load(open(DATA / "registry_v3.json"))
    sha = json.load(open(DATA / "sha_file_index.json"))
    fz = np.load(ROOT / "synthetic/vis_cache/e8g_frozen_scores.npz", allow_pickle=True)
    score_of = {str(i): float(s) for i, s in zip(fz["ids"], fz["scores"])}

    rows = []
    for r in reg:
        p = sha.get(r.get("sha256"))
        if not p:
            continue
        fp = DATA / p
        if not fp.exists():
            continue
        rows.append({**r, "_rel": p, "_rep": bool(r.get("is_representative")),
                     "_score": score_of.get(r["image_id"])})

    sel, why = [], collections.defaultdict(list)
    def take(rs, tag):
        for r in rs:
            sel.append((r, tag)); why[tag].append(r["image_id"])

    rep = [r for r in rows if r["_rep"]]
    nonrep = [r for r in rows if not r["_rep"]]
    take(rep, "rep_all")                                   # 330 代表语料全采
    take([r for r in nonrep if r["label"] == "positive"], "nonrep_pos")  # 6
    ntag = lambda r: r.get("neg_type", "")
    take([r for r in nonrep if r["label"] == "negative" and ntag(r) != "render_stock"], "nonrep_edge")  # 25
    pool = [r for r in nonrep if r["label"] == "negative" and ntag(r) == "render_stock"]
    take(random.sample(pool, 39), "nonrep_stock_rand")     # 39

    assert len(sel) == 400, f"expect 400, got {len(sel)}"
    random.shuffle(sel)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "keys").mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)

    key, manifest = {}, []
    for i, (r, tag) in enumerate(sel, 1):
        bid = f"A{i:04d}"
        ext = Path(r["_rel"]).suffix or ".jpg"
        shutil.copy2(DATA / r["_rel"], OUT / "images" / f"{bid}{ext}")
        key[bid] = {"image_id": r["image_id"], "label": r["label"], "neg_type": ntag(r),
                    "candidate_style": (r.get("candidate_style") or "").split("_")[0],
                    "store_id": r.get("store_id"), "representative": r["_rep"],
                    "e8g_score": r["_score"], "sample_tag": tag, "rel_path": r["_rel"]}
        manifest.append({"blind_id": bid, "file": f"images/{bid}{ext}",
                         "annotator1": "", "annotator2": "", "notes": ""})

    json.dump(key, open(OUT / "keys" / "gold_key.json", "w"), ensure_ascii=False, indent=1)
    with open(OUT / "gold_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "file", "annotator1", "annotator2", "notes"])
        w.writeheader(); w.writerows(manifest)

    # 接触表: 20 张 × 20 格(4×5), 640px 格, 红底盲标编号(与既有标注表风格一致)
    F = "/System/Library/Fonts/STHeiti Medium.ttc"
    fc = lambda s: ImageFont.truetype(F, s)
    CW, CH, BADGE = 640, 512, 44
    nsheets = (len(manifest) + 19) // 20
    for si in range(nsheets):
        sheet = Image.new("RGB", (CW * 4, CH * 5), (70, 74, 82))
        dr = ImageDraw.Draw(sheet)
        for ci in range(20):
            gi = si * 20 + ci
            if gi >= len(manifest):
                break
            bid = manifest[gi]["blind_id"]
            im = Image.open(OUT / manifest[gi]["file"]).convert("RGB")
            im.thumbnail((CW, CH - BADGE))
            x0, y0 = (ci % 4) * CW, (ci // 4) * CH
            sheet.paste(im, (x0 + (CW - im.width) // 2, y0 + BADGE + (CH - BADGE - im.height) // 2))
            dr.rectangle([x0, y0, x0 + 128, y0 + BADGE], fill=(196, 30, 30))
            dr.text((x0 + 10, y0 + 8), bid, font=fc(28), fill=(255, 255, 255))
            dr.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1], outline=(40, 42, 46), width=2)
        sheet.save(OUT / "sheets" / f"gold_sheet_{si:02d}.jpg", quality=88)

    dist = collections.Counter((k["label"], k["neg_type"] or k["candidate_style"]) for k in key.values())
    repcov = collections.Counter(k["representative"] for k in key.values())
    lines = ["# 金标准 400 抽样(2026-11-13, seed 2026)", "",
             f"- 覆盖: 代表语料 {repcov[True]} / 非代表补充 {repcov[False]}",
             "- 组成:"]
    for (lab, t), n in sorted(dist.items(), key=lambda x: -x[1]):
        lines.append(f"  - {lab}/{t}: {n}")
    lines += ["", "- 盲标: A0001..A0400 全局洗牌, key 在 keys/gold_key.json(仅仲裁员可读)",
              "- 记录表: gold_manifest.csv(两名标注员独立填 annotator1/annotator2)",
              "- 接触表: sheets/gold_sheet_00..19.jpg(每张 20 格)"]
    (OUT / "SAMPLE_SUMMARY.md").write_text("\n".join(lines))
    print(f"gold 400 built: sheets={nsheets}")
    print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in sorted(dist.items(), key=lambda x: -x[1])}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
