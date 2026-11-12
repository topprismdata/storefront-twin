#!/usr/bin/env python3
"""
G — 独立泛化集构建（GPT 终审清单 D 项）
========================================
来源(与 330/池/金标注 sha 级零重叠):
  P: gen_full A-POI 未判兄弟图(同店新照片,门店组天然隔离)
  N: gen_full C-POI 未判兄弟图(同名异业/他牌)
  U: images/ 中 c0/seed 未注册图(标注后归类)
流程: 抽样 -> sha 排除 -> 接触表 -> 双通道标注(本地+GPT) -> 分歧全分辨率仲裁 -> registry_gen.json
"""
import json, os, random, hashlib
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
DATA = ROOT / "data"
SEED = 2026


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def poi_verdicts():
    poi2v = {}
    for line in open(DATA / "gen_verify/verdicts.jsonl"):
        for k, v in json.loads(line)["verdicts"].items():
            poi2v[int(k)] = v
    gi = json.load(open(DATA / "gen_verify/grid_index.json"))
    return {e["poi_id"]: poi2v[e["grid_id"]] for e in gi}


def main():
    rng = random.Random(SEED)
    vs1 = poi_verdicts()
    gf = json.load(open(DATA / "gen_full/gen_candidates.json"))
    prov = {r["poi_id"]: r.get("province", "") for r in gf}
    sib = defaultdict(list)
    for r in gf:
        v = vs1.get(r["poi_id"])
        if v not in ("A", "C"):
            continue
        fn = r["file"] if r["file"].startswith("gen_") else f"gen_{r['file']}"
        fp = DATA / "gen_full" / fn
        if fp.exists():
            sib[(v, r["poi_id"])].append(str(fp))

    # registry sha 排除
    sha_idx = json.load(open(DATA / "sha_file_index.json"))
    reg_shas = {x["sha256"] for x in json.load(open(DATA / "registry_v3.json"))}
    reg_shas |= {x["sha256"] for x in json.load(open(DATA / "registry_v0.json"))}
    reg_files = {str((DATA / sha_idx[s]).resolve()) for s in reg_shas if s in sha_idx}

    def take(kind_v, n_stores, per_store):
        keys = [k for k in sib if k[0] == kind_v]
        rng.shuffle(keys)
        out, stores = [], []
        for k in keys:
            files = [f for f in sib[k] if f not in reg_files]
            rng.shuffle(files)
            if not files:
                continue
            stores.append((k, files[:per_store]))
            out.extend(files[:per_store])
            if len(stores) >= n_stores:
                break
        return stores, out

    a_stores, a_files = take("A", 50, 2)
    c_stores, c_files = take("C", 60, 2)
    cand = json.load(open(DATA / "gen_set_candidates.json"))["cand"]
    cand = [c for c in cand if str(Path(c).resolve()) not in reg_files]
    rng.shuffle(cand)
    u_files = cand[:40]

    items = []
    for f in a_files:
        items.append({"path": f, "src": "pos_sib", "poi": None, "exp": "A"})
    for (v, poi), fs in a_stores:
        for f in fs:
            for it in items:
                if it["path"] == f:
                    it["poi"] = f"pos_{poi}"
                    it["province"] = prov.get(poi, "")
    for f in c_files:
        items.append({"path": f, "src": "neg_sib", "poi": None, "exp": "C"})
    for (v, poi), fs in c_stores:
        for f in fs:
            for it in items:
                if it["path"] == f:
                    it["poi"] = f"neg_{poi}"
                    it["province"] = prov.get(poi, "")
    for f in u_files:
        items.append({"path": str(f), "src": "uncovered", "poi": "u_" + Path(f).stem, "exp": "?"})

    # sha 去重(集合内部)
    seen, dedup = {}, []
    for it in items:
        s = sha256(it["path"])
        if s in seen:
            continue
        seen[s] = True
        it["sha256"] = s
        dedup.append(it)
    rng.shuffle(dedup)
    for i, it in enumerate(dedup, 1):
        it["gid"] = f"G{i:04d}"

    # 接触表: 20 格 4x5, 420px
    CELL, COLS, ROWS = 420, 4, 5
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    except Exception:
        font = ImageFont.load_default()
    sheets = []
    for si in range(0, len(dedup), COLS * ROWS):
        chunk = dedup[si:si + COLS * ROWS]
        W, H = COLS * CELL, ROWS * CELL
        sheet = Image.new("RGB", (W, H), (240, 240, 240))
        dr = ImageDraw.Draw(sheet)
        for j, it in enumerate(chunk):
            r, c = divmod(j, COLS)
            im = Image.open(it["path"]).convert("RGB")
            im.thumbnail((CELL, CELL), Image.LANCZOS)
            x0, y0 = c * CELL, r * CELL
            sheet.paste(im, (x0 + (CELL - im.width) // 2, y0 + (CELL - im.height) // 2))
            dr.rectangle([x0, y0, x0 + 150, y0 + 44], fill=(200, 16, 26))
            dr.text((x0 + 8, y0 + 3), it["gid"], fill=(255, 255, 255), font=font)
        p = DATA / "annotation_pack/gen_sheets" / f"gsheet_{si // (COLS * ROWS):02d}.jpg"
        p.parent.mkdir(exist_ok=True, parents=True)
        sheet.save(p, quality=88)
        sheets.append(str(p))

    man = [{"gid": it["gid"], "path": it["path"], "src": it["src"], "poi": it["poi"],
            "province": it.get("province", ""), "sha256": it["sha256"]} for it in dedup]
    json.dump(man, open(DATA / "annotation_pack/gen_manifest.json", "w"), ensure_ascii=False, indent=1)
    json.dump(sheets, open(DATA / "annotation_pack/gen_sheets/list.json", "w"), ensure_ascii=False)
    from collections import Counter
    print("泛化集候选:", len(dedup), "张 |", Counter(m["src"] for m in man))
    print("组数(pos/neg/u):", len({m["poi"] for m in man if m["src"] == "pos_sib"}),
          len({m["poi"] for m in man if m["src"] == "neg_sib"}),
          len({m["poi"] for m in man if m["src"] == "uncovered"}))
    print("表:", len(sheets), "张 ->", sheets[0] if sheets else "-")


if __name__ == "__main__":
    main()
