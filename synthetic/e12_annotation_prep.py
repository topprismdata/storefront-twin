#!/usr/bin/env python3
"""
E12 — 金标准标注包生成（分层 v2：省份 × 捕获条件簇 + VI 变体旗标）
================================================================
分配：非浙 A 类普查(344) + 浙江分层抽样(300) + VI 变体普查(10) ≈ 654 图
产物：data/annotation_pack/{sheets/*.jpg, anno_manifest.json, verdict_template.jsonl, GUIDE.md}
"""
import json, sys, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import e6_real_probe as e1
import e6_real_probe_v2 as e2
from sklearn.cluster import KMeans

E6 = e1.E6
DATA = e1.DATA
OUT = DATA / "annotation_pack"


def load_real_A():
    rows = [r for r in e2.build_pool_v2() if r["lab"] == "A" and r["src"] != "render3d"]
    m = {}
    for n in ("clip", "dinov2"):
        for v in ("full", "crop"):
            for tag in ("pool", "x"):
                f = E6 / f"{tag}_{n}_{v}.npz"
                if f.exists():
                    z = np.load(f, allow_pickle=True)
                    m[(n, v)] = m.get((n, v), {})
                    m[(n, v)].update(zip(z["ids"].tolist(), z["vecs"]))
    keep = [r for r in rows if r["image_id"] in m[("clip", "full")]
            and r["image_id"] in m[("dinov2", "full")]]
    X = np.concatenate([np.stack([m[("clip", "full")][r["image_id"]] for r in keep]),
                        np.stack([m[("dinov2", "full")][r["image_id"]] for r in keep])], 1).astype(np.float32)
    c = X[:, :512] / np.linalg.norm(X[:, :512], axis=1, keepdims=True)
    d = X[:, 512:] / np.linalg.norm(X[:, 512:], axis=1, keepdims=True)
    F = np.concatenate([c, d], 1)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    return keep, F


def main():
    keep, F = load_real_A()
    km = KMeans(6, n_init=10, random_state=2026).fit_predict(F)
    unc_list = json.load(open(E6 / "uncovered.json"))
    unc = {u["image_id"] for u in unc_list}
    # E11 目检定名的 VI 变体子集（uncovered_sheet 编号 0 基位置）
    vi_pos = {7, 30, 32, 44, 46, 50, 51, 57, 58, 59}
    VI = {unc_list[i]["image_id"] for i in vi_pos if i < len(unc_list)}
    for r, k in zip(keep, km):
        r["cap_cluster"] = int(k)
        r["vi_variant"] = r["image_id"] in VI
    # 分配
    zj = [r for r in keep if r["province"] == "浙江省"]
    nonzj = [r for r in keep if r["province"] != "浙江省"]
    rng = random.Random(2026)
    zj_by_cap = defaultdict(list)
    for r in zj:
        zj_by_cap[r["cap_cluster"]].append(r)
    zj_sample = []
    for k, lst in sorted(zj_by_cap.items()):
        take = round(300 * len(lst) / len(zj))
        zj_sample.extend(rng.sample(lst, min(take, len(lst))))
    vi = [r for r in keep if r["vi_variant"]]
    selected = {r["image_id"]: r for r in nonzj + zj_sample + vi}
    sel = list(selected.values())
    print(f"选择: 总{len(sel)} = 非浙普查{len(nonzj)} + 浙抽{len(zj_sample)} + VI旗标{len(vi)}(部分重叠)")
    # 接触表 20/张
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "sheets").mkdir(exist_ok=True)
    import cv2
    cell_w, cell_h, per = 420, 340, 20
    nsheet = (len(sel) + per - 1) // per
    anno_manifest = []
    for s in range(nsheet):
        chunk = sel[s * per:(s + 1) * per]
        cols, rows_n = 4, 5
        sheet = np.full((rows_n * (cell_h + 34) + 12, cols * (cell_w + 8) + 8, 3), 255, np.uint8)
        for i, r in enumerate(chunk):
            anno_id = f"A{s * per + i + 1:04d}"
            img = cv2.imread(r["path"])
            if img is None:
                continue
            h, w = img.shape[:2]
            sc = min(cell_w / w, cell_h / h)
            img = cv2.resize(img, (int(w * sc), int(h * sc)))
            rr, cc = divmod(i, cols)
            y = 12 + rr * (cell_h + 34)
            x = cc * (cell_w + 8) + 8
            sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
            pc = {"浙江省": "ZJ", "江西省": "JX", "福建省": "FJ", "广东省": "GD", "湖南省": "HN",
                  "河南省": "HA", "四川省": "SC", "湖北省": "HB", "安徽省": "AH", "广西壮族自治区": "GX",
                  "贵州省": "GZ", "云南省": "YN"}.get(r["province"], "XX")
            cv2.putText(sheet, f"{anno_id} {pc} K{r['cap_cluster']}"
                        + ("*VI" if r["vi_variant"] else ""),
                        (x + 4, y + cell_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            anno_manifest.append({"anno_id": anno_id, "image_id": r["image_id"], "path": r["path"],
                                  "province": r["province"], "cap_cluster": r["cap_cluster"],
                                  "vi_variant": r["vi_variant"], "sheet": s})
        cv2.imwrite(str(OUT / "sheets" / f"anno_{s:02d}.jpg"), sheet)
    json.dump(anno_manifest, open(OUT / "anno_manifest.json", "w"), ensure_ascii=False, indent=1)
    with open(OUT / "verdict_template.jsonl", "w") as f:
        for a in anno_manifest:
            f.write(json.dumps({"anno_id": a["anno_id"], "verdict": "", "sign_text": "",
                                "notes": ""}, ensure_ascii=False) + "\n")
    # 设计文档
    plan = {
        "date": "2026-11-12", "n_selected": len(sel), "n_sheets": nsheet,
        "stratification": "province x cap_cluster(kMeans k=6, 审计实证的捕获条件轴) + vi_variant 旗标",
        "allocation": {"non_zhejiang_census": len(nonzj), "zhejiang_sample": len(zj_sample),
                       "vi_variant_flagged": len(vi)},
        "zhejiang_by_cap": {str(k): round(300 * len(v) / len(zj)) for k, v in sorted(zj_by_cap.items())},
        "prov_dist": dict(Counter(a["province"] for a in anno_manifest)),
        "rubric": "A=真左邻右舍门头 B=室内/非门头 C=同名异业 D=不清 + style_variant 旗标",
        "note": "替换原 CS 风格分层(E11 审计: CS 轴未获真实分布验证)"}
    json.dump(plan, open(DATA / "gen_verify/stratified_annotation_design_v2.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"表数: {nsheet} | 设计文档+清单+模板已写")


if __name__ == "__main__":
    main()
