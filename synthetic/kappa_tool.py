#!/usr/bin/env python3
"""P0-A 金标准双盲标注一致性工具。

用法:
  python3 kappa_tool.py data/annotation_pack_v2/gold/gold_manifest.csv --key keys/gold_key.json
  # manifest 中 annotator1/annotator2 列需填 M/N/U/O (MATCH/NON_MATCH/UNKNOWN/OUT_OF_SCOPE)

输出:
  - 每对标注员 Cohen's kappa(总体 + 每态)
  - 混淆矩阵 + 分歧清单(arbitration.csv)
  - 多数共识 vs 注册表标签: 一致率 + 每类 P/R(定位注册表疑误标注)
  - 判定: kappa>=0.75 可合并; <0.75 需仲裁后重算
退出码: 0=可合并, 2=需仲裁, 1=输入错误
"""
import csv, json, sys, argparse, collections
from pathlib import Path

STATES = ["M", "N", "U", "O"]
FULL = {"M": "MATCH", "N": "NON_MATCH", "U": "UNKNOWN", "O": "OUT_OF_SCOPE"}

def kappa(a, b):
    n = len(a)
    assert n == len(b) and n > 0
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    exp = sum(ca[s] * cb[s] for s in set(a) | set(b)) / (n * n)
    return (obs - exp) / (1 - exp) if exp < 1 else 1.0, obs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--key", default=None)
    ap.add_argument("--out", default=None, help="输出目录(默认 manifest 同目录)")
    args = ap.parse_args()
    mp = Path(args.manifest)
    outdir = Path(args.out) if args.out else mp.parent
    rows = list(csv.DictReader(open(mp)))
    pairs = [("annotator1", "annotator2")]
    done = []
    for c1, c2 in pairs:
        rec = [(r["blind_id"], r[c1].strip().upper()[:1], r[c2].strip().upper()[:1])
               for r in rows if r[c1].strip() and r[c2].strip()]
        bad = [(i, x, y) for i, x, y in rec if x not in STATES or y not in STATES]
        if bad:
            print(f"非法标注值(须 M/N/U/O): {bad[:8]}"); sys.exit(1)
        if len(rec) < 30:
            print(f"有效双标记录不足: {len(rec)}"); sys.exit(1)
        a = [x for _, x, _ in rec]; b = [y for _, _, y in rec]
        k, obs = kappa(a, b)
        per = {}
        for s in STATES:
            aa = ["M" if x == s else "X" for x in a]
            bb = ["M" if y == s else "X" for y in b]
            ks, _ = kappa(aa, bb)
            per[FULL[s]] = round(ks, 3)
        conf = collections.Counter((x, y) for x, y in zip(a, b))
        done.append({"pair": f"{c1}~{c2}", "n": len(rec), "kappa": round(k, 3), "agree": round(obs, 3),
                     "kappa_per_state": per,
                     "confusion": {f"{x}->{y}": c for (x, y), c in sorted(conf.items()) if x != y},
                     "disagreements": [i for i, x, y in rec if x != y]})
    print(json.dumps(done, ensure_ascii=False, indent=1))
    kk = done[0]["kappa"]
    verdict = "MERGE_OK" if kk >= 0.75 else "NEED_ARBITRATION"
    print(f"verdict: {verdict} (kappa={kk})")

    # 共识 vs 注册表
    keyp = Path(args.key) if args.key else mp.parent / "keys" / "gold_key.json"
    key = json.load(open(keyp)) if keyp.exists() else None
    if key:
        maj, hit = {}, 0
        for r in rows:
            v1, v2 = r["annotator1"].strip().upper()[:1], r["annotator2"].strip().upper()[:1]
            if v1 in STATES and v2 in STATES:
                maj[r["blind_id"]] = v1 if v1 == v2 else None
        tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
        errors = []
        for bid, m in maj.items():
            if m is None:
                continue
            k = key[bid]
            truth = "M" if k["label"] == "positive" else ("N" if k["neg_type"] in ("competitor", "confusable_brand") else "O")
            if m == truth:
                hit += 1
                tp[truth] += 1
            else:
                errors.append({"blind_id": bid, "human": FULL.get(m, m), "registry_expect": FULL.get(truth, truth),
                               "image_id": k["image_id"], "neg_type": k["neg_type"], "style": k["candidate_style"]})
                fp[m] += 1; fn[truth] += 1
        acc = hit / max(1, sum(1 for v in maj.values() if v))
        percls = {FULL[s]: {"P": round(tp[s] / (tp[s] + fp[s]), 3) if tp[s] + fp[s] else None,
                            "R": round(tp[s] / (tp[s] + fn[s]), 3) if tp[s] + fn[s] else None}
                  for s in STATES}
        print(f"consensus vs registry: acc={acc:.3f}  per-class={json.dumps(percls)}")
        with open(outdir / "registry_label_errors.json", "w") as f:
            json.dump({"accuracy": round(acc, 3), "per_class": percls, "errors": errors}, f, ensure_ascii=False, indent=1)
        with open(outdir / "arbitration.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["blind_id", "annotator1", "annotator2", "arbiter", "final"])
            for bid in done[0]["disagreements"]:
                r = next(x for x in rows if x["blind_id"] == bid)
                w.writerow([bid, r["annotator1"], r["annotator2"], "", ""])
        print(f"arbitration.csv written ({len(done[0]['disagreements'])} rows), registry_label_errors.json written")
    sys.exit(0 if verdict == "MERGE_OK" else 2)

if __name__ == "__main__":
    main()
