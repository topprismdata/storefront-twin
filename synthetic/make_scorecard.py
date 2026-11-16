#!/usr/bin/env python3
"""双成绩单生成器(GPT 条件④固化): 读最新 replay json → 输出双成绩单 markdown。
用法: python3 make_scorecard.py [replay.json] (默认取 vis_cache 下最新 *_replay.json)"""
import glob, json, os, sys
import numpy as np

ROOT = "/Users/mac/chatgpt-harvest/6af41562"
cands = sorted(glob.glob(f"{ROOT}/synthetic/vis_cache/*_replay.json"), key=os.path.getmtime)
path = sys.argv[1] if len(sys.argv) > 1 else cands[-1]
out = json.load(open(path))

def fok(r):
    p = r.get("pred", "")
    if not p.startswith("非目录·"):
        return False
    d = p[4:]
    return d in r.get("truth", "") or r.get("truth", "") in d

def pick(*keys):
    for k in keys:
        if k in out:
            return out[k]
    return []

v18 = pick("v18")
cat = pick("cat")
zuo = pick("zuo")
lines = [f"# 双成绩单(数据: {os.path.basename(path)})\n"]
if v18:
    fr = sum(1 for r in v18 if fok(r))
    unk = sum(1 for r in v18 if r["pred"] == "unknown")
    wr = sum(1 for r in v18 if not fok(r) and r["pred"] != "unknown")
    hold = [r for r in v18 if r["truth"] in ("瑞幸咖啡", "库迪咖啡")]
    h = sum(1 for r in hold if fok(r) or r["pred"] == "unknown")
    lines += ["## 决策诚实性成绩单\n",
              f"| 目录外诚实率 | {fr + unk}/{len(v18)} ({(fr + unk)/len(v18)*100:.1f}%) |",
              f"| 错进目录 | {wr}/{len(v18)} ({wr/len(v18)*100:.1f}%) |",
              f"| 未见品牌留出拒答(瑞幸/库迪) | {h}/{len(hold)} ({h/len(hold)*100:.1f}%) |\n"]
if cat:
    ok = sum(1 for r in cat if r["pred"] == r["truth"])
    lines += ["## 识别能力成绩单(E25 协议可比)\n",
              f"| 目录品牌准确率 | {ok}/{len(cat)} ({ok/len(cat)*100:.1f}%) |"]
if zuo:
    okz = sum(1 for r in zuo if r["pred"] == r["truth"])
    lines += [f"| 左邻右舍(专场品牌) | {okz}/{len(zuo)} ({okz/len(zuo)*100:.1f}%) |\n"]
vef = f"{ROOT}/synthetic/vis_cache/verify_expanded.json"
if os.path.exists(vef):
    v = json.load(open(vef))
    g = v["genuine"]; im = v["impostor"]
    tar = sum(1 for x in g if x["verdict"] == "VERIFIED")
    far = sum(1 for x in im if x["verdict"] == "VERIFIED")
    ztar = sum(1 for x in g if x["truth"] == "左邻右舍" and x["verdict"] == "VERIFIED")
    ztot = sum(1 for x in g if x["truth"] == "左邻右舍")
    lines += ["## 验证模式成绩单(1:1)\n",
              f"| 冒充 FAR | {far}/{len(im)} ({far/max(len(im),1)*100:.1f}%) |",
              f"| 真实宣称 TAR | {tar}/{len(g)} ({tar/max(len(g),1)*100:.1f}%) |",
              f"| 左邻右舍 TAR | {ztar}/{ztot} |\n"]
print("\n".join(lines))
