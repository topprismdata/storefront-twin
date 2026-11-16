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
    # 评估隔离门禁#4: 阈值校准门店行与留出门店行分开报(无偏估计以留出为准)
    try:
        import numpy as _np
        from collections import defaultdict
        _z = _np.load(f"{ROOT}/synthetic/vis_cache/e25_feats.npz", allow_pickle=True)
        _rows = list(_z["rows"])
        _brands = sorted(set(r["brand"] for r in _rows))
        _rng = _np.random.RandomState(0)
        _gidx = {}
        for _b in _brands:
            _bi = [i for i, r in enumerate(_rows) if r["brand"] == _b]
            _st = sorted(set(_rows[i]["poi"] for i in _bi))
            _rng.shuffle(_st)
            _gs = set(_st[:max(1, int(len(_st) * 0.7))])
            _gidx[_b] = [i for i in _bi if _rows[i]["poi"] in _gs]
        _vs = defaultdict(list)
        for _i in range(len(_rows)):
            _bb = _rows[_i]["brand"]
            if _bb != "local" and _i not in set(_gidx[_bb]):
                _vs[_rows[_i]["poi"]].append(_i)
        _cal = set()
        _r2 = _np.random.RandomState(2)
        for _b in sorted(_brands):
            if _b == "local":
                continue
            _rb = [i for s in _vs for i in _vs[s] if _rows[i]["brand"] == _b]
            if len(_rb) < 4:
                continue
            _ss = sorted(set(_rows[i]["poi"] for i in _rb))
            _r2.shuffle(_ss)
            for s in _ss[:max(1, len(_ss) // 2)]:
                _cal.add(s)
        import random as _rd
        _sampler = _rd.Random(11)
        _sampled = _sampler.sample([r for r in _rows if r["brand"] not in ("local", "左邻右舍")], len(cat))
        _oc = _of = _nc = _nf = 0
        for r, x in zip(cat, _sampled):
            _hit = r["pred"] == x["brand"]
            if x["poi"] in _cal:
                _nc += 1; _oc += _hit
            else:
                _nf += 1; _of += _hit
        lines += [f"| — 校准门店行 | {_oc}/{_nc} ({_oc/max(_nc,1)*100:.1f}%) |",
                  f"| — 留出门店行(无偏) | {_of}/{_nf} ({_of/max(_nf,1)*100:.1f}%) |"]
    except Exception as e:
        lines += [f"| — 拆分失败: {e} |"]
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
