#!/usr/bin/env python3
"""v15 — 液奶连锁表大面积扩展采集, 跑到 amap 日额度耗尽自动停。
优先级: 黄色零食三兄弟第二城市(扩画廊) → 全国名连锁(便利店/零食/卖场/超市/折扣)。
每品牌落盘一次日志; infocode 10044(日配额超限)即停。"""
import hashlib, json, time
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
K = "7cefa505e38404f1a83f8eea34565e6c"
OUT = ROOT / "data/images_v15_milkchain2"
OUT.mkdir(parents=True, exist_ok=True)
LOGF = ROOT / "data/amap_v15_milkchain_log.json"
S = requests.Session()
S.headers["Referer"] = "https://www.amap.com/"

# (品牌查询词, 城市, 目录品牌名)  目录名=canon 用; None=待核验后定
PLAN = [
    # 黄三兄弟第二城市(画廊扩展)
    ("零食很忙", "武汉市", "零食很忙"),
    ("赵一鸣零食", "上饶市", "赵一鸣零食"),
    ("好想来零食", "石家庄市", "好想来"),
    # 便利店
    ("天福便利店", "东莞市", "天福"),
    ("苏宁小店", "南京市", "苏宁小店"),
    ("TODAY便利店", "武汉市", "TODAY"),
    ("喜士多", "上海市", "喜士多"),
    ("快客便利店", "上海市", "快客"),
    ("万嘉便利店", "福州市", "万嘉"),
    ("新佳宜", "长沙市", "新佳宜"),
    ("悦来悦喜", "郑州市", "悦来悦喜"),
    ("8字便利店", "北京市", "8字"),
    ("好德便利店", "上海市", "好德"),
    # 零食
    ("良品铺子", "武汉市", "良品铺子"),
    ("老婆大人", "宁波市", "老婆大人"),
    ("糖巢零食", "合肥市", "糖巢"),
    ("戴永红", "长沙市", "戴永红"),
    ("零食有鸣", "成都市", "零食有鸣"),
    ("恰货铺子", "长沙市", "恰货铺子"),
    # 卖场/超市/折扣
    ("华润万家", "深圳市", "华润万家"),
    ("利群超市", "青岛市", "利群"),
    ("山姆会员店", "上海市", "山姆"),
    ("开市客", "上海市", "开市客"),
    ("天虹超市", "深圳市", "天虹"),
    ("京客隆", "北京市", "京客隆"),
    ("丹尼斯便利店", "郑州市", "丹尼斯"),
    ("北京华联", "北京市", "北京华联"),
    ("盒马NB", "上海市", "盒马NB"),
    ("谊品生鲜", "合肥市", "谊品生鲜"),
]

seen_sha, dhashes = set(), []
for lg in sorted((ROOT / "data").glob("amap_*log*.json")):
    try:
        old = json.load(open(lg))
        items = old.values() if isinstance(old, dict) else old
        for v in items:
            if isinstance(v, dict) and v.get("sha256"):
                seen_sha.add(v["sha256"])
            if isinstance(v, dict) and v.get("dhash"):
                dhashes.append(v["dhash"])
    except Exception:
        pass
print(f"历史账本: sha {len(seen_sha)}, dhash {len(dhashes)}", flush=True)
log = {}
if LOGF.exists():
    log = json.load(open(LOGF))
    for v in log.values():
        seen_sha.add(v["sha256"]); dhashes.append(v["dhash"])


def dh(im):
    g = im.convert("L").resize((9, 8))
    px = list(g.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if px[row * 9 + col] > px[row * 9 + col + 1] else 0)
    return bits


def ham(a, b):
    return bin(a ^ b).count("1")


n_dl = n_dup = 0
quota_hit = False
for brand, city, cat in PLAN:
    if any(v["brand_query"] == brand and v["city"] == city for v in log.values()):
        print(f"{brand} {city}: 已采过, 跳过", flush=True)
        continue
    try:
        j = requests.get("https://restapi.amap.com/v5/place/text",
                         params={"key": K, "keywords": brand, "region": city,
                                 "city_limit": "true", "page_size": 15, "page_num": 1},
                         timeout=15).json()
    except Exception as e:
        print(brand, city, "ERR", e, flush=True)
        continue
    if j.get("status") != "1":
        ic = j.get("infocode")
        print(f"{brand} {city}: API异常 {ic} {j.get('info')}", flush=True)
        if ic in ("10044", "10021", "10020"):
            quota_hit = True
            break
        continue
    pois = j.get("pois", [])
    print(f"{brand} {city}: {len(pois)} POI", flush=True)
    got_here = 0
    for p in pois[:12]:
        pid = p.get("id")
        if not pid:
            continue
        try:
            d = requests.get("https://restapi.amap.com/v3/place/detail",
                             params={"key": K, "id": pid}, timeout=15).json()
        except Exception:
            continue
        if d.get("status") != "1":
            ic = d.get("infocode")
            if ic in ("10044", "10021", "10020"):
                quota_hit = True
                break
            continue
        poi = (d.get("pois") or [None])[0]
        if not poi:
            continue
        for ph in (poi.get("photos") or [])[:6]:
            url = ph.get("url", "")
            if "showpic" not in url and "aos-comment" not in url:
                continue
            try:
                r = S.get(url, timeout=20)
                if r.status_code != 200 or len(r.content) < 8000:
                    continue
                sha = hashlib.sha256(r.content).hexdigest()
                if sha in seen_sha:
                    n_dup += 1
                    continue
                im = Image.open(BytesIO(r.content))
                h = dh(im)
                if any(ham(h, g) <= 8 for g in dhashes[-3000:]):
                    n_dup += 1
                    continue
                fn = f"v15_{pid}_{len(log)}.jpg"
                (OUT / fn).write_bytes(r.content)
                seen_sha.add(sha); dhashes.append(h)
                log[url] = {"file": fn, "sha256": sha, "dhash": h, "poi": pid,
                            "poi_name": poi.get("name"), "brand_query": brand,
                            "brand_cat": cat, "city": poi.get("cityname") or city,
                            "addr": poi.get("address", "")}
                n_dl += 1
                got_here += 1
            except Exception:
                continue
            time.sleep(0.1)
        if quota_hit:
            break
        time.sleep(0.1)
    json.dump(log, open(LOGF, "w"), ensure_ascii=False, indent=1)
    print(f"  -> {got_here} 张 (累计 {len(log)})", flush=True)
    if quota_hit:
        print("额度耗尽, 停止", flush=True)
        break
    time.sleep(0.2)
json.dump(log, open(LOGF, "w"), ensure_ascii=False, indent=1)
print(f"完成: 下载 {n_dl}, 去重丢弃 {n_dup}, 总 {len(log)}", flush=True)
