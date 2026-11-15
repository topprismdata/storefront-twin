#!/usr/bin/env python3
"""v16 — 薄画廊品牌第二城市扩画廊 (E22 教训: 跨城补画廊立竿见影)。
天虹/好德/新佳宜/丹尼斯/喜士多/苏宁小店/悦来悦喜/谊品/零食有鸣/TODAY/利群/好特卖/快客
等 E22 弱品牌 + 黄三兄弟第三城。额度耗尽自动停。"""
import hashlib, json, time
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
K = "7cefa505e38404f1a83f8eea34565e6c"
OUT = ROOT / "data/images_v16_milkchain3"
OUT.mkdir(parents=True, exist_ok=True)
LOGF = ROOT / "data/amap_v16_milkchain_log.json"
S = requests.Session()
S.headers["Referer"] = "https://www.amap.com/"

# (查询词, 城市, 目录品牌名) — 全部是已有目录品牌的第二/第三城
PLAN = [
    ("天虹超市", "东莞市", "天虹"),
    ("好德便利店", "苏州市", "好德"),
    ("新佳宜", "株洲市", "新佳宜"),
    ("丹尼斯全日鲜", "洛阳市", "丹尼斯"),
    ("喜士多", "苏州市", "喜士多"),
    ("苏宁小店", "苏州市", "苏宁小店"),
    ("悦来悦喜", "洛阳市", "悦来悦喜"),
    ("谊品生鲜", "南京市", "谊品生鲜"),
    ("BHG超市", "北京市", "北京华联"),
    ("零食有鸣", "重庆市", "零食有鸣"),
    ("TODAY便利店", "南宁市", "TODAY"),
    ("利群超市", "威海市", "利群"),
    ("好特卖", "北京市", "好特卖"),
    ("快客便利店", "杭州市", "快客"),
    ("万嘉便利店", "宁德市", "万嘉"),
    ("每一天便利店", "郑州市", "每一天"),
    ("见福便利店", "厦门市", "见福"),
    ("天福便利店", "惠州市", "天福"),
    ("盒马鲜生", "杭州市", "盒马鲜生"),
    ("好想来零食", "济南市", "好想来"),
    ("老婆大人", "杭州市", "老婆大人"),
    ("戴永红零食", "株洲市", "戴永红"),
    ("恰货铺子", "株洲市", "恰货铺子"),
    ("良品铺子", "成都市", "良品铺子"),
    ("糖巢", "合肥市", "糖巢"),
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
log = {}
if LOGF.exists():
    log = json.load(open(LOGF))
    for v in log.values():
        seen_sha.add(v["sha256"]); dhashes.append(v["dhash"])
print(f"历史账本: sha {len(seen_sha)}", flush=True)


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
    got = 0
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
            if d.get("infocode") in ("10044", "10021", "10020"):
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
                fn = f"v16_{pid}_{len(log)}.jpg"
                (OUT / fn).write_bytes(r.content)
                seen_sha.add(sha); dhashes.append(h)
                log[url] = {"file": fn, "sha256": sha, "dhash": h, "poi": pid,
                            "poi_name": poi.get("name"), "brand_query": brand,
                            "brand_cat": cat, "city": poi.get("cityname") or city,
                            "addr": poi.get("address", "")}
                n_dl += 1
                got += 1
            except Exception:
                continue
            time.sleep(0.1)
        if quota_hit:
            break
        time.sleep(0.1)
    json.dump(log, open(LOGF, "w"), ensure_ascii=False, indent=1)
    print(f"  -> {got} 张 (累计 {len(log)})", flush=True)
    if quota_hit:
        print("额度耗尽, 停止", flush=True)
        break
    time.sleep(0.2)
json.dump(log, open(LOGF, "w"), ensure_ascii=False, indent=1)
print(f"完成: 下载 {n_dl}, 去重 {n_dup}, 总 {len(log)}", flush=True)
