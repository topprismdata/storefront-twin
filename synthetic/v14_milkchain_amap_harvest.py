#!/usr/bin/env python3
"""v14 — 液奶连锁表新增品牌采集 (amap REST)。
10 新品牌 × 主城: 便利店4(每一天/见福/唐久/红旗连锁) 零食3(零食很忙/赵一鸣/好想来)
卖场2(家家悦/盒马鲜生) 折扣1(好特卖)。v5 text 搜 POI → v3 detail 拉图 → sha256/dhash 全账本去重。
落盘 data/images_v14_milkchain/ + log。仅采集, 标签待核验(pending_review)。"""
import hashlib, json, time
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
K = "7cefa505e38404f1a83f8eea34565e6c"
OUT = ROOT / "data/images_v14_milkchain"
OUT.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers["Referer"] = "https://www.amap.com/"

BRAND_CITY = [
    ("每一天便利店", "西安市"),
    ("见福便利店", "福州市"),
    ("唐久便利店", "太原市"),
    ("红旗连锁", "成都市"),
    ("零食很忙", "长沙市"),
    ("赵一鸣零食", "南昌市"),
    ("好想来零食", "郑州市"),
    ("家家悦", "济南市"),
    ("盒马鲜生", "上海市"),
    ("好特卖", "上海市"),
]

seen_sha, dhashes = set(), []
for lg in ["amap_v11_hbhn_download_log.json", "amap_v12_confusable_log.json",
           "amap_v13_natl_log.json", "amap_v4_download_log.json",
           "amap_v4_variants_download_log.json", "amap_v5_jzs_download_log.json",
           "amap_v6_ah_download_log.json", "amap_v7_jfs_download_log.json"]:
    try:
        old = json.load(open(ROOT / "data" / lg))
        items = old.values() if isinstance(old, dict) else old
        for v in items:
            if isinstance(v, dict) and v.get("sha256"):
                seen_sha.add(v["sha256"])
            if isinstance(v, dict) and v.get("dhash"):
                dhashes.append(v["dhash"])
    except FileNotFoundError:
        pass
print(f"历史账本: sha {len(seen_sha)}, dhash {len(dhashes)}", flush=True)
log = {}


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


n_poi = n_dl = n_dup = 0
quota_hit = False
for brand, city in BRAND_CITY:
    try:
        j = requests.get("https://restapi.amap.com/v5/place/text",
                         params={"key": K, "keywords": brand, "region": city,
                                 "city_limit": "true", "page_size": 15, "page_num": 1},
                         timeout=15).json()
    except Exception as e:
        print(brand, city, "ERR", e, flush=True)
        continue
    if j.get("status") != "1":
        print(f"{brand} {city}: API异常 {j.get('infocode')} {j.get('info')}", flush=True)
        if j.get("infocode") in ("10044", "10021", "10020"):
            quota_hit = True
            break
        continue
    pois = j.get("pois", [])
    print(f"{brand} {city}: {len(pois)} POI", flush=True)
    for p in pois[:12]:
        pid = p.get("id")
        if not pid:
            continue
        n_poi += 1
        try:
            d = requests.get("https://restapi.amap.com/v3/place/detail",
                             params={"key": K, "id": pid}, timeout=15).json()
        except Exception:
            continue
        if d.get("status") != "1":
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
                fn = f"v14_{pid}_{len(log)}.jpg"
                (OUT / fn).write_bytes(r.content)
                seen_sha.add(sha); dhashes.append(h)
                log[url] = {"file": fn, "sha256": sha, "dhash": h, "poi": pid,
                            "poi_name": poi.get("name"), "brand_query": brand,
                            "city": poi.get("cityname") or city,
                            "addr": poi.get("address", "")}
                n_dl += 1
            except Exception:
                continue
            time.sleep(0.12)
    time.sleep(0.3)
    if quota_hit:
        break
json.dump(log, open(ROOT / "data/amap_v14_milkchain_log.json", "w"), ensure_ascii=False, indent=1)
print(f"POI {n_poi}, 下载 {n_dl}, 去重丢弃 {n_dup}", flush=True)
