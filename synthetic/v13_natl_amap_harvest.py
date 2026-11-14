#!/usr/bin/env python3
"""v12 — confusable_brand 真实困难负例采集 (amap REST)。
品牌走廊城市 × 红蓝系竞品便利店, v5 text 搜 POI → v3 detail 拉图 → sha256/dhash 去重 →
落盘 data/images_v13_natl/ + log。仅采集, 标签待全分辨率核验(pending_review)。"""
import hashlib, json, os, time
import requests
from pathlib import Path
from PIL import Image

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
K = "7cefa505e38404f1a83f8eea34565e6c"
OUT = ROOT / "data/images_v13_natl"
OUT.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers["Referer"] = "https://www.amap.com/"

BRANDS = ["罗森便利店", "全家便利店", "711便利店"]
CITIES = ["上海市", "苏州市", "杭州市", "南昌市"]

old = json.load(open(ROOT / "data/amap_v11_hbhn_download_log.json"))
seen_sha = {v["sha256"] for v in old.values() if isinstance(v, dict) and v.get("sha256")}
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


dhashes = [v.get("dhash") for v in old.values() if v.get("dhash")]
n_poi = n_dl = n_dup = 0
for brand in BRANDS:
    for city in CITIES:
        try:
            j = requests.get("https://restapi.amap.com/v5/place/text",
                             params={"key": K, "keywords": brand, "region": city,
                                     "city_limit": "true", "page_size": 15, "page_num": 1},
                             timeout=15).json()
        except Exception as e:
            print(brand, city, "ERR", e, flush=True)
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
                    from io import BytesIO
                    im = Image.open(BytesIO(r.content))
                    h = dh(im)
                    if any(ham(h, g) <= 8 for g in dhashes[-3000:]):
                        n_dup += 1
                        continue
                    fn = f"v13_{pid}_{len(log)}.jpg"
                    (OUT / fn).write_bytes(r.content)
                    seen_sha.add(sha); dhashes.append(h)
                    log[url] = {"file": fn, "sha256": sha, "dhash": h, "poi": pid,
                                "poi_name": poi.get("name"), "brand_query": brand,
                                "city": poi.get("cityname") or city,
                                "addr": poi.get("address", "")}
                    n_dl += 1
                except Exception:
                    continue
                time.sleep(0.15)
        time.sleep(0.3)
json.dump(log, open(ROOT / "data/amap_v13_natl_log.json", "w"), ensure_ascii=False, indent=1)
print(f"POI {n_poi}, 下载 {n_dl}, 去重丢弃 {n_dup}", flush=True)
