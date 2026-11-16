#!/usr/bin/env python3
"""v17 — 液奶连锁表第三批: 弱品牌加深 + 新品牌铺开。额度耗尽自动停, 断点续跑。"""
import hashlib, json, time
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
K = "7cefa505e38404f1a83f8eea34565e6c"
OUT = ROOT / "data/images_v19_teafood"
OUT.mkdir(parents=True, exist_ok=True)
LOGF = ROOT / "data/amap_v19_teafood_log.json"
S = requests.Session()
S.headers["Referer"] = "https://www.amap.com/"

PLAN = [
    # 茶饮(负类新品类主力)
    ("奈雪的茶", "深圳市", "奈雪的茶"),
    ("喜茶", "上海市", "喜茶"),
    ("古茗", "杭州市", "古茗"),
    ("沪上阿姨", "上海市", "沪上阿姨"),
    ("益禾堂", "武汉市", "益禾堂"),
    ("书亦烧仙草", "成都市", "书亦烧仙草"),
    # 快餐/卤味
    ("华莱士", "福州市", "华莱士"),
    ("正新鸡排", "上海市", "正新鸡排"),
    ("绝味鸭脖", "长沙市", "绝味鸭脖"),
    ("煌上煌", "南昌市", "煌上煌"),
    # 烘焙/小吃/中式快餐
    ("好利来", "北京市", "好利来"),
    ("巴比馒头", "上海市", "巴比馒头"),
    ("老乡鸡", "合肥市", "老乡鸡"),
    ("来伊份", "上海市", "来伊份"),
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
                fn = f"v19_{pid}_{len(log)}.jpg"
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
