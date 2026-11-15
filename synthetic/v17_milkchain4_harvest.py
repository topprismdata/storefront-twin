#!/usr/bin/env python3
"""v17 — 液奶连锁表第三批: 弱品牌加深 + 新品牌铺开。额度耗尽自动停, 断点续跑。"""
import hashlib, json, time
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
K = "7cefa505e38404f1a83f8eea34565e6c"
OUT = ROOT / "data/images_v17_milkchain4"
OUT.mkdir(parents=True, exist_ok=True)
LOGF = ROOT / "data/amap_v17_milkchain_log.json"
S = requests.Session()
S.headers["Referer"] = "https://www.amap.com/"

PLAN = [
    # 弱品牌加深(E23b 低分)
    ("7-11便利店", "成都市", "7-11"),
    ("全家便利店", "苏州市", "全家"),
    ("丹尼斯全日鲜", "新乡市", "丹尼斯"),
    ("零食很忙", "南昌市", "零食很忙"),
    ("快客便利店", "苏州市", "快客"),
    ("唐久便利店", "石家庄市", "唐久"),
    ("万嘉便利店", "泉州市", "万嘉"),
    ("北京华联超市", "太原市", "北京华联"),
    ("利群超市", "烟台市", "利群"),
    ("华润万家", "广州市", "华润万家"),
    ("盒马NB", "深圳市", "盒马NB"),
    ("TODAY便利店", "厦门市", "TODAY"),
    # 便利店新品牌
    ("中百罗森", "武汉市", "中百罗森"),
    ("苏果便利", "南京市", "苏果便利"),
    ("好邻居便利店", "北京市", "好邻居"),
    ("每日每夜", "乌鲁木齐市", "每日每夜"),
    ("八点半便利店", "南宁市", "八点半"),
    ("友客便利店", "青岛市", "友客"),
    ("金虎便利店", "太原市", "金虎"),
    ("舞东风超市", "成都市", "舞东风"),
    ("合家欢便利店", "东莞市", "合家欢"),
    ("佳乐家便利", "潍坊市", "佳乐家"),
    # 零食新品牌
    ("三只松鼠投食店", "芜湖市", "三只松鼠"),
    ("吖嘀吖嘀", "佛山市", "吖嘀吖嘀"),
    ("恰货", "长沙市", "恰货铺子"),
    # 卖场/超市新品牌
    ("中百超市", "武汉市", "中百"),
    ("三江超市", "宁波市", "三江"),
    ("北国超市", "石家庄市", "北国"),
    ("大商超市", "大连市", "大商"),
    ("人人乐", "深圳市", "人人乐"),
    ("华冠超市", "北京市", "华冠"),
    ("家得乐超市", "贵阳市", "家得乐"),
    ("合力超市", "贵阳市", "合力超市"),
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
                fn = f"v17_{pid}_{len(log)}.jpg"
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
