#!/usr/bin/env python3
"""独立新门店泛化集采集骨架（阶段A产物，复用已验证 getPoiInfo 通道）。
用法：
  python3 collect_independent.py --poi-ids ids.txt --out gen_candidates
  python3 collect_independent.py --ledger brand_ledger.csv --out gen_candidates
流程：POI ID → getPoiInfo gallery → 下载门头图 → sha256+dHash去重 → 门店组隔离校验 → 候选登记。
注意：本脚本只做采集+去重+隔离校验；品牌真伪/双盲标注/权利审查须后续人工+视觉核验（见方案）。
治理：探索性，rights=pending_review，不训练不部署。"""
import json, os, hashlib, argparse, ssl, urllib.request, csv, sys
from urllib.parse import urlparse

BASE = "/Users/mac/chatgpt-harvest/6af41562"
DATA = os.path.join(BASE, "data")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE


def http_get(url, referer="https://www.amap.com/", binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer})
    with urllib.request.urlopen(req, timeout=25, context=CTX) as r:
        data = r.read(12_000_000)
    return data if binary else data.decode("utf-8", "ignore")


def search_pois_rest(key, cities, keywords="左邻右舍", types="060200", max_pages=8):
    """高德 Web服务 REST POI 搜索（无反爬，无需浏览器）→ POI + photos[].url。
    v5/place/text: show_fields=photos 直接返回图片; types=060200 过滤便利店(排除小区/餐馆);
    city_limit=true 限定城市; 分页 page=1..max_pages（同请求最多200条, page_size=25）。"""
    from urllib.parse import quote
    import time
    pois = {}
    for c in cities:
        n_city = 0
        for page in range(1, max_pages + 1):
            url = (f"https://restapi.amap.com/v5/place/text?key={key}"
                   f"&keywords={quote(keywords)}&region={quote(c)}&city_limit=true"
                   f"&page_size=25&page_num={page}&show_fields=photos")
            if types: url += f"&types={types}"
            batch = None
            for attempt in range(5):  # QPS 限流重试（个人key并发配额低）
                try:
                    j = json.loads(http_get(url, referer="https://restapi.amap.com/"))
                    info = j.get("info", "")
                    if j.get("status") == "1":
                        batch = j.get("pois", []); break
                    if "QPS" in info or "LIMIT" in info:
                        time.sleep(1.5 * (attempt + 1)); continue
                    if page == 1: print(f"  {c}: API {info}")
                    break
                except Exception as e:
                    print(f"  {c} p{page}: ERR {str(e)[:50]}"); break
            time.sleep(0.4)  # 请求间隔，避免触发 QPS
            if batch is None: break
            if not batch: break
            for p in batch:
                if p["id"] not in pois:
                    pois[p["id"]] = p; n_city += 1
            if len(batch) < 25: break
        print(f"  {c}: +{n_city} new POI (types={types or 'all'})", flush=True)
    return list(pois.values())


def fetch_poi_gallery(poi_id):
    """高德 getPoiInfo（无需登录）→ 相册 gallery，带 category 标签。"""
    url = f"https://www.amap.com/ssr/api/getPoiInfo?id={poi_id}"
    try:
        txt = http_get(url)
        if txt.lstrip()[:1] == "<":
            return {"poi_id": poi_id, "error": "ANTI_BOT_HTML: getPoiInfo 需经登录态浏览器会话(CDP 9222)调用，裸 urllib 被反爬", "photos": []}
        j = json.loads(txt)
    except Exception as e:
        return {"poi_id": poi_id, "error": f"{type(e).__name__}:{str(e)[:60]}", "photos": []}
    # 结构随版本变化，防御性提取
    photos = []
    def walk(o):
        if isinstance(o, dict):
            if "url" in o and ("category" in o or "type" in o):
                photos.append(o)
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(j)
    return {"poi_id": poi_id, "photos": photos, "raw_keys": list(j.keys()) if isinstance(j, dict) else []}


def dhash(path, size=16):
    """256-bit 差值哈希（16x16）。需 cv2。"""
    import cv2
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None: return 0
    im = cv2.resize(im, (size + 1, size))
    bits = (im[:, 1:] > im[:, :-1]).flatten()
    h = 0
    for b in bits: h = (h << 1) | int(b)
    return h


def hamming(a, b): return bin(a ^ b).count("1")


def load_existing_groups():
    """现有 330 的 store_id/group + photo_family，用于隔离校验。"""
    reg = json.load(open(os.path.join(DATA, "registry_v3.json")))
    rows = reg if isinstance(reg, list) else reg.get("rows", reg.get("images", []))
    groups = {r.get("store_id") or r["image_id"] for r in rows}  # 与 visual_baseline.load_registry 同约定: store_id 优先否则 image_id
    hashes = {r["sha256"] for r in rows if r.get("sha256")}
    return groups, hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poi-ids", help="文本文件，每行一个高德 POI ID")
    ap.add_argument("--ledger", help="品牌方台账 CSV（含 poi_id/store_id/name/province 列）")
    ap.add_argument("--out", default="gen_candidates")
    ap.add_argument("--category", default="门头", help="getPoiInfo 模式只下载该 category；REST 模式忽略")
    ap.add_argument("--key", help="高德 Web服务 API key（启用 REST 搜索模式，无反爬）")
    ap.add_argument("--cities", help="逗号分隔城市名（REST 模式搜索区域）")
    ap.add_argument("--keywords", default="左邻右舍", help="REST 搜索关键词")
    ap.add_argument("--types", default="060200", help="POI类型过滤(060200便利店;空=全部)")
    args = ap.parse_args()

    outdir = os.path.join(DATA, args.out)
    os.makedirs(outdir, exist_ok=True)
    existing_groups, existing_sha = load_existing_groups()
    print(f"existing groups: {len(existing_groups)}, existing sha: {len(existing_sha)}")

    # 收集 POI：REST 搜索（优先，无反爬）/ POI ID 文件 / 品牌方台账
    pois = []
    rest_mode = bool(args.key and args.cities)
    if rest_mode:
        cities = [c.strip() for c in args.cities.split(",") if c.strip()]
        print(f"REST search: {args.keywords} in {cities}")
        for p in search_pois_rest(args.key, cities, args.keywords, types=args.types):
            pois.append({"poi_id": p["id"], "store_id": f"gen_amap_{p['id']}",
                         "name": p.get("name", ""), "province": p.get("pname", ""),
                         "city": p.get("cityname", ""), "addr": p.get("address", ""),
                         "typecode": p.get("typecode", ""),
                         "photos": p.get("photos", [])})
    if args.poi_ids:
        for line in open(args.poi_ids):
            pid = line.strip()
            if pid: pois.append({"poi_id": pid})
    if args.ledger:
        for row in csv.DictReader(open(args.ledger)):
            if row.get("poi_id"): pois.append(row)
    if not pois:
        print("NO POIS. 提供 --key+--cities（REST搜索）或 --poi-ids 或 --ledger。"); sys.exit(1)

    seen_sha = set(existing_sha)
    seen_dhash = []
    records = []
    iso_violations = []
    for p in pois:
        pid = p["poi_id"]
        store_id = p.get("store_id") or f"gen_amap_{pid}"
        # 门店组隔离校验
        if store_id in existing_groups:
            iso_violations.append({"poi_id": pid, "store_id": store_id, "reason": "store_id intersects existing group"})
            print(f"  SKIP {pid}: store_id {store_id} 已存在（隔离）")
            continue
        if rest_mode:
            photos = p.get("photos", [])  # REST 搜索已带 photos，无需 getPoiInfo
        else:
            gal = fetch_poi_gallery(pid)
            if gal.get("error"):
                print(f"  {pid}: {gal['error']}"); continue
            photos = gal["photos"]
        n_dl = 0
        for ph in photos:
            cat = ph.get("category") or ph.get("type") or ph.get("title") or ""
            if not rest_mode and args.category not in str(cat): continue
            url = ph.get("url")
            if not url: continue
            try:
                data = http_get(url, binary=True)
                if len(data) < 4000: continue
                sha = hashlib.sha256(data).hexdigest()
                if sha in seen_sha: continue  # 字节去重（含跨现有集）
                ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else (".webp" if data[:4] == b"RIFF" else ".jpg")
                fn = f"gen_{pid}_{sha[:12]}{ext}"
                fp = os.path.join(outdir, fn)
                open(fp, "wb").write(data)
                dh = dhash(fp)
                if any(hamming(dh, h) <= 12 for h in seen_dhash):
                    os.remove(fp); continue  # 感知去重
                seen_sha.add(sha); seen_dhash.append(dh)
                records.append({"image_id": f"gen_{len(records):04d}", "file": fn, "sha256": sha,
                                "dhash": dh, "store_id": store_id, "poi_id": pid,
                                "photo_family": f"genfam_{pid}", "category": cat,
                                "name": p.get("name", ""), "province": p.get("province", ""),
                                "city": p.get("city", ""), "addr": p.get("addr", ""),
                                "typecode": p.get("typecode", ""), "src_url": url,
                                "rights_status": "pending_review", "label": "unlabeled",
                                "brand_verdict_fullres": "", "gold_label": ""})
                n_dl += 1
            except Exception as e:
                print(f"    dl err {url[:50]}: {str(e)[:40]}")
        print(f"  {pid} ({store_id}): downloaded {n_dl} 门头图")

    json.dump(records, open(os.path.join(outdir, "gen_candidates.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(iso_violations, open(os.path.join(outdir, "isolation_violations.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\nTOTAL candidates: {len(records)} | isolation violations: {len(iso_violations)}")
    print("下一步：全分辨率视觉核验品牌真伪 → 双盲标注 → 权利审查 → 冻结模型盲测（见方案§5）")


if __name__ == "__main__":
    main()