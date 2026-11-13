#!/usr/bin/env python3
"""高德 REST v5 品牌相册采集 (2026-11-13, key 自 11-11 会话恢复)。
1) 浙江 11 地级市 × 关键词全扫 → POI + photos[]
2) v5/place/detail 测试单店全相册
3) 全量下载新照片 → sha256+dHash 去重 vs 存量 656
输出: data/amap_v4_pois.json, data/images_v4/, data/amap_v4_download_log.json"""
import urllib.request, ssl, json, hashlib, os, re, time, io
from urllib.parse import quote
from PIL import Image

ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
KEY = '7cefa505e38404f1a83f8eea34565e6c'
ROOT = '/Users/mac/chatgpt-harvest/6af41562'
OUTDIR = os.path.join(ROOT, 'data/images_v4')
os.makedirs(OUTDIR, exist_ok=True)
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/156.0.0.0 Safari/537.36'

CITIES = ['杭州', '宁波', '温州', '嘉兴', '湖州', '绍兴', '金华', '衢州', '舟山', '台州', '丽水']
KEYWORDS = ['左邻右舍']


def get(url, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Referer': 'https://restapi.amap.com/'})
            return urllib.request.urlopen(req, timeout=25, context=ctx).read()
        except Exception as e:
            if a == tries - 1:
                raise
            time.sleep(1.2 * (a + 1))


def jget(url):
    for a in range(5):
        j = json.loads(get(url))
        if j.get('status') == '1':
            return j
        info = j.get('info', '')
        if 'QPS' in info or 'LIMIT' in info or 'TOO' in info.upper():
            time.sleep(1.5 * (a + 1)); continue
        return j
    return j


def main():
    pois = {}
    for city in CITIES:
        n0 = len(pois)
        for kw in KEYWORDS:
            for page in range(1, 9):
                url = (f"https://restapi.amap.com/v5/place/text?key={KEY}&keywords={quote(kw)}"
                       f"&region={quote(city)}&city_limit=true&page_size=25&page_num={page}&show_fields=photos,business")
                j = jget(url)
                batch = j.get('pois') or []
                for p in batch:
                    pid = p.get('id')
                    if pid and pid not in pois:
                        pois[pid] = {'id': pid, 'name': p.get('name'), 'city': p.get('cityname'),
                                     'address': p.get('address'), 'type': p.get('type'),
                                     'photos': [{'title': ph.get('title', ''), 'url': ph.get('url')}
                                                 for ph in (p.get('photos') or []) if ph.get('url')]}
                if len(batch) < 25:
                    break
                time.sleep(0.4)
        print(f'{city}: +{len(pois)-n0} POI', flush=True)
        time.sleep(0.4)
    tot_ph = sum(len(p['photos']) for p in pois.values())
    print(f'TOTAL POI {len(pois)}, photos {tot_ph}', flush=True)
    json.dump(list(pois.values()), open(os.path.join(ROOT, 'data/amap_v4_pois.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
