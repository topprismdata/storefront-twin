#!/usr/bin/env python3
"""高德 REST 赣闽鲁扩展轮 (2026-11-13, 用户指令: 江浙沪全扫)。
赣闽鲁 36 地级市 × 7 关键词(基础词+6变体) 全扫 → POI + photos → 全量下载新照片
(sha256+dhash 双重去重 vs 存量 images_v4/ 5916 文件)。
输出: data/amap_v11_hbhn_pois.json, data/images_v4/c*.jpg(接续编号), data/amap_v11_hbhn_download_log.json
沿用 v4/v7 管线: 变体词已证是最大杠杆(中点写法漏检)。浙江 11 市 v4+v7 已扫, 本轮不重扫。
"""
import urllib.request, ssl, json, hashlib, os, time
from urllib.parse import quote
from PIL import Image

ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
KEY = '7cefa505e38404f1a83f8eea34565e6c'
ROOT = '/Users/mac/chatgpt-harvest/6af41562'
OUTDIR = os.path.join(ROOT, 'data/images_v4')
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/156.0.0.0 Safari/537.36'

CITIES = ['武汉', '黄石', '十堰', '宜昌', '襄阳', '鄂州', '荆门', '孝感', '荆州', '黄冈', '咸宁', '随州', '恩施',
          '长沙', '株洲', '湘潭', '衡阳', '邵阳', '岳阳', '常德', '张家界', '益阳', '郴州', '永州', '怀化', '娄底', '湘西']
KEYWORDS = ['左邻右舍', '左邻·右舍', '左邻右社', '左邻ZUOLIN', 'GoodNeighbors',
            '左邻 便利店', '亲如一家 便利']


def get(url, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Referer': 'https://restapi.amap.com/'})
            return urllib.request.urlopen(req, timeout=25, context=ctx).read()
        except Exception:
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


def dhash(img, size=8):
    g = img.convert('L').resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    rows = [px[i * (size + 1):(i + 1) * (size + 1)] for i in range(size)]
    bits = 0
    for r in rows:
        for i in range(size):
            bits = (bits << 1) | (1 if r[i] > r[i + 1] else 0)
    return bits


def main():
    # 存量索引: 全量重算 sha+dhash (索引只覆盖 1046/5916, 不全)
    sha_idx = json.load(open(os.path.join(ROOT, 'data/sha_file_index.json')))
    have_sha, have_dh = set(), set()
    allfiles = sorted(f for f in os.listdir(OUTDIR) if f.endswith('.jpg'))
    for i, f in enumerate(allfiles):
        try:
            b = open(os.path.join(OUTDIR, f), 'rb').read()
            have_sha.add(hashlib.sha256(b).hexdigest())
            have_dh.add(dhash(Image.open(__import__('io').BytesIO(b))))
        except Exception:
            pass
        if i % 1000 == 0:
            print(f'索引存量 {i}/{len(allfiles)}', flush=True)
    existing = [f for f in allfiles if f.startswith('c')]
    next_c = max((int(f[1:].split('.')[0]) for f in existing), default=0) + 1
    print(f'存量: sha {len(have_sha)}, dhash {len(have_dh)}, next c{next_c}', flush=True)

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
        print(f'{city}: +{len(pois)-n0} POI (cum {len(pois)})', flush=True)
        time.sleep(0.4)
    json.dump(list(pois.values()), open(os.path.join(ROOT, 'data/amap_v11_hbhn_pois.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'POI 总 {len(pois)}, photos {sum(len(p["photos"]) for p in pois.values())} — 开始下载', flush=True)

    # 下载去重
    log, dled, dup_sha, dup_dh, bad = {}, 0, 0, 0, 0
    for pi, p in enumerate(pois.values()):
        for ph in p['photos']:
            u = ph['url']
            if u in log:
                continue
            try:
                b = get(u)
                sha = hashlib.sha256(b).hexdigest()
                if sha in have_sha:
                    dup_sha += 1; log[u] = {'file': None, 'dup': 'sha', 'poi': p['id']}; continue
                img = Image.open(io_ := __import__('io').BytesIO(b))
                dh = dhash(img)
                if dh in have_dh:
                    dup_dh += 1; log[u] = {'file': None, 'dup': 'dhash', 'poi': p['id']}; continue
                fn = f'c{next_c}.jpg'; next_c += 1
                open(os.path.join(OUTDIR, fn), 'wb').write(b)
                have_sha.add(sha); have_dh.add(dh)
                sha_idx[sha] = f'images_v4/{fn}'
                log[u] = {'file': fn, 'sha256': sha, 'dhash': dh, 'poi': p['id'],
                          'poi_name': p['name'], 'city': p['city'], 'title': ph.get('title', '')}
                dled += 1
            except Exception:
                bad += 1; log[u] = {'file': None, 'err': 1, 'poi': p['id']}
        if pi % 200 == 0:
            print(f'[{pi}/{len(pois)}] 下载 {dled} (sha重 {dup_sha}, dh重 {dup_dh}, 坏 {bad})', flush=True)
            json.dump(log, open(os.path.join(ROOT, 'data/amap_v11_hbhn_download_log.json'), 'w'), ensure_ascii=False)
    json.dump(log, open(os.path.join(ROOT, 'data/amap_v11_hbhn_download_log.json'), 'w'), ensure_ascii=False)
    json.dump(sha_idx, open(os.path.join(ROOT, 'data/sha_file_index.json'), 'w'), ensure_ascii=False)
    print(f'DONE: 新下载 {dled}, sha重 {dup_sha}, dhash重 {dup_dh}, 坏 {bad}, 总文件 {next_c-1}', flush=True)


if __name__ == '__main__':
    main()
