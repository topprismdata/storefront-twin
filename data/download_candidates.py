#!/usr/bin/env python3
import json, os, hashlib, concurrent.futures as cf, urllib.request, ssl, socket
from urllib.parse import urlparse

BASE = "/Users/mac/chatgpt-harvest/6af41562/data"
IMG = os.path.join(BASE, "images")
os.makedirs(IMG, exist_ok=True)
cands = json.load(open(os.path.join(BASE, "bing_candidates.json")))
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"

def fetch(i, c):
    url = c["murl"]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": c.get("purl") or "https://www.bing.com/", "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"})
        with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
            ct = r.headers.get("Content-Type","")
            data = r.read(8_000_000)
        if not ct.startswith("image") and not (data[:3]==b"\xff\xd8\xff" or data[:8]==b"\x89PNG\r\n\x1a\n" or data[:4]==b"RIFF"):
            return {"i":i,"ok":False,"err":"not-image:"+ct[:30]}
        if len(data) < 4000:
            return {"i":i,"ok":False,"err":"too-small"}
        sha = hashlib.sha256(data).hexdigest()
        ext = ".png" if data[:8]==b"\x89PNG\r\n\x1a\n" else (".webp" if data[:4]==b"RIFF" else ".jpg")
        fn = f"c{i:04d}_{sha[:12]}{ext}"
        with open(os.path.join(IMG, fn), "wb") as f: f.write(data)
        return {"i":i,"ok":True,"file":fn,"sha":sha,"bytes":len(data),"url":url,"purl":c.get("purl",""),"turl":c.get("turl",""),"title":c.get("title",""),"query":c.get("query",""),"host":urlparse(url).netloc}
    except Exception as e:
        return {"i":i,"ok":False,"err":f"{type(e).__name__}:{str(e)[:60]}"}

results=[]
with cf.ThreadPoolExecutor(max_workers=24) as ex:
    futs=[ex.submit(fetch,i,c) for i,c in enumerate(cands)]
    for fu in cf.as_completed(futs):
        results.append(fu.result())
results.sort(key=lambda r:r["i"])
ok=[r for r in results if r["ok"]]
json.dump(results, open(os.path.join(BASE,"download_log.json"),"w"), ensure_ascii=False, indent=1)
# byte dedup
seen={}
for r in ok:
    seen.setdefault(r["sha"], []).append(r["file"])
dups={k:v for k,v in seen.items() if len(v)>1}
print("downloaded_ok", len(ok), "failed", len(results)-len(ok))
print("unique_sha", len(seen), "dup_groups", len(dups), "dup_files", sum(len(v)-1 for v in dups.values()))
from collections import Counter
errs=Counter(r["err"].split(":")[0] for r in results if not r["ok"])
print("err_types", dict(errs))