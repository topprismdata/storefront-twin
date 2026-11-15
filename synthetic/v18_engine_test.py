#!/usr/bin/env python3
"""v18 新类目验收: 新抓的门头照全部过 demo 引擎, 期望=目录外品牌诚实拒答(unknown)。"""
import json, glob, sys
import urllib.request
import warnings
warnings.filterwarnings("ignore")

ROOT = "/Users/mac/chatgpt-harvest/6af41562"
log = json.load(open(f"{ROOT}/data/amap_v18_newchains_log.json"))
items = [v for v in (log.values() if isinstance(log, dict) else log)
         if isinstance(v, dict) and v.get("sha256")]

B = "http://127.0.0.1:8787"
res = []
for i, v in enumerate(items):
    path = f"{ROOT}/data/images_v18_newchains/{v['file']}"
    try:
        boundary = "X"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
                f"filename=\"{v['file']}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() \
               + open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"{B}/infer", data=body, headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"})
        j = json.loads(urllib.request.urlopen(req, timeout=60).read())
        res.append({"brand_query": v["brand_query"], "city": v["city"],
                    "pred": j["brand"], "source": j["source"], "margin": j["margin"],
                    "ocr": j["ocr"][:2], "lat": j["latency_s"]})
    except Exception as e:
        res.append({"brand_query": v["brand_query"], "city": v["city"], "pred": f"ERR:{e}"})
    if (i + 1) % 40 == 0:
        print(f"...{i+1}/{len(items)}", flush=True)

unk = sum(1 for r in res if r.get("pred") == "unknown")
hit_own = [r for r in res if r.get("pred") not in ("unknown",) and not r["pred"].startswith("ERR")]
print(f"\nv18 新类目 {len(res)} 张: unknown {unk} ({unk/max(len(res),1)*100:.0f}%), 误判入目录 {len(hit_own)}")
import collections
per = collections.defaultdict(lambda: [0, 0])
for r in res:
    per[r["brand_query"]][1] += 1
    if r.get("pred") == "unknown":
        per[r["brand_query"]][0] += 1
for b, (u, n) in sorted(per.items()):
    print(f"  {b}: unknown {u}/{n}")
wrong = [r for r in res if r.get("pred") not in ("unknown",) and not r["pred"].startswith("ERR")]
for r in wrong[:15]:
    print(f"  误判: {r['brand_query']}({r['city']}) -> {r['pred']} [{r['source']}] margin={r['margin']} OCR={r['ocr']}")
json.dump(res, open(f"{ROOT}/synthetic/vis_cache/v18_newchains_engine_test.json", "w"),
          ensure_ascii=False, indent=1)
