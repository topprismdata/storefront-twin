#!/usr/bin/env python3
"""demo_server — 左邻右舍多品牌门头识别 demo(本地离线)。
拖图 → 品牌 + 招牌文字 + 候选分数。管线 = CLIP+DINO 特征 → per-brand 探针 top-3
→ 本地 PP-OCR 文字确认(最长别名优先) → 低置信/文字不匹配 → unknown。
仅内部演示; 权利链审核中, 不得公开宣传准确率。"""
import io, json, re, sys, time
import numpy as np
from pathlib import Path
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
sys.path.insert(0, str(ROOT / "synthetic"))
import e8_train as e8
from e20_text_confirm import det_boxes, merge_lines, rec_crop
import visual_baseline as vb

# ---------- 启动: 特征缓存 + 探针 ----------
z = np.load(ROOT / "synthetic/vis_cache/e25_feats.npz", allow_pickle=True)
Vc, Vd, okrows = z["Vc"], z["Vd"], list(z["rows"])
X = np.hstack([e8.l2(Vc), e8.l2(Vd)])
brands = sorted(set(r["brand"] for r in okrows))
rng = np.random.RandomState(0)
gidx = {}
for b in brands:
    bi = [i for i, r in enumerate(okrows) if r["brand"] == b]
    stores = sorted(set(okrows[i]["poi"] for i in bi))
    rng.shuffle(stores)
    g_stores = set(stores[:max(1, int(len(stores) * 0.7))])
    gidx[b] = [i for i in bi if okrows[i]["poi"] in g_stores]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
all_gal = sorted(i for b in brands for i in gidx[b])
gsc = StandardScaler().fit(X[all_gal])
Xg = gsc.transform(X[all_gal])
gbrands = [okrows[i]["brand"] for i in all_gal]
probes = {}
for b in brands:
    if b == "local":
        continue  # local=杂牌合桶, demo 目录外应走 unknown 而非杂牌猜
    y = np.array([1 if g == b else 0 for g in gbrands])
    if y.sum() >= 5:
        probes[b] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xg, y)
galX = {b: X[gidx[b]] for b in probes}
print(f"探针 {len(probes)} 品牌就绪", flush=True)

# ---------- 样例画廊: 左邻右舍专场 12 张(测试分割)置顶 + 每品牌 ≤3 张 + 6 张目录外 ----------
_rng = np.random.RandomState(7)
SAMPLES = []
zn_test = [i for i, r in enumerate(okrows) if r["brand"] == "左邻右舍" and i not in set(gidx["左邻右舍"])]
for i in _rng.choice(zn_test, min(12, len(zn_test)), replace=False):
    SAMPLES.append({"k": len(SAMPLES), "brand": "左邻右舍", "idx": int(i), "path": okrows[i]["path"]})
for b in sorted(probes):
    if b == "左邻右舍":
        continue
    test_bi = [i for i, r in enumerate(okrows) if r["brand"] == b and i not in set(gidx[b])]
    if not test_bi:
        continue
    for i in _rng.choice(test_bi, min(3, len(test_bi)), replace=False):
        SAMPLES.append({"k": len(SAMPLES), "brand": b, "idx": int(i), "path": okrows[i]["path"]})
locs = [i for i, r in enumerate(okrows) if r["brand"] == "local"]
for i in _rng.choice(locs, min(6, len(locs)), replace=False):
    SAMPLES.append({"k": len(SAMPLES), "brand": "目录外", "idx": int(i), "path": okrows[i]["path"]})
print(f"画廊 {len(SAMPLES)} 张 (左邻右舍专场 {min(12, len(zn_test))})", flush=True)

# ---------- 别名表(与 E25 冻结一致) ----------
ALIAS = {
    "每一天": ["每一天"], "见福": ["见福"], "唐久": ["唐久"], "红旗连锁": ["红旗连锁", "红旗超市", "红旗"],
    "零食很忙": ["零食很忙"], "赵一鸣零食": ["赵一鸣", "赵一鸣省钱超市"], "好想来": ["好想来"],
    "家家悦": ["家家悦"], "盒马鲜生": ["盒马", "盒马鲜生"], "好特卖": ["好特卖", "HOTMAXX"],
    "快客": ["快客"], "恰货铺子": ["恰货铺子", "恰货"], "天福": ["天福"], "万嘉": ["万嘉"],
    "好德": ["好德"], "悦来悦喜": ["悦来悦喜"], "零食有鸣": ["零食有鸣"], "喜士多": ["喜士多"],
    "丹尼斯": ["丹尼斯"], "苏宁小店": ["苏宁小店", "苏宁"], "京客隆": ["京客隆"],
    "北京华联": ["北京华联", "BHG"], "天虹": ["天虹"], "盒马NB": ["盒马NB"], "TODAY": ["TODAY"],
    "戴永红": ["戴永红"], "新佳宜": ["新佳宜"], "良品铺子": ["良品铺子", "良品"],
    "老婆大人": ["老婆大人"], "谊品生鲜": ["谊品生鲜", "谊品"], "利群": ["利群"],
    "山姆": ["山姆", "SAM", "SAMS"], "华润万家": ["华润万家", "华润"], "开市客": ["开市客", "COSTCO"],
    "左邻右舍": ["左邻", "右舍"], "美宜佳": ["美宜佳"], "芙蓉兴盛": ["芙蓉兴盛", "芙蓉", "兴盛"],
    "京东便利店": ["京东"], "天猫小店": ["天猫"], "便利蜂": ["便利蜂"], "十足": ["十足"],
    "罗森": ["罗森", "LAWSON"], "7-11": ["7-ELEVEN", "711"], "全家": ["全家", "FAMILYMART"],
    "中百罗森": ["中百罗森"], "苏果便利": ["苏果", "苏果便利", "苏果超市"], "好邻居": ["好邻居"],
    "每日每夜": ["每日每夜"], "八点半": ["八点半"], "友客": ["友客"], "金虎": ["金虎"],
    "舞东风": ["舞东风"], "合家欢": ["合家欢"], "佳乐家": ["佳乐家"], "三只松鼠": ["三只松鼠"],
    "吖嘀吖嘀": ["吖嘀吖嘀"], "中百": ["中百超市", "中百仓储"], "三江": ["三江购物", "三江超市"],
    "北国": ["北国超市", "北国"], "大商": ["大商超市", "大商集团"], "人人乐": ["人人乐"],
    "华冠": ["华冠超市", "华冠"], "家得乐": ["家得乐"], "合力超市": ["合力超市", "合力便利"],
}
STOP = {"便利", "超市", "小店", "便利店", "零食"}
ECC, EDD = vb.Embedder("clip"), vb.Embedder("dinov2")


def confirm(cands, texts):
    tn = [re.sub(r"\s+", "", t).upper() for t in texts if t and t.strip()]

    def hit(a, strict_only=False):
        na = re.sub(r"\s+", "", a).upper()
        if not na:
            return False
        if na in "".join(tn):
            return True
        if strict_only:
            return False  # text_all(非候选)路径只允许整词, 防通用前缀误吃
        if len(na) >= 3 and na[:2] not in STOP and any(na[:2] in t for t in tn):
            return True
        if len(na) >= 5 and all(ord(c) < 256 for c in na):
            if any(na in t or t in na for t in tn if t):
                return True
            import difflib
            return any(difflib.SequenceMatcher(None, na, t).ratio() >= 0.75 for t in tn if len(t) >= 5)
        if len(na) >= 4 and any(ord(c) > 255 for c in na):
            tail, head = na[1:], na[:-1]
            return any(tail in t or head in t for t in tn)
        return False

    def best_len(als):
        best = 0
        for a in als:
            na = re.sub(r"\s+", "", a).upper()
            if hit(a) and len(na) > best:
                best = len(na)
        return best

    def best_len_strict(als):
        best = 0
        for a in als:
            na = re.sub(r"\s+", "", a).upper()
            if hit(a, strict_only=True) and len(na) > best:
                best = len(na)
        return best

    scored = [(b, best_len_strict(ALIAS[b])) for b in ALIAS]
    scored = [(b, L) for b, L in scored if L]
    cand_set = set(cands)
    cand_hits = [(b, best_len(ALIAS[b])) for b in ALIAS if b in cand_set]
    cand_hits = [(b, L) for b, L in cand_hits if L]
    if cand_hits:
        cand_hits.sort(key=lambda kv: -kv[1])
        return cand_hits[0][0], "text_cand"
    if scored:
        scored.sort(key=lambda kv: -kv[1])
        return scored[0][0], "text_all"
    return cands[0], "visual_fallback"


def infer(pil):
    t0 = time.time()
    vec_c = ECC.embed_pil([pil])[0]
    vec_d = EDD.embed_pil([pil])[0]
    q = np.hstack([e8.l2(vec_c[None]), e8.l2(vec_d[None])])
    ss = {b: float(probes[b].predict_proba(gsc.transform(q))[0, 1]) for b in probes}
    top = sorted(ss.items(), key=lambda kv: -kv[1])[:3]
    margin = top[0][1] - top[1][1]
    import cv2
    arr = np.asarray(pil.convert("RGB"))[:, :, ::-1].copy()
    texts = [rec_crop(arr, bx) for bx in merge_lines(det_boxes(arr))]
    pred, source = confirm([b for b, _ in top], texts)
    cjk = sum(1 for t in texts for ch in t if "\u4e00" <= ch <= "\u9fff")
    unknown = False
    if source == "visual_fallback":
        if cjk >= 4:               # 有可读文字但目录别名全不匹配 → 非目录品牌
            unknown = True
        elif margin < 0.02:        # 探针分不开 → 不猜
            unknown = True
    return {
        "brand": "unknown" if unknown else pred,
        "source": source, "margin": round(margin, 3),
        "top3": [{"brand": b, "score": round(s, 3)} for b, s in top],
        "ocr": texts[:6], "latency_s": round(time.time() - t0, 2),
    }


PAGE = """<!doctype html><html lang=zh><meta charset=utf-8>
<title>门头品牌识别 demo</title>
<style>
body{font-family:PingFang SC,sans-serif;max-width:720px;margin:40px auto;color:#222}
#drop{border:2px dashed #999;border-radius:12px;padding:48px;text-align:center;cursor:pointer}
#drop.hot{border-color:#e60012;background:#fff5f5}
img#prev{max-width:100%;max-height:320px;display:block;margin:16px auto}
.row{display:flex;gap:12px;margin-top:14px;flex-wrap:wrap}
.card{border:1px solid #ddd;border-radius:10px;padding:10px 16px;min-width:150px}
.brand{font-size:22px;font-weight:700;color:#e60012}
.mut{color:#888;font-size:12px}
</style>
<h2>门头品牌识别 <span class=mut>内部演示 · 未部署 · 权利链审核中</span></h2>
<div id=drop>拖一张门头照片到这里, 或点击选择<input type=file id=f accept=image/* hidden></div>
<img id=prev hidden>
<div id=out></div>
<h3>测试画廊 <span class=mut>点击即识别 · 含目录外样例(应显示 unknown)</span></h3>
<div id=gal>加载中…</div>
<style>
#gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.gcell{cursor:pointer;border:1px solid #ddd;border-radius:8px;overflow:hidden;text-align:center}
.gcell:hover{border-color:#e60012}
.gcell img{width:100%;height:100px;object-fit:cover;display:block}
.gcell .gb{font-size:11px;color:#666;padding:3px}
.gcell.out .gb{color:#b60}
</style>
<script>
const drop=document.getElementById('drop'),f=document.getElementById('f'),out=document.getElementById('out'),prev=document.getElementById('prev'),gal=document.getElementById('gal');
drop.onclick=()=>f.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hot')};
drop.ondragleave=()=>drop.classList.remove('hot');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hot');if(e.dataTransfer.files[0])go(e.dataTransfer.files[0])};
f.onchange=()=>{if(f.files[0])go(f.files[0])};
async function go(file){
  prev.src=URL.createObjectURL(file);prev.hidden=false;
  out.innerHTML='<p>识别中…</p>';
  const fd=new FormData();fd.append('image',file);
  const r=await fetch('/infer',{method:'POST',body:fd});
  render(await r.json());
}
function render(j){
  const top=j.top3.map(t=>`<div class=card><div class=mut>探针 ${Math.round(t.score*100)}%</div><b>${t.brand}</b></div>`).join('');
  const hit=(j.truth?(j.brand===j.truth?'<span style=color:#0a0>✓ 与标注一致</span>':'<span style=color:#c60>标注: '+j.truth+'</span>'):'');
  out.innerHTML=`<div class=row>
    <div class=card><div class=mut>判决(${j.source}) ${hit}</div><div class=brand>${j.brand}</div></div>
    ${top}</div>
    <p class=mut>OCR 读到: ${j.ocr.length?j.ocr.map(t=>`「${t}」`).join(' '):'(未读出文字)'} · 耗时 ${j.latency_s}s</p>`;
}
async function runK(k){
  out.innerHTML='<p>识别中…</p>';
  const r=await fetch('/run?k='+k);
  render(await r.json());
}
(async()=>{
  const items=await (await fetch('/gallery')).json();
  const zn=items.filter(s=>s.brand==='左邻右舍'), rest=items.filter(s=>s.brand!=='左邻右舍');
  const cell=s=>`<div class="gcell ${s.brand==='目录外'?'out':''}" onclick="runK(${s.k})">
    <img src="/img?k=${s.k}" loading=lazy><div class=gb>${s.brand}</div></div>`;
  gal.innerHTML=`<div class=mut style=margin:6px 0>—— 左邻右舍专场(测试图 ${zn.length} 张) ——</div>
    <div style=display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px>${zn.map(cell).join('')}</div>
    <div class=mut style=margin:12px 0 6px>—— 其他连锁与目录外 ——</div>
    <div style=display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px>${rest.map(cell).join('')}</div>`;
})();
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if self.path == "/gallery":
            items = [{"k": s["k"], "brand": s["brand"]} for s in SAMPLES]
            return self._send(200, json.dumps(items, ensure_ascii=False).encode("utf-8"),
                              "application/json")
        m = re.match(r"^/img\?k=(\d+)$", self.path)
        if m:
            s = SAMPLES[int(m.group(1))]
            with open(s["path"], "rb") as f:
                self._send(200, f.read(), "image/jpeg")
            return
        m = re.match(r"^/run\?k=(\d+)$", self.path)
        if m:
            s = SAMPLES[int(m.group(1))]
            pil = Image.open(s["path"]).convert("RGB")
            out = infer(pil)
            out["truth"] = s["brand"]
            return self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"),
                              "application/json")
        self._send(404, b"{}", "text/plain")

    def do_POST(self):
        if self.path != "/infer":
            return self._send(404, b"{}", "application/json")
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        # multipart: 取第一个文件段
        m = re.search(rb"\r\n\r\n(.*?)\r\n--", data, re.S)
        if not m:
            return self._send(400, json.dumps({"error": "no image"}).encode(), "application/json")
        try:
            pil = Image.open(io.BytesIO(m.group(1))).convert("RGB")
            out = infer(pil)
            self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"), "application/json")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    print(f"demo → http://127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
