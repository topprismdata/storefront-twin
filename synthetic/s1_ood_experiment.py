#!/usr/bin/env python3
"""S1 — CLIP 零样本开集打分头(品牌留出协议, GPT 评审条件)。

特征: f_gal=gallery 近邻相似度(CLIP/DINO), f_ood=OOD提示-ID提示文本相似度差(MCM/CLIPN思路)。
校准: 目录gallery行 + v18品牌(瑞幸/库迪除外); 测试: 目录测试行 + 瑞幸/库迪(未见品牌)。
指标: AUROC / FPR@95TPR / UnknownRecall / 已知FPR。
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import e8_train as e8
import visual_baseline as vb

ROOT = "/Users/mac/chatgpt-harvest/6af41562"
z = np.load(f"{ROOT}/synthetic/vis_cache/e25_feats.npz", allow_pickle=True)
Vc, Vd, okrows = z["Vc"], z["Vd"], list(z["rows"])
brands = sorted(set(r["brand"] for r in okrows))
rng = np.random.RandomState(0)
gidx = {}
for b in brands:
    bi = [i for i, r in enumerate(okrows) if r["brand"] == b]
    stores = sorted(set(okrows[i]["poi"] for i in bi))
    rng.shuffle(stores)
    g_stores = set(stores[:max(1, int(len(stores) * 0.7))])
    gidx[b] = [i for i in bi if okrows[i]["poi"] in g_stores]

gal_named = [i for b in brands if b != "local" for i in gidx[b]]
test_named = [i for b in brands if b != "local"
              for i in (set(range(len(okrows))) - set(gidx[b])) if okrows[i]["brand"] == b]

# v18 嵌入(缓存)
import os
cachef = f"{ROOT}/synthetic/vis_cache/v18_feats.npz"
if not os.path.exists(cachef):
    log = json.load(open(f"{ROOT}/data/amap_v18_newchains_log.json"))
    items = [v for v in (log.values() if isinstance(log, dict) else log)
             if isinstance(v, dict) and v.get("sha256")]
    from PIL import Image
    ECC = vb.Embedder("clip"); EDD = vb.Embedder("dinov2")
    paths = [f"{ROOT}/data/images_v18_newchains/{v['file']}" for v in items]
    imgs = [Image.open(p).convert("RGB") for p in paths]
    Vc18 = ECC.embed_pil(imgs); Vd18 = EDD.embed_pil(imgs)
    np.savez(cachef, Vc=Vc18, Vd=Vd18,
             brands=np.array([v["brand_query"] for v in items]))
else:
    zz = np.load(cachef, allow_pickle=True)
    Vc18, Vd18 = zz["Vc"], zz["Vd"]
    items = None
v18_brands = list(np.load(cachef, allow_pickle=True)["brands"])

# ---------- CLIP 文本塔 ----------
import torch
from transformers import AutoProcessor, CLIPModel
clip_path = str(vb.MODEL_CACHE / vb.MODELS["clip"])
cm = CLIPModel.from_pretrained(clip_path, local_files_only=True).eval().to(vb.DEVICE)
tok = AutoProcessor.from_pretrained(clip_path, local_files_only=True).tokenizer

ID_PROMPTS = [
    "a photo of a chinese convenience store chain storefront",
    "a storefront of a chain convenience store",
    "a supermarket chain storefront sign",
    "a snack food chain store front",
    "a chain store signboard on a street",
    "a chain retail shop exterior with a brand sign",
]
OOD_PROMPTS = [
    "a photo of a coffee shop storefront",
    "a bubble tea drink shop storefront",
    "a milk tea shop storefront",
    "a bakery storefront",
    "a fresh food market storefront",
    "a butcher shop storefront",
    "a gas station shop",
    "a fast food restaurant storefront",
    "a small local independent shop",
    "a restaurant storefront",
]

@torch.no_grad()
def text_emb(prompts):
    ti = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(vb.DEVICE)
    t = cm.get_text_features(**ti)
    return (t / t.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

E_id, E_ood = text_emb(ID_PROMPTS), text_emb(OOD_PROMPTS)

# ---------- 特征 ----------
l2 = e8.l2
Gal_c = l2(Vc[gal_named]); Gal_d = l2(Vd[gal_named])

def feats(Vc_, Vd_):
    c = l2(Vc_); d = l2(Vd_)
    fg_c = (c @ Gal_c.T).max(axis=1)
    fg_d = (d @ Gal_d.T).max(axis=1)
    fid = (c @ E_id.T).max(axis=1)
    food = (c @ E_ood.T).max(axis=1)
    return np.vstack([fg_c, fg_d, fid, food, food - fid]).T  # 5 特征

names = ["gal_clip", "gal_dino", "txt_id", "txt_ood", "ood_margin"]
cal_id = feats(Vc[gal_named], Vd[gal_named])
cal_ood_mask = np.array([b not in ("瑞幸咖啡", "库迪咖啡") for b in v18_brands])
cal_ood = feats(Vc18[cal_ood_mask], Vd18[cal_ood_mask])
test_id = feats(Vc[test_named], Vd[test_named])
test_ood_mask = np.array([b in ("瑞幸咖啡", "库迪咖啡") for b in v18_brands])
test_ood = feats(Vc18[test_ood_mask], Vd18[test_ood_mask])
gal_named = [i for b in brands if b != "local" for i in gidx[b]]
# 诚实的已知校准/测试拆分: 非gallery行按门店分 60/40 (gallery行自相似=1.0, 不能当校准已知)
from collections import defaultdict
store_rows = defaultdict(list)
for i in test_named:
    store_rows[okrows[i]["poi"]].append(i)
stores_all = sorted(store_rows)
r2 = np.random.RandomState(1)
r2.shuffle(stores_all)
cal_stores = set(stores_all[:int(len(stores_all) * 0.6)])
cal_known_idx = [i for s in cal_stores for i in store_rows[s]]
test_known_idx = [i for s in store_rows if s not in cal_stores for i in store_rows[s]]

print(f"校准: 已知 {len(cal_known_idx)}, 未知 {int(cal_ood_mask.sum())} | 测试: 已知 {len(test_known_idx)}, 未见未知(瑞幸/库迪) {int(test_ood_mask.sum())}")

def auroc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

cal_id = feats(Vc[cal_known_idx], Vd[cal_known_idx])
test_id = feats(Vc[test_known_idx], Vd[test_known_idx])
cal_ood = feats(Vc18[cal_ood_mask], Vd18[cal_ood_mask])
test_ood = feats(Vc18[test_ood_mask], Vd18[test_ood_mask])
names = ["gal_clip", "gal_dino", "txt_id", "txt_ood", "ood_margin"]

print("\n特征        AUROC(cal/测)   FPR@95TPR(cal/测)")
for j, nm in enumerate(names):
    if nm in ("txt_id", "txt_ood", "ood_margin"):
        continue
    s_ood_cal, s_id_cal = cal_ood[:, j], cal_id[:, j]
    s_ood_te, s_id_te = test_ood[:, j], test_id[:, j]
    a = auroc(s_ood_cal, s_id_cal)
    thr = float(np.quantile(s_ood_cal, 0.05))
    fpr = float((s_id_cal <= thr).mean())
    at = auroc(s_ood_te, s_id_te)
    thr_t = float(np.quantile(s_ood_te, 0.05))
    fpr_t = float((s_id_te <= thr_t).mean())
    print(f"{nm:10s} {a:.3f} / {at:.3f}   {fpr:.3f} / {fpr_t:.3f}")

j = names.index("gal_clip")
print("\ngal_clip 分位数(已知cal | OOD cal | 已知test | 瑞幸+库迪):")
for q in (0.01, 0.05, 0.5):
    print(f"  q{q}: {np.quantile(cal_id[:, j], q):.3f} | {np.quantile(cal_ood[:, j], q):.3f} | "
          f"{np.quantile(test_id[:, j], q):.3f} | {np.quantile(test_ood[:, j], q):.3f}")
zn = np.array(v18_brands)[test_ood_mask]
for b in ("瑞幸咖啡", "库迪咖啡"):
    m = zn == b
    print(f"  {b}: q5={np.quantile(test_ood[m, j], 0.05):.3f} q50={np.quantile(test_ood[m, j], 0.5):.3f}")

# 阈值策略A: OOD校准5%分位; 策略B: 已知校准1%分位
for tag, thr in (("A_ood_q5", float(np.quantile(cal_ood[:, j], 0.05))),
                 ("B_known_q1", float(np.quantile(cal_id[:, j], 0.01)))):
    ur_te = float((test_ood[:, j] <= thr).mean()) * 100
    fpr_te = float((test_id[:, j] <= thr).mean()) * 100
    fpr_cal = float((cal_id[:, j] <= thr).mean()) * 100
    print(f"策略{tag} thr={thr:.3f}: 校准KnownFPR {fpr_cal:.1f}%, 未见UnknownRecall {ur_te:.1f}%, 已知FPR(测) {fpr_te:.1f}%")
    np.savez(f"{ROOT}/synthetic/vis_cache/s1_ood_head_{tag}.npz", thr=np.array([thr]))
