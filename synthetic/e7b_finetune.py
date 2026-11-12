#!/usr/bin/env python3
"""
E7b — DINOv2 末块微调（打破线性探针 0.81-0.85 平台）
=====================================================
数据: gold(644, w2.0) + pool 弱标签 A(w0.5)/B/C(w1.0) + render3d(w0.7) + registry_extra(w1.0)
守卫: 训练集与 330 评测集零交集（已验证）
划分: poi 组 85/15 train/val; 阈值只从 val 冻结; 330 只做最终一次性评测
模型: 本地 facebook_dinov2-base, 解冻 encoder.layer[11] + layernorm + 线性头(CLS)
"""
import json, sys, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, "/Users/mac/chatgpt-harvest/6af41562/synthetic")
import e6_real_probe as e1
import e6_real_probe_v2 as e2
import e7_gold_train as e7

ROOT = Path("/Users/mac/chatgpt-harvest/6af41562")
E6, DATA, CACHE = e1.E6, e1.DATA, e1.CACHE
MODEL_DIR = ROOT / "models/facebook__dinov2-base"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 2026

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(path):
    im = Image.open(path).convert("RGB")
    im = im.resize((224, 224), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div_(255.0)
    return (x - MEAN) / STD


def build_dataset():
    rows = []
    for g in e7.load_gold():
        rr = dict(g); rr["w"] = 2.0
        rows.append(rr)
    gold_ids = {r["image_id"] for r in rows}
    for r in e2.build_pool_v2():
        if r["image_id"] in gold_ids:
            continue
        rr = dict(r)
        rr["lab"] = "N" if r["lab"] != "A" else "A"
        rr["w"] = 0.7 if r["src"] == "render3d" else (0.5 if r["src"] in ("gen_full", "gen_x1") and r["lab"] == "A" else 1.0)
        rr["poi"] = str(r.get("poi") or r.get("group") or r["image_id"])
        rows.append(rr)
    # 守卫: 与 330 零交集
    tk = json.load(open(CACHE / "e10_full/truth_key.json"))
    ev = {r["image_id"] for r in (tk if isinstance(tk, list) else list(tk.values()))}
    bad = [r["image_id"] for r in rows if r["image_id"] in ev]
    assert not bad, f"训练集泄漏评测集: {bad[:5]}"
    return rows


class Net(nn.Module):
    def __init__(self, model_dir):
        from transformers import Dinov2Model
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(str(model_dir))
        for p in self.backbone.parameters():
            p.requires_grad = False
        blk = self.backbone.encoder.layer[-1]
        for p in blk.parameters():
            p.requires_grad = True
        for p in self.backbone.layernorm.parameters():
            p.requires_grad = True
        self.head = nn.Linear(768, 1)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        return self.head(out.last_hidden_state[:, 0]).squeeze(-1)


@torch.no_grad()
def score(model, paths, bs=32):
    """训练后的头打分(标量 logit);用于 val/评测。"""
    model.eval()
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([load_image(p) for p in paths[i:i + bs]]).to(DEV)
        out.append(model(x).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


@torch.no_grad()
def embed(model, paths, bs=32):
    model.eval()
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([load_image(p) for p in paths[i:i + bs]]).to(DEV)
        v = model.backbone(pixel_values=x).last_hidden_state[:, 0]
        out.append(v.cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def run(rows, tag):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    pois = sorted({r["poi"] for r in rows})
    random.shuffle(pois)
    val_p = set(pois[: max(1, int(len(pois) * 0.15))])
    tr = [r for r in rows if r["poi"] not in val_p]
    va = [r for r in rows if r["poi"] in val_p]
    ytr = np.array([1 if r["lab"] == "A" else 0 for r in tr])
    yva = np.array([1 if r["lab"] == "A" else 0 for r in va])
    wtr = np.array([r["w"] for r in tr], dtype=np.float32)
    print(f"[{tag}] train={len(tr)} (pos {ytr.sum()}) val={len(va)} (pos {yva.sum()}) pois={len(pois)} dev={DEV}", flush=True)

    model = Net(MODEL_DIR).to(DEV)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "layer.11" in n or "layernorm" in n], "lr": 5e-5},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    EPOCHS, BS = 4, 16
    steps = math.ceil(len(tr) / BS) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-5, 1e-3], total_steps=steps)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    idx = list(range(len(tr)))
    best_auc, best_state = -1, None
    from sklearn.metrics import roc_auc_score
    g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        model.train()
        random.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            x = torch.stack([load_image(tr[j]["path"]) for j in b]).to(DEV)
            y = torch.tensor(ytr[b], dtype=torch.float32, device=DEV)
            w = torch.tensor(wtr[b], dtype=torch.float32, device=DEV)
            logits = model(x)
            loss = (lossf(logits, y) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(b)
        ev_logits = score(model, [r["path"] for r in va])
        s = ev_logits
        auc = roc_auc_score(yva, s)
        thrs = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
        f1s = []
        for t in thrs:
            p = s > t
            f1s.append(2 * ((p & (yva == 1)).sum()) / max(1, 2 * ((p & (yva == 1)).sum()) + ((p & (yva == 0)).sum()) + ((~p & (yva == 1)).sum())))
        bi = int(np.argmax(f1s))
        print(f"  ep{ep}: loss={tot/len(tr):.4f} valAUC={auc:.3f} valF1={f1s[bi]:.3f}@thr={thrs[bi]:.3f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr, best_f1 = float(thrs[bi]), float(f1s[bi])
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), E6 / "e7b_model.pt")
    return model, {"val_auc": round(best_auc, 3), "val_f1": round(best_f1, 3), "val_thr": round(best_thr, 4)}


def eval330(model, thr):
    from sklearn.metrics import average_precision_score
    tk = json.load(open(CACHE / "e10_full/truth_key.json"))
    tr = tk if isinstance(tk, list) else list(tk.values())
    s = score(model, [r["path"] for r in tr])
    vecs = embed(model, [r["path"] for r in tr])  # 768维留作融合素材
    yy = np.array([r["label"] == "positive" for r in tr])
    pred = s > thr
    tp = int((pred & yy).sum()); fp = int((pred & ~yy).sum()); fn = int((~pred & yy).sum())
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    res = {"P": round(P, 3), "R": round(R, 3), "F1": round(2 * P * R / max(1e-9, P + R), 3),
           "AuPR": round(float(average_precision_score(yy, s)), 3),
           "confusion": [tp, fp, fn], "thr": thr}
    print("330 评测:", res, flush=True)
    np.savez(E6 / "e7b_full.npz", ids=[r["image_id"] for r in tr], vecs=vecs)
    return res


if __name__ == "__main__":
    rows = build_dataset()
    print("dataset:", len(rows))
    model, vres = run(rows, "e7b")
    r330 = eval330(model, vres["val_thr"])
    out = {"tag": "e7b_dinov2_lastblock", "val": vres, "eval330": r330,
           "reference": {"e8g": {"P": 0.778, "R": 0.933, "F1": 0.848, "AuPR": 0.926},
                          "e7_probe_mixed": {"P": 0.765, "R": 0.867, "F1": 0.812, "AuPR": 0.866}}}
    json.dump(out, open(E6 / "e7b_results.json", "w"), ensure_ascii=False, indent=1)
    print("E7b 完成 -> e7b_results.json / e7b_model.pt / e7b_full.npz", flush=True)
