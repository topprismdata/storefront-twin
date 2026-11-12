# E1/E3 视觉表征基线报告（主力路线）

实现 2026-11-11。设计文档 §7.1 B1/B2/B3 / §9.1 E1/E3：CLIP / SigLIP / DINOv2 冻结编码器 + FAISS 检索。
基线：`synthetic/visual_baseline.py`（BASELINE_VERSION=`e1e3-visual-v1.0`），MPS 加速。

> **架构定位**（见 E0_REPORT.md §4 文献纠偏）：品牌/门头识别属视觉检索（LogoDet-3K），主力是视觉嵌入而非 OCR。本基线验证该路线。

## 1. 方法

| 项 | 配置 |
|---|---|
| 模型 | DINOv2-base（768d）/ CLIP-ViT-B/32（512d）/ SigLIP-base-patch16（768d），冻结 |
| 视图 | full（全图）+ crop（E0 投影检测器取招牌框，失败则中心裁剪） |
| 索引 | FAISS IndexFlatIP（精确余弦） |
| L0 品牌鉴别 | query → 参考集 kNN（k=5）投票，排除同门店组 |
| L1 候选簇检索 | **留一店交叉验证（LOSO）**：query 店从其余店正例参考集检索，Recall@1/@5 + 投票 |
| 隔离 | store_id 分组（24 组），无 store_id 正例各自独立组（§9.2） |

## 2. 结果（30 正例 / 300 负例，探索性探针）

### L0 品牌鉴别

| 模型/视图 | 品牌召回 | 特异度 |
|---|---|---|
| **CLIP / full** | **90.0%** | 90.7% |
| SigLIP / full | 86.7% | 92.0% |
| SigLIP / crop | 73.3% | 95.7% |
| DINOv2 / crop | 43.3% | 96.3% |
| CLIP / crop | 40.0% | 95.0% |
| DINOv2 / full | 20.0% | 94.3% |
| （E0 OCR 对照） | 6.7% | 100% |

### L1 候选簇检索（LOSO）

| 模型/视图 | R@1 | R@5 | 投票准确率 |
|---|---|---|---|
| CLIP / full | 46.7% | 73.3% | 43.3% |
| DINOv2 / full | 40.0% | 76.7% | 40.0% |
| SigLIP / full | 40.0% | 73.3% | 36.7% |
| CLIP / crop | 36.7% | 76.7% | 36.7% |
| SigLIP / crop | 30.0% | 63.3% | 33.3% |
| DINOv2 / crop | 16.7% | 76.7% | 16.7% |

### 模型融合（CLIP 语义 + DINOv2 结构互补）

| 策略 | 品牌召回 | 特异度 |
|---|---|---|
| CLIP/full 单独 | 90.0% | 90.7% |
| **OR 融合**（CLIP/full ∨ DINOv2/crop） | **96.7%** | 88.0% |
| 簇 R@5 并集（CLIP/full ∪ DINOv2/full） | **80.0%** | — |
| 三模型 OR（CLIP/full ∨ SigLIP/full ∨ DINOv2/crop） | 96.7% | 86.0% |

## 3. 核心发现

1. **视觉路线碾压 OCR**：CLIP 全图品牌召回 90% vs E0 OCR 6.7%（13 倍）。证实文献纠偏正确——品牌识别主力是视觉嵌入，不是 OCR。
2. **CLIP 语义品牌强，DINOv2 结构细粒度强**：CLIP 在网络图文上训练见过大量品牌/店面，L0 品牌鉴别最优（90%）；DINOv2 纯自监督无品牌语义（L0 仅 20%），但局部结构特征使 L1 簇 R@5 与 CLIP 相当（76.7%）。**两者互补**，OR 融合品牌召回 96.7%。
3. **视图效应**：CLIP 全图 > 裁剪（品牌语义需上下文）；DINOv2 裁剪 > 全图（聚焦招牌结构）。
4. **细粒度仍是瓶颈**：L1 簇 R@1 仅 40–47%，CS-01（编号标准）/CS-02（书法）/CS-06（超市）R@1=0%。原因：薄簇样本极少（CS-02/06 各 1 张，LOSO 无同簇参考）+ 簇间结构相似。CS-05（PLUS）R@1=100% 因特征鲜明。
5. **高召回但特异度有代价**：CLIP 90% 召回伴随 9.3% 误报（28/300 负例误判品牌）；AND 融合特异度 99% 但召回骤降。需 §7.4 选择性预测校准阈值。

## 4. 对后续的指引

- **主力 backbone 选 CLIP（L0 品牌，90%）+ DINOv2（L1 局部结构）融合**；SigLIP（86.7%）可作 CLIP 替代/集成成员。非单一模型或 OCR。
- **细粒度提升路径**：① 补薄簇真实样本（CS-02/06）；② DINOv2 patch 局部 ROI 匹配（E4，设计文档 B4）而非全局 CLS；③ 轻量度量学习（E6/B6，同簇跨场景正对 + 异簇同场景难负对）。
- **OCR 仅作消歧辅助**：CS-01 vs CS-03 若视觉不可分，用编号 OCR 消歧（§7.2 s_ocr）。
- **合成数据用途**：M0 合成可作 L1 簇检索的参考扩充（薄簇 CS-02/06），但须 E7/E8/E9 验证真实域增益。
- **拒识校准**：在独立验证集按错误代价选阈值，报告风险—覆盖率曲线（§7.4），不据测试集调参。

## 5. 治理与缺口

- **探索性探针**：真实图权利 pending_review，G0/G1 未过；30 正例样本量不足以发布 Macro-F1 高置信结论（§9.2 INCONCLUSIVE）。
- 候选簇真值是视觉模型聚类（candidate_unverified），非金标准。
- LOSO 下薄簇（CS-02/06 各 1 张）无同簇参考，R@1=0% 是数据缺口而非模型缺陷。

## 6. 产物与复现

- `visual_baseline.py`：嵌入 + FAISS + LOSO 评估
- `vis_cache/{model}_{view}.npz`：嵌入缓存
- `vis_cache/results_*.json`：逐样本预测
- `vis_cache/summary.json` / `eval_manifest.json`

```bash
cd ~/chatgpt-harvest/6af41562/synthetic
TRANSFORMERS_OFFLINE=1 python3 visual_baseline.py --all --models dinov2,clip,siglip
```