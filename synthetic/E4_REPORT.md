# E4 局部 ROI 多向量匹配报告

实现 2026-11-11。设计文档 §7.1 B4 / §9.1 E4："多向量局部信息是否必要？"
脚本：`synthetic/e4_local_match.py`（E4_VERSION=`e4-local-v1.0`），DINOv2-base 冻结 patch tokens（16×16 网格，256×768）。

## 1. 方法

| 变体 | 说明 |
|---|---|
| gem | patch GeM(p=3) 池化单向量——"类相关 ROI 全局描述子"强基线（§7.2 要求） |
| maxsim | ColBERT 式：query 每 patch 取参考最大余弦，再对 patch 均值 |
| maxsim_top50/25 | 仅取 query 判别性最强 50%/25% patch |
| spatial_w2/w4 | MaxSim + 空间窗约束（参考 patch 行列距 ≤2/≤4） |

双视图 full/crop；L1 LOSO 与 E3 完全同协议（store_id 分组、k=5）。全部变体全报告，不事后挑选（§9.2）。确定性：无 RNG。

## 2. 结果（30 正例 LOSO，探索性）

| 变体（full 视图） | R@1 | R@5 | 投票 |
|---|---|---|---|
| **E3 全局 CLS 对照** | **0.400** | **0.767** | **0.400** |
| gem（强基线） | 0.333 | 0.667 | 0.233 |
| maxsim | 0.400 | 0.767 | 0.300 |
| maxsim_top50 | 0.333 | 0.667 | 0.300 |
| maxsim_top25 | 0.233 | 0.533 | 0.233 |
| spatial_w2 | 0.300 | 0.700 | 0.267 |
| spatial_w4 | 0.400 | 0.767 | 0.400 |

crop 视图全部更差（R@1 ≤0.167）。GeM L0 品牌：full 召回 26.7%/特异度 97.3%；crop 26.7%/99.3%。

## 3. 结论：多向量局部信息在本数据规模下非必要

1. **MaxSim 不优于全局 CLS**：R@1/R@5 完全持平（0.400/0.767），投票反而下降（0.300 vs 0.400）。
2. **空间约束无增益**：w4 仅追平 CLS，w2 更差——招牌内 patch 空间对应假设在真实透视/遮挡下不成立。
3. **GeM 强基线反而更差**：池化丢失判别 patch，说明 CLS 已隐含关键局部信息。
4. **top-k 截断有害**：丢弃"低分"patch 同时丢掉了覆盖多样性。

**E4 判定**：不采用多向量路线（节省推理成本）。细粒度瓶颈不在表征粒度，而在**数据**（薄簇 CS-02/06 各 1 张、簇间结构相似）——与 E1/E3 结论一致。后续资源应投向数据扩充（3D 孪生合成，见 M1_3D_REPORT.md / E8）而非更复杂的匹配器。

## 4. 治理

探索性探针（权利 pending_review，G0/G1 未过）；候选簇真值非金标准；无训练、无超参事后选择。

## 5. 产物

`e4_local_match.py`、`vis_cache/patches_{full,crop}.npz`、`vis_cache/e4_results_*.json`、`vis_cache/e4_summary.json`、`vis_cache/e4_manifest.json`。

```bash
TRANSFORMERS_OFFLINE=1 python3 e4_local_match.py --all
```