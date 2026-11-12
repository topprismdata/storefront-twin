# 文献深查报告：门头识别瓶颈的学术对照与修正路线

2026-11-12。用户不认可当前结论（"成绩差/近天花板"），要求查论文。本报告系统检索 2023–2025 文献，
**发现一处我方方法论错误（E4）、一处评估协议缺陷、三条有实证支撑的提升路线**。所有数字均来自原文。

## 1. 最重要发现：E4 结论是方法论错误，不是"多向量无用"

**AnyLoc（arXiv:2308.00688，RA-L 2023 / ICRA 2024，CMU/MIT）** 证明 DINOv2 局部特征聚合
相对 CLS 有巨大增益，但依赖三个我方 E4 全部用错的设计选择：

| 设计点 | AnyLoc 正确做法 | 我方 E4 做法 | 后果 |
|---|---|---|---|
| 特征层 | **中间偏深层**（ViT-G 第 31/40 层；性能峰值在中间层而非最后层，Fig 5b） | 最后层 last_hidden_state | 最后层过度全局化，丢局部判别 |
| facet | **value facet**（注意力 V 投影，背景对比度最高，Fig 2/5c） | token（残差流输出） | 判别对比度低 |
| 聚合 | **hard-assignment VLAD**（聚类残差+intra/inter 归一化） | MaxSim/GeM/空间窗 | 见下方原文消融 |

AnyLoc 原文消融（Table VII，DINOv2，Baidu/Oxford R@1）：
- GAP 41.6/78.5，GeM 50.1/92.2，**hard-VLAD 71.5/94.8**，soft-VLAD 40.3/82.2。
- CLS→GeM→VLAD 结构化环境平均 **+9%/+23%**，非结构化 **+2%/+18%**（§V-A5）。
- 域特定词表（domain-specific vocabulary）再提升（Table V：aerial 57.1→76.2）。
- 免训练、免微调——完全符合本项目治理约束。

**E4 方法论确实有误**（最后层 token + 朴素匹配 ≠ 文献配方），已实施 E9 修正实验。

### ⚠️ E9 实测修正（2026-11-12，诚实交代）

E9 初版报出 L1 R@1 88–90% 的"突破"，**经自查是 bug 伪影，已撤回**：
`pos_mask = np.array([label=="positive" and candidate_style ...])` 中 `and` 返回字符串而非布尔，
导致空簇标签样本混入 LOSO 参考集，R@1 虚高。修复为真布尔后：

| 方法 | L1 R@1 | L1 R@5 |
|---|---|---|
| E3 全局 CLS | 40–47% | 73–77% |
| E4 MaxSim/GeM | 17–40% | 53–77% |
| **E9 AnyLoc（修正后最佳 L6_value/vlad）** | **43.3%** | **83.3%** |

**结论：AnyLoc 聚合在本任务仅带来 R@5 小幅提升（76.7→83.3%），R@1 无突破。**
原因分析：AnyLoc 面向 VPR（同一地点不同视角的**实例级**匹配），门头簇识别是
不同门店共享 VI 的**类别级细粒度**——任务结构不同，文献增益不能直接迁移。
per_style 修正后合理：CS-04 0.727 / CS-05 1.0，薄簇 CS-01/02/06 仍 0（数据缺口）。

**教训**：① 文献方法必须实测验证，不能凭原文数字推断本任务收益；
② 评估代码的布尔/标签类型必须自检（本次 bug 差点写入错误结论）；
③ E4"多向量无增益"的结论在修正后基本成立（R@1 层面），仅 R@5 有边际改善。

## 2. 评估协议缺陷：固定 0.5 阈值报 P/R 误导

**Open-Set Recognition in the Age of VLMs（arXiv:2403.16528，Miller et al. 2024）**：
- CLIP 类模型在开集条件下**高召回时精确率必然崩**：ImageNet 开集测试 CLIP P@95R 仅 46.2%，
  R@95P 仅 3.4%（Table 1）。我方 CLIP 单独 P49%/R90% 与此完全一致——**这是 VLM 的已知系统性缺陷，不是本项目特有失败**。
- **Softmax/Entropy 不确定性优于原始 Cosine**（Table 1：CLIP AuPR 71.5/72.3 vs 61.3）。
  我方 kNN 投票分数是粗粒度相对度量，应改用参考集相似度 softmax 熵。
- 专用负嵌入（negative embeddings）可削减开集误报但有任务性能代价；随机词负查询对 ImageBind 有效
  （AuPR +5.5），对 CLIP 效果有限——**我方合成竞品难负例参考库正是"领域专用负嵌入"，比论文的随机负查询更对症**。
- 协议建议：报 **AuPR + P@95R + R@95P**，而非单点 P/R。我方"80%/93%"是探索集挑阈值的单点，
  学术上应报整条风险-覆盖率曲线与 AuPR。

## 3. 合成数据域差：文献证实我方 E8 失败模式，并给出修复方向

**Mind the Gap Between Synthetic and Real（arXiv:2405.03243）**：
- 纯合成训练的性能损失**集中在网络最后几层**；用极少量真实数据微调最后层即可恢复大部分性能。
  → 我方冻结骨干+合成探针恰好把域差全部压在探针层；**E6 少量真实监督（权利审核后）是文献认证的正解**。
- 合成数据综述（arXiv:2403.10075 等）：2024–2025 共识 = **扩散模型做 sim-to-real 外观翻译**
  （ControlNet/img2img 把 3D 渲染转真实质感），比纯域随机化有效。
  → M1-3D 渲染 + 扩散外观翻译 = 设计文档 M2 的现代替代，成本低于全 RTX。
- **原始数量不能弥合域差**（盲目扩量会把渲染系统误差固化进训练）——我方 520→820 扩量无增益与此一致。

## 4. 少样本适配：Tip-Adapter 是我方 kNN 的原则化升级

**Tip-Adapter（arXiv:2207.09519，ECCV 2022）**：免训练 cache-model 适配 CLIP，
logits = α·cache检索 + CLIP零样本先验，16-shot 即达训练法水平。
→ 我方 kNN 投票是其粗糙版本；Tip-Adapter 提供温度化检索+先验融合的原则形式，且支持 cache 微调（合成 cache 可微调）。

**TPT/DiffTPT（NeurIPS 2022 / ICCV 2023）**：测试时单样本熵最小化 prompt 调优，
分布偏移下平均 +3.6%/+5.13%。→ 可作真实域测试时适配，但需 CLIP 文本端，收益中等。

## 5. VLM 比较式识别：解决 same_name_unrelated 的学术路线

**Logo-VGR（arXiv:2509.25811，ByteDance/南开 2025）**：
- 把 logo 识别重构为**比较任务**（query 图 vs 候选 logo 参考，含"以上都不是"选项），
  而非直接生成品牌标签——避免记忆品牌分布的捷径学习。
- Qwen2.5-VL-7B 零样本 OOD 81.72% → Logo-VGR 91.98%（+10.26）；3B 版 OOD +14.48。
- 关键：**少样本监督下学"推理过程"比学"答案"泛化好得多**（SFT 提 ID 但降 OOD）。
→ 我方剩余 FP（同名异业/竞品渲染图）正是"比较+拒识"任务：给 VLM 官方 E2 logo 参考 + query，
  问"是否同一品牌 VI，给出坐标证据"。本机 oMLX 无 VL 模型，可走中转 Chrome ChatGPT 视觉通道
  （本项目已有成熟管线）做 E10 探针；或下载 Qwen2.5-VL-3B/7B MLX（需带宽）。

## 6. 细粒度稠密匹配：DINOv2 免训练参考匹配已有成熟范式

Matcher 系（arXiv:2504.15669 等 2025）：冻结 DINOv2 patch 特征做 support-query 稠密相似度 +
双向剪枝，免训练少样本分割/匹配媲美训练法。→ 招牌 ROI 内做参考图稠密匹配（而非全图），
与 AnyLoc 聚互补：聚合管 L0 检索，稠密匹配管 L1 细粒度与同名异业判别。

## 7. 修正后的实验优先级（按证据强度/成本）

| 优先级 | 实验 | 文献依据 | 成本 | 治理 |
|---|---|---|---|---|
| ~~P0~~ 已测 | E9：AnyLoc 式聚合 → **R@1 无突破（43%），R@5 +6pp**；初版 90% 是 bug 伪影已撤回 | 2308.00688 | 已完成 | 免训练 ✓ |
| P1 | 评估协议升级：AuPR/P@95R/R@95P + softmax 熵不确定性 | 2403.16528 | 极低 | ✓ |
| P2 | E10：VLM 比较式拒识（E2 参考 vs query，坐标证据） | 2509.25811 | 中（relay 通道） | 探索性 ✓ |
| P3 | Tip-Adapter 原则化 cache 融合 | 2207.09519 | 低 | ✓ |
| P4 | 扩散外观翻译 M1 渲染（M2 现代替代） | 2405.03243/2403.10075 | 高（需扩散模型） | self_created ✓ |
| P5 | E6 少量真实监督（最后层微调） | 2405.03243 | 权利审核后才可 | 待 G0/G1 |

## 8. 对"成绩差"的重新定性

文献对照后，我方成绩应重新表述：
- CLIP P49/R90 **不是失败，是 VLM 开集已知缺陷**（2403.16528 实测 P@95R≈46%），且我方融合已把它修到 P80/R93。
- E4"多向量无用"在 R@1 层面**修正后基本成立**；AnyLoc（E9）仅 R@5 边际改善。文献 VPR 增益不迁移到类别级门头细粒度。
- 合成训练失败**与文献一致**（域差集中在末层），不是路线错误，是缺最后层真实锚点。
- 真正的天花板在**数据权利与薄簇**；模型侧未兑现红利主要是 **E10 VLM 比较式拒识**（解决同名异业 FP）与 **E6 真实监督**（待权利审核）。
- **本轮最大收获是方法论自检**：发现并修复了差点写入错误结论的布尔/标签 bug，确立"文献方法必须本任务实测"纪律。

## 引用

1. Keetha et al. AnyLoc: Towards Universal Visual Place Recognition. arXiv:2308.00688 (RA-L 2023). https://arxiv.org/abs/2308.00688
2. Izquierdo & Civera. Optimal Transport Aggregation for Visual Place Recognition (SALAD). CVPR 2024. arXiv:2311.15937. https://arxiv.org/abs/2311.15937
3. Miller et al. Open-Set Recognition in the Age of Vision-Language Models. arXiv:2403.16528. https://arxiv.org/abs/2403.16528
4. Mind the Gap Between Synthetic and Real. arXiv:2405.03243. https://arxiv.org/abs/2405.03243
5. A Survey of Synthetic Data Augmentation Methods in Computer Vision. arXiv:2403.10075. https://arxiv.org/abs/2403.10075
6. Zhang et al. Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification. ECCV 2022. arXiv:2207.09519. https://arxiv.org/abs/2207.09519
7. Shu et al. Test-Time Prompt Tuning for Zero-shot Generalization in Vision-Language Models. NeurIPS 2022. https://azshue.github.io/TPT
8. Feng et al. Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning (DiffTPT). ICCV 2023.
9. Liang et al. Logo-VGR: Visual Grounded Reasoning for Open-world Logo Recognition. arXiv:2509.25811. https://arxiv.org/abs/2509.25811
10. DINOv2-powered Few-Shot Semantic Segmentation (Matcher 系). arXiv:2504.15669. https://arxiv.org/abs/2504.15669