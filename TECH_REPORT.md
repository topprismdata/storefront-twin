# 左邻右舍门头识别:基线与证据层技术报告

*自动生成 2026-11-13 03:54 by synthetic/gen_tech_report.py;全部数字引自冻结 JSON,可复现命令见 §7。*

## 1. 摘要与声明边界

- 任务: 门店门头图像中的"左邻右舍"品牌识别(检索 + 验证),真实照片与硬负例混合语料。
- 当前最优运行点: **E8-G 冻结融合 F1 0.862**(P 0.800 / R 0.933,thr 0.636,330 张盲测)。
- VLM 四态证据层(E10): v1 模式 B F1 0.951;v2 加强指令 F1 0.951;**GEN 独立泛化集上模式 A 0.917 > 模式 B 0.909,排名跨集翻转**,全部配对差异跨零(330 与 GEN 均然)。
- 三项不可豁免门槛: ① 全部评估为**探索性**(rights=pending_review, G0/G1 未过);② 模式间差异无统计证据,不得宣布某模式"最优";③ F1 0.86–0.95 为合成+采集域指标,**不得**解读为真实业务准确率。

## 2. 数据与治理

| 库 | 规模 | 用途 | 权利 |
|---|---|---|---|
| registry_v3 | 499 行 / 330 代表(sha 去重) | 训练/评估语料 | 真实图 pending_review;合成 self_created |
| registry_gen | 独立泛化集 223 图/148 店组(110 正) | 跨采集源泛化评测,sha 级独立于 330 | 真实图 pending_review |
| renders_3d | 820 (正 360/硬负 300/竞品 96/无招牌 40/异常 24) | 合成域:融合先验+硬负模板 | self_created synthetic_train_only |
| stock_benchmark | 220 (B 素材/R 实拍双通道标注+仲裁) | 素材域专项负例 | 内部评估 |
| 金标准 gold | 400 (全库覆盖+边界全采) | 人工双盲标注(协议+Kappa 工具就绪) | 内部评估 |

金标准组成: negative/render_stock×295, negative/same_name_unrelated×18, negative/indoor×18, negative/confusable_brand×11, positive/CS-04×11, positive/CS-03×10, negative/competitor×10, negative/logo×8 …;正例变体覆盖 {'CS-05': 3, 'CS-03': 10, '': 6, 'CS-01': 4, 'CS-04': 11, 'CS-06': 1, 'CS-02': 1}。

## 3. 系统架构

```
L0 检索层:  CLIP / DINOv2 / SigLIP 嵌入 (cropfull + brand-cluster 检索)
L4 局部验证: patch 匹配 (patch-MaxSim 空间几何)
E8-G 融合:  s = 0.4·s_clip + 0.4·s_dino_mc + 0.2·s_probe, thr 0.636 (严格 >)
E10 证据层: VLM 四态盲判 (MATCH/NON_MATCH/UNKNOWN/OUT_OF_SCOPE)
  模式 A: 拒识即拒(UNKNOWN→按 E8-G 原判)   模式 B: UNKNOWN 回退 E8-G 原判
```

kNN 检索保持 store_id LOSO;探针仅在合成域训练;VLM 判决仅作推理证据,不作训练标签。

## 4. 结果(measured)

### 4.1 E8-G 冻结基线(330 张,30 正)
| 系统 | P | R | F1 | TP/FP/FN | 备注 |
|---|---|---|---|---|---|
| E8-G 冻结阈值 | 0.800 | 0.933 | 0.862 | 28/7/2 | thr 0.636 |
| 嵌套 LOSO(322 折阈值重选) | 0.778 | 0.933 | 0.848 | 28/8/2 | thr_std=0.0 |
| store 级 bootstrap | 0.774 [0.619,0.911] | 0.934 [0.838,1.000] | 0.844 [0.737,0.932] | — | B2000, 24 正店,CI 触 1.0 |

AuPR 0.925;P@95R 0.707;R@95P 0.600。通道 AuPR: clip 0.659 / dino_mc 0.766 / probe 0.324。

### 4.2 E10 证据层(v1 全盲 330)
| 系统 | P | R | F1 | TP/FP/FN | 95% CI(store, B2000) |
|---|---|---|---|---|---|
| VLM-alone | 0.966 | 0.933 | 0.949 | 28/1/2 | — |
| 模式 A | 0.964 | 0.900 | 0.931 | 27/1/3 | -–- |
| 模式 B | 0.935 | 0.967 | 0.951 | 29/2/1 | -–- |

裁决分布(v1): MATCH 29 / NON_MATCH 161 / UNKNOWN 32 / OUT_OF_SCOPE 108;E10 在 E8-G 正例上拒识 1TP+7FP;3/330 冲突全分辨率仲裁,注册表标签全对。

### 4.3 v2 加强指令复测(330/330)
| 系统 | P | R | F1 | TP/FP/FN |
|---|---|---|---|---|
| VLM-alone | 0.900 | 0.964 | 0.931 | 27/3/1 |
| 模式 A | 0.900 | 0.964 | 0.931 | 27/3/1 |
| 模式 B | 0.967 | 0.935 | 0.951 | 29/1/2 |

**证伪记录**: §6.1 曾据 13/13 定向回归预测 v2 模式 B F1≈0.966,全量复测未兑现(0.951)——13/13 是回归测试不是泛化验证。裁决迁移矩阵主项: NON_MATCH→UNKNOWN ×68(加强"两证据门槛"使 VLM 更保守,UNKNOWN 回退 E8-G 兜底,模式 B 的 R 由回退机制保住)。
配对 bootstrap: B−A +0.02 [-0.04,+0.08] 显著=false;B−VLM +0.02 显著=false;A−VLM +0.00 显著=false。**没有任何模式差异统计显著。**

### 4.4 GEN 独立泛化集(223 图/148 店组,sha 级独立)
| 系统 | P | R | F1 | TP/FP/FN |
|---|---|---|---|---|
| E8-G(冻结 thr) | 0.828 | 0.964 | 0.891 | 106/22/4 |
| VLM-alone | 0.939 | 0.836 | 0.885 | 92/6/18 |
| 模式 A | 0.934 | 0.900 | 0.917 | 99/7/11 |
| 模式 B | 0.909 | 0.909 | 0.909 | 100/10/10 |

**排名跨集翻转**: GEN 上模式 A(0.917) > 模式 B(0.909),与 330 的 v1(B>A)相反;全部配对差跨零(B−A -0.008 [-0.026,+0.010])。GEN 上 VLM 变体误拒显著(FN 18,最大模糊边界=门店样式变体)。E10 对基线的提升在真实泛化域缩水至 +0.026。

### 4.5 素材域专项(stock_benchmark 220)
| 检查 | 结果 |
|---|---|
| E8-G 代理误收(两通道 0.5/0.5) | **8/220 = 3.6%** @thr 0.636 |
| 双通道标注一致率 | 80.0%(220 张),44 分歧全分辨率仲裁后 B201+R19 |
| 素材搜索污染率(实拍混入) | 3.2% |
| VLM 四态层 | 待补(ChatGPT 附件通道灰度故障,网格已就绪) |

残余误收集中于"含品牌风格字体立体字的设计图"——与真实域 CS-04 立体发光字变体同型,是变体边界问题的素材域镜像。

## 5. 已知失效模式

1. **变体边界**: CS-04 立体发光字曾致 VLM 误拒(img_0483),设计图立体字致 E8-G 误收(stock stk_0471 等 8 张)——同一光谱两端。
2. **13/13 教训**: 定向回归集上的修复效果不可外推,任何指令/阈值改动必须全量复测(§4.3)。
3. **统计功效**: 30 正例/24 正店,CI 触 1.0;模式差异不可分辨是功效问题,不是证明等价。
4. **缩略图**: 512px 缩略图丢失关键文字(VLM 高清读取>本地缩略核查,img_0022 判例)。

## 6. 结论

- 架构有效: 检索+局部验证+融合基线(0.862)经 VLM 证据层提升至 0.951(点估计),素材域误收 3.6%。
- 全部结论受 §1 三门槛约束;权利审核(G0/G1)与人工金标准(400 张,协议就绪)是进入正式声明的先决条件。

## 7. 复现

| 产物 | 命令/来源 |
|---|---|
| E8-G 冻结分数 | synthetic/vis_cache/e8g_frozen_scores.npz;协议 e8g_protocol.json |
| 统计(nested LOSO+bootstrap) | synthetic/e8g_stats.py → e8g_stats.json |
| E10 v1 全盲 | synthetic/e10_full_gen.py(seed 2026)→ verdict_batch01-14.md → e10_full_reconcile.py |
| v2 复测 | v2_verdict_batch01-14.md → v2 前缀脚本 → v2_reconciliation.json |
| 配对 bootstrap | e10_full_bootstrap.py(v1/v2_paired_bootstrap.json) |
| 素材基准 | data/stock_benchmark.json;代理评分 stock_e8g_results.json |
| 金标准 | synthetic/gold_sample.py → annotation_pack_v2/gold/;synthetic/kappa_tool.py |
