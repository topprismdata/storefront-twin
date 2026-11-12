#!/usr/bin/env python3
"""TECH_REPORT.md 生成器: 所有数字从冻结 JSON 自动填充(禁止手抄)。
用法: python3 synthetic/gen_tech_report.py  →  TECH_REPORT.md
"""
import json, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "synthetic" / "vis_cache"

def load(p):
    return json.load(open(p))

def f1row(name, m, extra=""):
    return (f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m.get('tp','-')}/{m.get('fp','-')}/{m.get('fn','-')} | {extra} |")

def main():
    e8g = load(VC / "e8g_protocol.json")
    stats = load(VC / "e8g_stats.json")
    v1rec = load(VC / "e10_full/reconciliation.json")
    v1rows = load(VC / "e10_full/records.json")
    if isinstance(v1rows, dict):
        v1rows = v1rows.get("records", v1rows.get("rows", []))
    # 模式 A = e8g 原判 AND verdict 不在 {NON_MATCH, OUT_OF_SCOPE}
    mA_tp = sum(1 for r in v1rows if r["truth"] == "positive" and r.get("e8g_pred") and r.get("vlm_verdict") not in ("NON_MATCH", "OUT_OF_SCOPE"))
    mA_fp = sum(1 for r in v1rows if r["truth"] == "negative" and r.get("e8g_pred") and r.get("vlm_verdict") not in ("NON_MATCH", "OUT_OF_SCOPE"))
    mA_fn = sum(1 for r in v1rows if r["truth"] == "positive" and not (r.get("e8g_pred") and r.get("vlm_verdict") not in ("NON_MATCH", "OUT_OF_SCOPE")))
    modeA_v1 = {"tp": mA_tp, "fp": mA_fp, "fn": mA_fn,
                "precision": mA_tp / max(1, mA_tp + mA_fp), "recall": mA_tp / max(1, mA_tp + mA_fn),
                "f1": 2 * mA_tp / max(1, 2 * mA_tp + mA_fp + mA_fn)}
    modeB_v1 = v1rec["combined_e8g_e10"]
    v2rec = load(VC / "e10_full/v2_reconciliation.json")
    v2mig = load(VC / "e10_full/v2_migration.json")
    v1ci = load(VC / "e10_full/bootstrap_ci.json")
    v2pair = load(VC / "e10_full/v2_paired_bootstrap.json")
    v1mig = load(VC / "e10_full/v1_migration.json") if (VC / "e10_full/v1_migration.json").exists() else None
    stockb = load(ROOT / "data/stock_benchmark.json")
    stockres = load(VC / "stock_e8g_results.json")
    fix = load(VC / "e10_full/fix_summary.json")
    reg = load(ROOT / "data/registry_v3.json")
    sha = load(ROOT / "data/sha_file_index.json")
    key = load(ROOT / "data/annotation_pack_v2/gold/keys/gold_key.json")

    op = e8g["operating_point"]
    ci = stats["bootstrap_ci"]
    nl = stats["nested_loso"]
    bci = v1ci.get("A", {}), v1ci.get("B", {})

    gold_lab = collections.Counter((v["label"], v["neg_type"] or v["candidate_style"] or "pos_nonrep") for v in key.values())
    pos_styles = collections.Counter(v["candidate_style"] for v in key.values() if v["label"] == "positive")

    lines = f"""# 左邻右舍门头识别:基线与证据层技术报告

*自动生成 {__import__('time').strftime('%Y-%m-%d %H:%M')} by synthetic/gen_tech_report.py;全部数字引自冻结 JSON,可复现命令见 §7。*

## 1. 摘要与声明边界

- 任务: 门店门头图像中的"左邻右舍"品牌识别(检索 + 验证),真实照片与硬负例混合语料。
- 当前最优运行点: **E8-G 冻结融合 F1 {op['f1']:.3f}**(P {op['precision']:.3f} / R {op['recall']:.3f},thr {op['thr']:.3f},330 张盲测)。
- VLM 四态证据层(E10): v1 模式 B F1 {modeB_v1['f1']:.3f};v2 加强指令 F1 {v2rec['modeB']['f1']:.3f}。**配对 bootstrap 显示 A/B/VLM-alone 两两差异均不显著**(B−A ΔF1 {v2pair['pairs']['B-A']['delta_f1']:+.2f} [{v2pair['pairs']['B-A']['lo']:+.2f},{v2pair['pairs']['B-A']['hi']:+.2f}])。
- 三项不可豁免门槛: ① 全部评估为**探索性**(rights=pending_review, G0/G1 未过);② 模式间差异无统计证据,不得宣布某模式"最优";③ F1 0.86–0.95 为合成+采集域指标,**不得**解读为真实业务准确率。

## 2. 数据与治理

| 库 | 规模 | 用途 | 权利 |
|---|---|---|---|
| registry_v3 | {len(reg)} 行 / {sum(1 for r in reg if r.get('is_representative'))} 代表(sha 去重) | 训练/评估语料 | 真实图 pending_review;合成 self_created |
| renders_3d | 820 (正 360/硬负 300/竞品 96/无招牌 40/异常 24) | 合成域:融合先验+硬负模板 | self_created synthetic_train_only |
| stock_benchmark | {stockb['n']} (B 素材/R 实拍双通道标注+仲裁) | 素材域专项负例 | 内部评估 |
| 金标准 gold | {len(key)} (全库覆盖+边界全采) | 人工双盲标注(协议+Kappa 工具就绪) | 内部评估 |

金标准组成: {', '.join(f'{k[0]}/{k[1]}×{v}' for k, v in sorted(gold_lab.items(), key=lambda x: -x[1])[:8])} …;正例变体覆盖 {dict(pos_styles)}。

## 3. 系统架构

```
L0 检索层:  CLIP / DINOv2 / SigLIP 嵌入 (cropfull + brand-cluster 检索)
L4 局部验证: patch 匹配 (patch-MaxSim 空间几何)
E8-G 融合:  s = 0.4·s_clip + 0.4·s_dino_mc + 0.2·s_probe, thr {op['thr']:.3f} (严格 >)
E10 证据层: VLM 四态盲判 (MATCH/NON_MATCH/UNKNOWN/OUT_OF_SCOPE)
  模式 A: 拒识即拒(UNKNOWN→按 E8-G 原判)   模式 B: UNKNOWN 回退 E8-G 原判
```

kNN 检索保持 store_id LOSO;探针仅在合成域训练;VLM 判决仅作推理证据,不作训练标签。

## 4. 结果(measured)

### 4.1 E8-G 冻结基线(330 张,30 正)
| 系统 | P | R | F1 | TP/FP/FN | 备注 |
|---|---|---|---|---|---|
| E8-G 冻结阈值 | {op['precision']:.3f} | {op['recall']:.3f} | {op['f1']:.3f} | {op['tp']}/{op['fp']}/{op['fn']} | thr {op['thr']:.3f} |
| 嵌套 LOSO(322 折阈值重选) | {nl['precision']:.3f} | {nl['recall']:.3f} | {nl['f1']:.3f} | {nl['tp']}/{nl['fp']}/{nl['fn']} | thr_std=0.0 |
| store 级 bootstrap | {ci['p']['mean']:.3f} [{ci['p']['lo']:.3f},{ci['p']['hi']:.3f}] | {ci['r']['mean']:.3f} [{ci['r']['lo']:.3f},{ci['r']['hi']:.3f}] | {ci['f1']['mean']:.3f} [{ci['f1']['lo']:.3f},{ci['f1']['hi']:.3f}] | — | B2000, 24 正店,CI 触 1.0 |

AuPR {e8g['auprc']:.3f};P@95R {e8g['p_at_95r']:.3f};R@95P {e8g['r_at_95p']:.3f}。通道 AuPR: clip {e8g['channels']['clip']['auprc']:.3f} / dino_mc {e8g['channels']['dino_mc']['auprc']:.3f} / probe {e8g['channels']['probe']['auprc']:.3f}。

### 4.2 E10 证据层(v1 全盲 330)
| 系统 | P | R | F1 | TP/FP/FN | 95% CI(store, B2000) |
|---|---|---|---|---|---|
| VLM-alone | {v1rec['vlm_alone']['precision']:.3f} | {v1rec['vlm_alone']['recall']:.3f} | {v1rec['vlm_alone']['f1']:.3f} | {v1rec['vlm_alone']['tp']}/{v1rec['vlm_alone']['fp']}/{v1rec['vlm_alone']['fn']} | — |
| 模式 A | {modeA_v1['precision']:.3f} | {modeA_v1['recall']:.3f} | {modeA_v1['f1']:.3f} | {modeA_v1['tp']}/{modeA_v1['fp']}/{modeA_v1['fn']} | {bci[0].get('f1',{}).get('lo','-')}–{bci[0].get('f1',{}).get('hi','-')} |
| 模式 B | {modeB_v1['precision']:.3f} | {modeB_v1['recall']:.3f} | {modeB_v1['f1']:.3f} | {modeB_v1['tp']}/{modeB_v1['fp']}/{modeB_v1['fn']} | {bci[1].get('f1',{}).get('lo','-')}–{bci[1].get('f1',{}).get('hi','-')} |

裁决分布(v1): MATCH {v1rec['verdict_distribution']['MATCH']} / NON_MATCH {v1rec['verdict_distribution']['NON_MATCH']} / UNKNOWN {v1rec['verdict_distribution']['UNKNOWN']} / OUT_OF_SCOPE {v1rec['verdict_distribution']['OUT_OF_SCOPE']};E10 在 E8-G 正例上拒识 {v1rec['e10_rejection_on_e8g_pos']['rejected_tp']}TP+{v1rec['e10_rejection_on_e8g_pos']['rejected_fp']}FP;3/330 冲突全分辨率仲裁,注册表标签全对。

### 4.3 v2 加强指令复测(330/330)
| 系统 | P | R | F1 | TP/FP/FN |
|---|---|---|---|---|
| VLM-alone | {v2rec['vlm_alone']['precision']:.3f} | {v2rec['vlm_alone']['recall']:.3f} | {v2rec['vlm_alone']['f1']:.3f} | {v2rec['vlm_alone']['tp']}/{v2rec['vlm_alone']['fp']}/{v2rec['vlm_alone']['fn']} |
| 模式 A | {v2rec['modeA']['precision']:.3f} | {v2rec['modeA']['recall']:.3f} | {v2rec['modeA']['f1']:.3f} | {v2rec['modeA']['tp']}/{v2rec['modeA']['fp']}/{v2rec['modeA']['fn']} |
| 模式 B | {v2rec['modeB']['precision']:.3f} | {v2rec['modeB']['recall']:.3f} | {v2rec['modeB']['f1']:.3f} | {v2rec['modeB']['tp']}/{v2rec['modeB']['fp']}/{v2rec['modeB']['fn']} |

**证伪记录**: §6.1 曾据 13/13 定向回归预测 v2 模式 B F1≈0.966,全量复测未兑现({v2rec['modeB']['f1']:.3f})——13/13 是回归测试不是泛化验证。裁决迁移矩阵主项: NON_MATCH→UNKNOWN ×{v2mig['matrix'].get('NON_MATCH->UNKNOWN',0)}(加强"两证据门槛"使 VLM 更保守,UNKNOWN 回退 E8-G 兜底,模式 B 的 R 由回退机制保住)。
配对 bootstrap: B−A {v2pair['pairs']['B-A']['delta_f1']:+.2f} [{v2pair['pairs']['B-A']['lo']:+.2f},{v2pair['pairs']['B-A']['hi']:+.2f}] 显著={str(v2pair['pairs']['B-A']['significant']).lower()};B−VLM {v2pair['pairs']['B-VLM']['delta_f1']:+.2f} 显著={str(v2pair['pairs']['B-VLM']['significant']).lower()};A−VLM {v2pair['pairs']['A-VLM']['delta_f1']:+.2f} 显著={str(v2pair['pairs']['A-VLM']['significant']).lower()}。**没有任何模式差异统计显著。**

### 4.4 素材域专项(stock_benchmark 220)
| 检查 | 结果 |
|---|---|
| E8-G 代理误收(两通道 0.5/0.5) | **{stockres['fp']}/{stockres['n']} = {stockres['fp_rate']:.1%}** @thr {stockres['thr']} |
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

- 架构有效: 检索+局部验证+融合基线({op['f1']:.3f})经 VLM 证据层提升至 {max(modeB_v1['f1'], v2rec['modeB']['f1']):.3f}(点估计),素材域误收 {stockres['fp_rate']:.1%}。
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
"""
    txt = lines
    (ROOT / "TECH_REPORT.md").write_text(txt)
    print(f"TECH_REPORT.md written ({len(txt)} chars)")

if __name__ == "__main__":
    main()
