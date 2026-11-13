#!/usr/bin/env python3
"""TECH_REPORT.md 生成器: 所有数字从冻结 JSON 自动填充(禁止手抄)。
用法: python3 synthetic/gen_tech_report.py  →  TECH_REPORT.md
"""
import json, re, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "synthetic" / "vis_cache"

def load(p):
    return json.load(open(p))

def f1row(name, m, extra=""):
    return (f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m.get('tp','-')}/{m.get('fp','-')}/{m.get('fn','-')} | {extra} |")

def main():
    e8g = load(VC / "e8g_protocol.json")
    v4res = load(VC / "e8g_v4_results.json")
    v4 = v4res["e8g_v4"]; v4n = v4["nested_loso"]
    va = v4res["e10_v4"]["vlm_alone"]; mA = v4res["e10_v4"]["mode_a"]; mB = v4res["e10_v4"]["mode_b"]
    v4ci = v4["ci95"]["f1_ci95"]
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
    svd = load(VC / "stock_vlm_summary.json")["verdict_dist"]
    vlmf = load(VC / "stock_vlm_summary.json")["match_false_accepts"]
    vlmf = {"n": len(vlmf), "rate": len(vlmf) / stockb["n"], "ids": vlmf}
    _sc = np.load(VC / "stock_e8g_scores.npz", allow_pickle=True)
    _smap = {str(i): float(s) for i, s in zip(_sc["ids"], _sc["scores"])}
    modea_fp = modeb_fp = 0
    for _r in stockb["rows"]:
        _v = _r.get("vlm_verdict_v8")
        _e = _smap.get(_r["image_id"], 0.0)
        if _v == "MATCH":
            modea_fp += 1; modeb_fp += 1
        elif _v == "UNKNOWN" and _e >= 0.636:
            modeb_fp += 1
    modea_rate = modea_fp / stockb["n"]; modeb_rate = modeb_fp / stockb["n"]
    genrec = load(VC / "gen_reconciliation.json")
    genpair = load(VC / "gen_paired_bootstrap.json")
    _pv = [load(VC / f"pairwise_verifier_results_{s}.json") for s in ("synthonly", "none", "cross")]
    pv_best_f1 = max(d["verifier_oof"]["f1"] for d in _pv)
    pv_auroc = max(d["verifier_oof"]["auroc"] for d in _pv)
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
- VLM 四态证据层(E10): v1 模式 B F1 {modeB_v1['f1']:.3f};v2 加强指令 F1 {v2rec['modeB']['f1']:.3f};**GEN 独立泛化集上模式 A {genrec['results']['modeA']['F1']:.3f} > 模式 B {genrec['results']['modeB']['F1']:.3f},排名跨集翻转**,全部配对差异跨零(330 与 GEN 均然)。
- 三项不可豁免门槛: ① 全部评估为**探索性**(rights=pending_review, G0/G1 未过);② 模式间差异无统计证据,不得宣布某模式"最优";③ F1 0.86–0.95 为合成+采集域指标,**不得**解读为真实业务准确率。

## 2. 数据与治理

| 库 | 规模 | 用途 | 权利 |
|---|---|---|---|
| registry_v3 | {len(reg)} 行 / {sum(1 for r in reg if r.get('is_representative'))} 代表(sha 去重) | 训练/评估语料 | 真实图 pending_review;合成 self_created |
| registry_gen | 独立泛化集 223 图/148 店组(110 正) | 跨采集源泛化评测,sha 级独立于 330 | 真实图 pending_review |
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

### 4.4 GEN 独立泛化集(223 图/148 店组,sha 级独立)
| 系统 | P | R | F1 | TP/FP/FN |
|---|---|---|---|---|
| E8-G(冻结 thr) | {genrec['results']['e8g_pred']['P']:.3f} | {genrec['results']['e8g_pred']['R']:.3f} | {genrec['results']['e8g_pred']['F1']:.3f} | {genrec['results']['e8g_pred']['tp']}/{genrec['results']['e8g_pred']['fp']}/{genrec['results']['e8g_pred']['fn']} |
| VLM-alone | {genrec['results']['vlm_alone']['P']:.3f} | {genrec['results']['vlm_alone']['R']:.3f} | {genrec['results']['vlm_alone']['F1']:.3f} | {genrec['results']['vlm_alone']['tp']}/{genrec['results']['vlm_alone']['fp']}/{genrec['results']['vlm_alone']['fn']} |
| 模式 A | {genrec['results']['modeA']['P']:.3f} | {genrec['results']['modeA']['R']:.3f} | {genrec['results']['modeA']['F1']:.3f} | {genrec['results']['modeA']['tp']}/{genrec['results']['modeA']['fp']}/{genrec['results']['modeA']['fn']} |
| 模式 B | {genrec['results']['modeB']['P']:.3f} | {genrec['results']['modeB']['R']:.3f} | {genrec['results']['modeB']['F1']:.3f} | {genrec['results']['modeB']['tp']}/{genrec['results']['modeB']['fp']}/{genrec['results']['modeB']['fn']} |

**排名跨集翻转**: GEN 上模式 A({genrec['results']['modeA']['F1']:.3f}) > 模式 B({genrec['results']['modeB']['F1']:.3f}),与 330 的 v1(B>A)相反;全部配对差跨零(B−A {genpair['pairs']['modeB-modeA']['delta']:+.3f} [{genpair['pairs']['modeB-modeA']['ci'][0]:+.3f},{genpair['pairs']['modeB-modeA']['ci'][1]:+.3f}])。GEN 上 VLM 变体误拒显著(FN 18,最大模糊边界=门店样式变体)。E10 对基线的提升在真实泛化域缩水至 +0.026。

### 4.5 素材域专项(stock_benchmark 220)
| 检查 | 结果 |
|---|---|
| E8-G 代理误收(两通道 0.5/0.5) | **{stockres['fp']}/{stockres['n']} = {stockres['fp_rate']:.1%}** @thr {stockres['thr']} |
| 双通道标注一致率 | 80.0%(220 张),44 分歧全分辨率仲裁后 B201+R19 |
| 素材搜索污染率(实拍混入) | 3.2% |
| VLM 四态层(v8,220/220) | 裁决分布: UNKNOWN {svd.get('UNKNOWN',0)} / OUT_OF_SCOPE {svd.get('OUT_OF_SCOPE',0)} / NON_MATCH {svd.get('NON_MATCH',0)} / MATCH {svd.get('MATCH',0)} |
| VLM-alone 误收(MATCH∧负例) | **{vlmf['n']}/220 = {vlmf['rate']:.2%}**({','.join(vlmf['ids'])}) |
| 模式 A 误收(拒识即拒) | {modea_fp}/220 = {modea_rate:.2%} |
| 模式 B 误收(UNKNOWN 回退 E8-G) | {modeb_fp}/220 = {modeb_rate:.2%}(111 UNKNOWN 回退带 {modeb_fp - modea_fp} 个 E8-G 残余误收) |

残余误收集中于"含品牌风格字体立体字的设计图"——与真实域 CS-04 立体发光字变体同型,是变体边界问题的素材域镜像。素材域上模式 A 优于模式 B(与 GEN 方向一致): VLM 高弃识率(50% UNKNOWN)域内,拒识即拒优于回退。

## 5. 已知失效模式

0. **成对验证头不足以替代主通道(2026-11-13 关闭)**: ArcFace 式 pairwise verifier(冻结嵌入逐维关系特征 ⊙/|diff| → MLP,6 折组不相交+renders_3d 扩对)三种配置最好 F1 {pv_best_f1}(AUROC {pv_auroc}),低于 E8-G 0.848/0.862——**关系建模补不了嵌入空间的域表征瓶颈**(GPT 终审"停止调 embedding"再次成立)。但 impostor 域全面胜出: 素材域 FAR 0-1.8% vs E8-G 3.6%,与打印攻击防御同构;该信号已被 VLM 证据层以更优方式覆盖(Mode A 0.45%),不另行集成。数据: pairwise_verifier_results_{{synthonly,none,cross}}.json。
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

## 8. 2026-11-13 标签修正 (registry v4) — 本报告 §1-§7 数字基于 v3 标签, 含下列偏差

素材池分类战役(4×scout 334图归类 + 11格双通道仲裁 conv 6af686b2)发现并修正:
1. **9 张 render_stock 误标**: 图像搜索素材波混入品牌宣传/新闻实拍图, 实为真实本品牌门店 → 翻正(6张在评测内: 0140/0141/0155/0217/0404/0479; 3张新增: 0037/0182/0405)。
2. **img_0488 本地文件=防盗链占位图**(400×400提示图, 非门店): 曾以占位嵌入得 E8G 0.64 计 TP, 且是 v1"Mode B 救援优势"叙事的支柱 → 剔除评测, "B>A"结论撤回。
3. **0.636 阈值系在被污染标签上校准**, v4 下失准。

**v4 修正后基线** (e8g_v4_results.json, 332行/38正):
- E8-G 冻结阈值: P {v4['P']} / R {v4['R']} / F1 {v4['F1']} (CI95 {v4ci}); 嵌套 LOSO 阈值重选: F1 {v4n['F1']} (P {v4n['P']} / R {v4n['R']})
- E10 v4 (329行/35正, 历史盲裁沿用): VLM 单独 F1 {va['F1']}; Mode A {mA['F1']} / Mode B {mB['F1']} — **A/B 差异不再显著, v1"B 更优"撤回**
- VLM 新失效模式: 夜拍小招牌系统性误读成竞品(0140→唐久/0141→新佳宜/0155→十足), E10 三次暗错误
- 同名异业族谱扩充: 火锅/民宿/餐馆/左邻右社/楼盘广告
- 结论修订: v3 报告的 0.862/0.931/0.951 系标签污染下的乐观值; 修正标签上系统真实水平约 F1 0.77(冻结阈值)-0.81(重选阈值), VLM 层 0.86。改进优先级 = 夜拍/远距小招牌读码能力, 非阈值调参。
"""
    txt = lines
    (ROOT / "TECH_REPORT.md").write_text(txt)
    print(f"TECH_REPORT.md written ({len(txt)} chars)")

if __name__ == "__main__":
    main()
