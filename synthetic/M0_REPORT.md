# M0 2D 参数化门头母本 — 实现报告 v1.0

实现 2026-11-11。设计文档 §6.1 M0 路线落地：参数化图层母本 + 自建程序化街景 + 四点透视合成 + 摄影退化 + 标注随生成。
生成器：`synthetic/m0_generate.py`（GEN_VERSION=`m0-2d-parametric-v1.0`）。

## 1. 交付物

| 产物 | 路径 | 说明 |
|---|---|---|
| 生成器 | `synthetic/m0_generate.py` | 母本注册表 + 渲染 + 透视 + 光照 + 退化 + CF + 三层QC + schema校验 |
| manifest schema | `synthetic/schemas/synthetic_manifest.schema.json` | 从设计包 zip 解压（draft 2020-12） |
| 冒烟批 | `synthetic/renders/`（13 张） | 每簇 1 + CF 7 |
| 批量 v1.0 | `synthetic/renders_batch/`（55 张） | 每簇 8（48）+ CF 7 |
| manifest | `renders_batch/synthetic_manifest.jsonl` | 55 条，schema 0 错误 |
| 审样对照表 | `renders_batch/batch_contact.jpg`、`ensub_zoom.jpg` | 视觉 QC3 证据 |

## 2. 母本（6 候选簇，全部 candidate_unverified）

颜色/几何按官方 E2 标定（红 #e60012 / 蓝 #005dac / 彩钢条 白蓝白红白 / 灯箱比例）。**禁止命名简约/标准/都市**（E2 对应未证实）。

| 簇 | 名称 | 类因子（关键差异） | 字体 | E2 依据 |
|---|---|---|---|---|
| CS-01 | 编号标准型 | 红底+蓝横带+No.XXXX+英文副标 | 黑体 | bzb B3-5 |
| CS-02 | 书法字体型 | ⊙间隔+编号，**宋体代理行楷** | song_proxy | real_img_0000 |
| CS-03 | 红蓝常规型 | 红底+蓝带+英文副标，无编号无PLUS | 黑体 | jyb B3-1 |
| CS-04 | 立体发光字型 | **蓝底+居中红logo面板**+白色立体发光字 | Hiragino W6 | dsb B3-6/7/8/9 |
| CS-05 | PLUS升级型 | 主标右侧 PLUS 角标（深色底白字） | 黑体 | jyb 品牌故事页 |
| CS-06 | 超市型 | "左邻右舍超市"+彩钢条，无色带无编号 | Hiragino W6 | real_img_0012 |

**CS-02 字体代理声明**：本机无楷体/行楷，用宋体 Bold + 轻微剪切形变作代理（`font_proxy=True` 显式标注）。非真行楷，待真字体标定后替换；当前 CS-02 合成样本仅用于流程验证，不作为该簇正式训练真值。

## 3. 可控 / 不可控变量分离（§6.3）

- **类因子**（白名单 `CLASS_INVARIANT_KEYS`，CF-2 不得改动）：font / number / en_sub / plus / dim3d / board / band / caigang / separator / suffix。
- **无关因子**（独立随机）：yaw[-28,28] / pitch[-10,10] / 招牌宽度[0.52,0.82] / 位置 / 曝光 / 模糊 / JPEG[55,92] / 遮挡[0,0.22] / 背景{street_a,b,c,mall} / 光照{day,dusk,night,rain}。
- **物理/品牌合理联动**：夜景/黄昏 × 灯箱自发光（`lightbox_glow` 在摄影退化【之后】施加，避免被再压暗）；雨天 × 玻璃反光/雨丝。不造物理不合理怪样本。

## 4. 标注随生成（by construction，§6.5）

每张自动产出：`class_id` / `style_id` / `sign_polygon`（4点）/ `landmarks`（board/band/main_text/number/en_sub/plus/caigang/logo_panel 各组件四点框）/ `visible_attributes` / `store_number` / `en_subtitle` / camera(yaw,pitch) / nuisance 全因子 / `asset_hash`（母本内容哈希）/ `generator_commit` / `seed` / `image_sha256`。

**像素级一致性已验证**：sign_polygon 与渲染共用同一单应矩阵（`homography_yaw_pitch` → `fit_quad_transform`）；抽查 12 张多边形内部红/蓝占比 0.67–0.96（白天 0.9+），证明多边形精确贴合招牌。

## 5. 反事实组（§6.4，7 张）

| 组 | 样本 | 设计 |
|---|---|---|
| CF-1 类别干预 | cf1_CS-03 / cf1_CS-05 | 固定背景/机位/光照/退化，只换版型（差异=PLUS） |
| CF-2 环境干预 | cf2_0/1/2 | 固定 CS-03 类因子，独立随机环境（雨/夜/雨） |
| CF-3 局部破损 | cf3_damage | 关键文字大面积遮挡+模糊，`expected_behavior=lower_confidence_or_reject`，豁免可读性门禁 |
| CF-4 非法搭配 | cf4_anomaly | **真实渲染**绿底+黄带（非法 VI），`class_id=synthetic_unknown`，非冒充真实门头 |

## 6. 三层 QC（§6.5）

- **QC1 参数合法性**：yaw/pitch/宽度范围 + 类别关键元素白名单（PLUS/编号/彩钢条不得非法缺失）。
- **QC2 渲染后可读性**：招牌面积占比>0.04 + 局部对比度 std>18 + 宽度>120px。
- **QC3 视觉审样**：vision 模型逐批审（本报告 §8 记录三轮迭代）。

批量结果：**55 张 schema 0 错误；QC 49/55 通过**。6 张 FAIL 全为 rain/dusk + 遮挡致对比度 13–17（略低于阈值），无参数违规——属合理暗光遮挡难例，标记供人工审样，非渲染缺陷。

## 7. 确定性复现

同 seed 重生成 **6/6 字节级 sha256 一致**。manifest 记录 seed + generator_commit + asset_hash + image_sha256，全可追溯。

## 8. 视觉审样迭代（三轮）

1. **首轮**：发现 CS-04 整块蓝板（应蓝底+红logo居中）、CS-06 透视漂浮（yaw 过大）、夜景过暗。→ 修：CS-04 加红 logo 面板（E2 B3-6）、yaw 收窄[-28,28]、招牌下移贴玻璃带、夜景灯箱发光。
2. **二轮**：发光被退化再压暗。→ 修：`lightbox_glow` 移到退化后 + 增强（night×2.1/dusk×1.45）。
3. **三轮**：遮挡广告与红招牌同色、⊙太小像®。→ 修：广告改橙色异色、⊙放大。英文副标"裁切"经全分辨率核实为**误读**（GoodNeighbor 无 s 是有意拼写变体 + 前景立柱局部遮挡），非 bug。

## 9. 合成数据隔离规则（强制，§9.2）

1. **永不入真实 test split**：所有合成 manifest `split=synthetic_train_only`，`domain=synthetic`。真实测试集只含可合法评估的真实图片。
2. **背景零交叉**：背景全程序化自建（`source_policy=self_created`），不使用任何真实街景/受版权图，故合成训练与真实测试无同源背景泄漏。真实图全部 `link_only_pending`，未当背景。
3. **标签权威有限**：`label_authority=synthetic_by_construction`——合成标签只对自身生成语义有效，不替代真实世界验证。
4. **剂量受控**：合成数量非 KPI；后续 E7/E8/E9 消融按 100/300/500 量级，对照组同等训练更新次数。
5. **母本未证实**：`asset_status=candidate_unverified`，E2 官方对应审核 + 双人母本验收前，不得宣称正式分类精度（G0 门禁）。

## 10. 缺口与下一步

- **M0 是 2D 矢量母本**：无招牌厚度/玻璃反射/悬挂侧标/复杂夜景照明。若真实泛化不足且误差与几何/光照相关，启动 M1（2.5D，§6.1）。
- **CS-02 字体代理**需真行楷标定；CS-04 立体字发光感仍偏弱（M1 增强）。
- **背景程序化**风格较模板化（4 种），M1 可引入更多立面几何/自主拍摄许可背景。
- 合成数据**尚未用于训练**；需先完成真实集权利审查 + 双盲金标准（G0/G1），再跑 E0–E10 消融验证 sim-to-real 增益（G2）。
- 当前 55 张为流程验证基线，非经验最优产量。

## 11. 复现命令

```bash
cd ~/chatgpt-harvest/6af41562/synthetic
python3 m0_generate.py --smoke                 # 冒烟 13 张
python3 m0_generate.py --per-cluster 8 --out renders_batch   # 批量 55 张
python3 m0_generate.py --verify --out renders_batch          # 仅 schema 校验
```