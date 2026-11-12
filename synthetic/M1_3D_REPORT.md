# M1-3D 参数化数字孪生母本报告（Blender）

实现 2026-11-11。用户指令：以 3D 孪生合成数据做识别训练（真实图权利未审，合成是唯一可合法训练的数据源）。
脚本：`synthetic/m1_3d_twin.py`（GEN_VERSION=`m1-3d-blender-v1.0`），Blender 5.2.2 LTS，EEVEE-Next + AgX。

## 1. 定位

设计文档 §6：M0（2D 矢量）已完成 → 本工作实现 **M1（2.5D）并部分触及 M2**：招牌厚度/铝合金包边、灯箱自发光、立体发光字（extrude 0.05m）、玻璃橱窗（transmission）、雨棚、邻栋语境、夜景路灯/室内货架灯带、遮挡物（树/杆/货车）。D-02 原将全 3D 后置，用户明确指令提前执行。

## 2. 母本与真值

- 类因子与 M0 MASTERS 完全对齐（CS-01..06，candidate_unverified）；VI 常量来自 E2（红 #e60012 / 蓝 #005dac / ⊙ / 彩钢条白蓝白红白 / 编号铝板）。
- 中文字体：fontTools 从系统 TTC 提取 TTF（hei/hira_w6/song_bold/arial_bold）→ Blender FONT 曲线，中文渲染验证无乱码。CS-02 宋体代理 font_proxy=True。
- **构造真值**：每渲染导出 sign_polygon（8 角投影）+ 组件框（main_text/number/en_sub/plus）+ 相机参数 + lighting + seed。480/520 含招牌真值（no_sign 40 张无）。
- 干扰随机化：yaw ±30° / pitch -8..12° / 距离 5.5–11m / 焦距 28–45mm / 曝光 ±0.45 / 光照 day45:dusk20:night25:rain10 / 遮挡 40%。

## 3. 批量产出（renders_3d/，700 张）

| 类别 | 数量 | 说明 |
|---|---|---|
| positive CS-01..06 | 6×60=360 | 每簇均衡 |
| other_brand | 96 | 8 个异品牌（美宜佳/天福/芙蓉兴盛/Today/见福/红旗/唐久/十足） |
| no_sign | 40 | 无招牌立面 |
| anomaly | 24 | 违规 VI：真品牌名+绿板黄带（brand=other，拒识训练） |
| **hard_negative** | **180** | **红/蓝板结构+异品牌文字**（难负例，教模型辨文字非颜色） |

难负例动机：首轮 E8 探针精确率仅 12%（208/300 误报）——合成负例太容易，模型学成"红板=品牌"。难负例保留品牌色板结构但换文字，直击该捷径。

## 4. 质量验证

- 烟雾 11 张 + 批量抽检 8 张经视觉模型逐格核验：中文正确、VI 颜色正确、夜景发光、负例/违规/无招牌符合设定，无黑图/几何错乱。
- 修复记录：夜景自发光过曝（红板发白 234/159/146）→ glow 2.4→1.1、letter 4.5→2.2，复验红色饱和恢复。
- 确定性：每渲染独立 seed，场景从零重建；同 seed 同参数。

## 5. 治理（硬约束）

- 全部几何/材质/字体程序化自建 → `source_policy=self_created`，`split=synthetic_train_only`。
- **不使用任何真实照片作背景/贴图**（权利 pending_review）；难负例的"难"来自程序化结构模仿，非真实图复制。
- style_id 候选簇未证实（E2 对应待品牌确认）；franchise_type=null。
- 合成永不进入真实测试；真实评估仅探索性（G0/G1 未过）。

## 6. 复现

```bash
cd ~/chatgpt-harvest/6af41562/synthetic
blender -b -P m1_3d_twin.py -- --smoke                     # 11 张质检
blender -b -P m1_3d_twin.py -- --per-cluster 60 --neg 160 --out renders_3d
blender -b -P m1_3d_twin.py -- --per-cluster 0 --hard-neg 180 --seed-base 30261111 --out renders_3d
```

训练/验收结果见 `E8_REPORT.md`。