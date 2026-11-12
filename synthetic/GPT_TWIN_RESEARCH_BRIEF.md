# 门店数字孪生研究资料汇编（供 GPT 整理报告）

## 任务
请把以下研究资料整理成一份结构化中文报告《左邻右舍全面门店数字孪生：行业案例与技术路线报告》。要求：①基于所给资料，不虚构数字；②引用标出处；③结合"我们项目现状"给出可执行建议；④区分"行业受控场景精度"与"我们开放门头场景"，不外推；⑤治理约束不可违反。

## 一、项目背景与现状（measured）
- 项目：左邻右舍（驰骋控股）便利店门头视觉识别。最终目标（用户明确）：**全面门店数字孪生，门头只是第一步**。
- 当前最佳管线 E8-G：CLIP+DINOv2+合成探针融合，F1 0.862（330评估集，30正例/24门店组）。E10 VLM证据层 F1 0.93–0.95（探索性，系统间差异配对Bootstrap均不显著）。
- **合成探针 AuPR 0.324（<0.5，负贡献）**——纯合成训练 sim-to-real 域差症状。
- 3D孪生 v1：820张低多边形示意渲染，相机仅 yaw±30°。v2 升级：程序化纹理（砖墙/招牌磨损/沥青）+宽域相机 yaw±75°/pitch-20~35°/dist3-22m+传感器噪声后处理。视觉真实感评分 v1≈3→v2≈5.5-7/10。室内仍偏暗。
- 真实采集：高德REST API（restapi.amap.com/v5/place/text，types=060200便利店过滤+分页）采集 2542候选图/1178唯一门店POI/0隔离违规，浙江占95%。全部 rights=pending_review。
- 治理硬约束：G0/G1权利门未过→不训练不部署不宣称业务准确率；合成 self_created/synthetic_train_only；**绝不以真实照片为背景/贴图**；VLM裁决=推理证据非训练标签。

## 二、行业案例（已核原始来源）

### 案例1：Kinetic Vision + NVIDIA（门店孪生闭环范本）
出处：NVIDIA官方博客2021 https://developer.nvidia.com/blog/deploying-highly-accurate-retail-applications-using-a-digital-twin
- 合成数据生成器：多角度/光照/背景/商品，手段含GAN/照片级3D/模拟传感器(LiDAR/RADAR/IMU)/合成X光/物理仿真。
- 闭环：合成数据→数字孪生(仓库/产线/门店虚拟副本)里训练→仿真→再训练达标→Fleet Command边缘部署→真实数据回流再优化。
- 硬数字：数字孪生法开发时间/成本≈物理环境法的10%；无需物理基础设施/不打断生产；新品无需实拍标注。

### 案例2：7-Eleven台湾7000+店（同行评审，最扎实）
出处：Scientific Reports 2025, DOI 10.1038/s41598-025-27773-5, PMC12708730
- 任务：陈列图合规(planogram compliance)，货架实际摆放vs标准陈列图比对，判每个商品 位置对/错架/缺货。
- 管线三段+数据量+精度：
  - 货架检测 YOLOv8：15232图，P99.23%/R98.93%/mAP@50 99.41%
  - 商品检测 YOLOv8：99135图，P94.61%/R93.02%/mAP@50 95.7%（超Deformable DETR）
  - 商品分类：471类×均210图，ResNet101/FAN 99.86%
- 关键技术：①虚拟货架(virtual shelf)=数字孪生，窄过道多张重叠图拼接成完整货架；②Needleman-Wunsch序列比对算法(生物信息学DNA比对)比对实际vs标准商品序列。
- 少样本：FAN每类5样本，未见商品Top-1 98.39%/Top-5 99.48%。
- 数据采集：4家不同规模门店，3天不同时段，故意变角度/光照/货架状态；自动标注+聚类降人工。

### 案例3：Neurolabs ZIA（纯合成训练SKU检测）
出处：https://www.neurolabs.ai/blog-post/using-neurolabs-retail-specific-synthetic-dataset-in-production
- Zero Image Annotations零人工标注，Unity合成货架场景训SKU检测。
- 公开数据集CPGDet-129：1200货架图/129 SKU/~1.7万标注(bbox+分割掩膜)，含结构+外观变化，含次优门店光照。
- 注意：合成的是货架商品(几何规则可控)，非开放街景门头。

### 案例4：Clobotics / Infosys（规模化）
- Clobotics：货架视觉分析覆盖2万美国门店 https://clobotics.com/resources/insights/north-america-shelf-analytics
- Infosys+NVIDIA Metropolis：Omniverse Replicator合成货架图+TAO低代码训练+DeepStream边缘推理；痛点SKU/包装持续变化→合成数据保鲜 https://www.infosys.com/services/applied-ai/offerings/smarter-retail-infra-automation.html

## 三、域随机化与混合训练文献
- Tobin et al. 2017 Domain Randomization (arXiv:1703.06907, OpenAI/IROS)：10万高度随机化场景(随机纹理/相机/光照/干扰物)，不追求照片级真实，变化度>保真度→sim-to-real成功。
- Tobin RSS 2019 Beyond Domain Randomization："More Variability, More Data, Less Fidelity"。
- 混合训练(arXiv:2506.03449, 2506.24093)：合成+10-40%真实逼近全真实训练；合成占比60-80%甜区；纯合成/低保真→"两个世界"陷阱(合成验证高真实停滞)。

## 四、工具链
| 工具 | 许可 | 定位 | 适配 |
|---|---|---|---|
| BlenderProc | GPL-3 | Blender脚本化渲染+域随机化+自动真值 | 近期首选(已用Blender零迁移) |
| Omniverse Replicator | NVIDIA | OpenUSD物理级+自动depth/分割/bbox+多GPU | 中期全店物理孪生 |
| Unity Perception | Unity | 合成数据+标注(Neurolabs用) | 货架SKU备选 |
| Kubric | Apache-2 | Google Blender合成数据 | 备选 |
| Geometry Nodes插件(Nodee/Buildify/Procedural Building Generator) | 各异 | 参数化建筑立面/楼层/窗 | 全立面组件加速 |

## 五、我们的架构草案（组件化）
StoreTwin框架：组件层(C1门头招牌[已实现]/C2全立面/C3室内货架/C4陈列合规/C5环境)→装配层→随机化层→真值层(统一schema:bbox/polygon/分割/depth/SKU)→渲染层(BlenderProc/Replicator)。
识别任务扩展：门头品牌识别→门店检测→货架SKU检测→陈列合规→全店场景理解。

## 六、报告应回答的关键问题
A. 行业案例对我们"门头→全店"路线的验证程度与边界（受控货架vs开放门头的域差差异）。
B. 合成探针AuPR 0.324的根因诊断与修复路径（域随机化够不够？是否必须真实锚定？）。
C. 工具选型决策（BlenderProc近期 vs Omniverse中期）的利弊与迁移成本。
D. 组件化架构与统一真值schema的合理性，室内组件优先级。
E. 数据量现实（行业11万标注 vs 我们合成820+真实候选2542）下的务实路线。
F. 治理约束下的可行边界（权利门未过，哪些能做哪些不能）。
请给每问 CLOSED/PARTIAL/OPEN 判断 + 可执行建议，最后给总体 APPROVE/REVISE。