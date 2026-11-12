# 左邻右舍门头数据采集报告 v3（browser relay 自主采集）

采集时间：2026-11-11。通道：已登录中转 Chrome（CDP 9222）+ Bing 图片搜索 + 6 条种子源页 + 驰骋官网 SI 手册 + 地图 POI 相册（360/百度网页发现 POI ID → 高德 getPoiInfo）。

性质：**公开来源候选集 + 视觉模型逐张核验**，非人工双盲金标准；权利状态全部 pending_review，**默认仅可链接、未授权不得用于训练**（设计文档 §4.4）。

## 1. 数量总览

| 类别 | 标注数 | 去重后代表数 |
|---|---|---|
| 正例（左邻右舍真门头实拍） | 36 | 30 |
| 负例（合计） | 463 | 300 |
| 总计 | 499 | 330 |

**独立门店（store_id，7 家）**：store_0062（3 张同店）、store_2047、store_2471、store_amap_B0FFM0KX3P（金华环镇东路）、store_amap_B0JRLL5YTC（杭州红丰路，3 张）、store_amap_B0KG990ZNE（南平梦笔大厦，3 张）、store_amap_B023B0A7NA。其余正例店号不可辨，store_id 留空（不猜）。

**正例属性覆盖**：含 PLUS 3 张；夜景 6 张；含英文副标 3 张。

## 2. 负例分类（高价值困难负样本）

| 负例类型 | 去重代表数 | 说明 |
|---|---|---|
| render_stock | 256 | 效果图/3D渲染/素材站模板（昵图/千图/包图）——非实拍，须排除出真实测试集 |
| same_name_unrelated | 12 | 同名无关实体：社区火锅/便民服务中心/门诊部/餐馆 + 同名住宅小区（成都/杭州）/园区大门/义乌商贸城入口——品牌鉴别关键负例 |
| indoor | 9 | 纯室内/货架/商品/品牌展示墙——非门头（含高德 category 误标"门头"的货架照） |
| confusable_brand | 9 | 混淆品牌：邻几GOOD SPOT/佐邻佑邻/美邻优/好邻居/99便利店/甄选——细粒度负例 |
| logo | 6 | 品牌logo/平面素材/书法插画 |
| competitor | 4 | 竞品便利店：美宜佳/7-11/全家/罗森/快客/十足 |
| uncertain | 3 | 疑似渲染或招牌不清，待原图复核 |
| map | 1 | 地图截图/无招牌街景 |

## 3. 去重与泄漏控制

- 字节级 sha256 去重 + 256-bit dHash 感知哈希近重复聚类（Hamming≤12）。
- 正例 36 张标注 → 30 个唯一门头；6 张为转载重复（如 No.0062 同时出现在 Bing 与百度加盟星）。地图 POI 新增 8 张正例与既有 22 张跨源 dHash 去重 0 冲突。
- 每个近重复簇赋 photo_family=fam_<cluster>，train/val/test 须按 store_id + photo_family 双重隔离（设计文档 §4.4）。
- **印证设计文档警告**：公开图大量互相转载，凑张数会制造虚假样本量；本报告以唯一代表数为准。

## 4. 官方 E2 形象标准（详见 OFFICIAL_E2_FINDINGS.md）

- 官网三版独立页：简约版/jyb、标准版/bzb、都市版/dsb；下载 16 张官方 SI 手册图（约28MB，含 2–3 万像素长图，已切 82 段）。
- 品牌 VI：标准红 #e60012（PANTONE 3346C）/辅助蓝 #005dac（PANTONE 2175C）；标志「左邻·右舍」⊙；口号「亲如一家 Good Neighbors」。
- 门头规范（B3）：红底白字吸塑灯箱主招牌（2000×1000mm）+彩钢条「白蓝白红白」色带（总高600mm）+350×700mm柱灯箱+防水飘檐。
- **关键**：官方 SI 标准门头**无 PLUS、无英文副标**；PLUS 是真实变体（简约版品牌故事页夜景实拍 + 搜狐稿 P05_00 实证），非官方标准版，对应加盟/升级版本待品牌方确认（E3）。
- 视觉版本（L1）≠加盟合同（L4）；官方未公布对应规则，franchise_type 默认 null。

## 5. 缺口与下一步（诚实披露）

- 设计文档目标 30–50 张可核验独立实拍；当前**唯一已核验正例 30 张（达下限）**，独立门店 7 家，统计效力仍有限。
- 地图 POI 已补齐 30 张下限（详见 MAP_POI_REPORT.md）；搜索引擎 POI 索引近枯竭，进一步扩充需登录态地图账号或品牌方门店台账。
- 薄簇：CS-02 书法字体型 / CS-06 超市型各仅 1 张，簇内样本不足。
- 权利：全部 pending_review；进入训练前须逐张完成授权审查（设计文档 E2/E3 门禁）。
- 标注：当前为视觉模型单轮核验，正式金标准须双人独立盲标+第三方裁决（§4.4）。

## 6. 产物清单

- registry_v3.csv / registry_v3.json：499 行完整登记表（含 sha256/感知哈希/photo_family/权利状态/地图 POI 正负例）；registry_v0–v2 为历史版本。
- images/：511 Bing 候选 + 129 种子图（69 唯一）+ 高德 POI 相册图；official/：16 张官方 SI 图 + slices/ 82 段。
- sheets/：24 页候选核验表 + 种子核验表 + positive_final.jpg 正例终核表 + amap*_check.jpg 地图核验表。
- OFFICIAL_E2_FINDINGS.md：官方形象标准证据；MAP_POI_REPORT.md：地图 POI 采集报告；STYLE_CLUSTERING.md：候选簇 v0（22 张时点）。
- bing_candidates.json / download_log.json / image_meta.json / seed_pages.json / verified_labels.json / seed_labels.json / verify_all.json / amap_pois_batch*.json：可追溯中间产物。