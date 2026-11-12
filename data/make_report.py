#!/usr/bin/env python3
import json, csv
from collections import Counter, defaultdict
reg=json.load(open("registry_v0.json"))
pos=[r for r in reg if r["label"]=="positive"]
pos_rep=[r for r in pos if r["is_representative"]]
neg=[r for r in reg if r["label"]=="negative"]
neg_rep=[r for r in neg if r["is_representative"]]
nt=Counter(r["neg_type"] for r in neg)
# store ids
sids=Counter(r["store_id"] for r in pos if r["store_id"])
# plus / english / night attributes
plus=[r for r in pos if "PLUS" in r["features"]]
night=[r for r in pos if "夜" in r["features"]]
eng=[r for r in pos if "英文" in r["features"] or "Good" in r["features"]]
L=[]
L.append("# 左邻右舍门头数据采集报告 v0（browser relay 自主采集）\n")
L.append("采集时间：2026-11-11。通道：已登录中转 Chrome（CDP 9222）+ Bing 图片搜索 + 6 条种子源页 + 驰骋官网 SI 手册。\n")
L.append("性质：**公开来源候选集 + 视觉模型逐张核验**，非人工双盲金标准；权利状态全部 pending_review，**默认仅可链接、未授权不得用于训练**（设计文档 §4.4）。\n")
L.append("## 1. 数量总览\n")
L.append(f"| 类别 | 标注数 | 去重后代表数 |")
L.append(f"|---|---|---|")
L.append(f"| 正例（左邻右舍真门头实拍） | {len(pos)} | {len(pos_rep)} |")
L.append(f"| 负例（合计） | {len(neg)} | {len(neg_rep)} |")
L.append(f"| 总计 | {len(reg)} | {len(pos_rep)+len(neg_rep)} |")
L.append("")
L.append(f"**独立门店线索（店号）**：{dict(sids)}。其余正例店号不可辨，store_id 留空（不猜）。\n")
L.append(f"**正例属性覆盖**：含 PLUS {len(plus)} 张；夜景 {len(night)} 张；含英文副标 {len(eng)} 张。\n")
L.append("## 2. 负例分类（高价值困难负样本）\n")
L.append("| 负例类型 | 数量 | 说明 |")
L.append("|---|---|---|")
desc={"same_name_unrelated":"同名无关店：左邻右舍社区火锅/便民服务中心/门诊部/餐馆——品牌鉴别关键负例",
"confusable_brand":"混淆品牌：邻几GOOD SPOT/佐邻佑邻/美邻优/好邻居/99便利店/甄选——细粒度负例",
"competitor":"竞品便利店：美宜佳/7-11/全家/罗森/快客/十足",
"render_stock":"效果图/3D渲染/素材站模板（昵图/千图/包图）——非实拍，须排除出真实测试集",
"indoor":"纯室内/货架/商品/品牌展示墙——非门头",
"logo":"品牌logo/平面素材/书法插画",
"map":"地图截图/无招牌街景",
"uncertain":"疑似渲染或招牌不清，待原图复核"}
for k,v in nt.most_common():
    L.append(f"| {k} | {v} | {desc.get(k,'')} |")
L.append("")
L.append("## 3. 去重与泄漏控制\n")
L.append("- 字节级 sha256 去重 + 256-bit dHash 感知哈希近重复聚类（Hamming≤12）。")
L.append(f"- 正例 {len(pos)} 张标注 → {len(pos_rep)} 个唯一门头；{len(pos)-len(pos_rep)} 张为种子源页与 Bing 的转载重复（如 No.0062 同时出现在 Bing 与百度加盟星）。")
L.append("- 每个近重复簇赋 photo_family=fam_<cluster>，train/val/test 须按 store_id + photo_family 双重隔离（设计文档 §4.4）。")
L.append("- **印证设计文档警告**：公开图大量互相转载，凑张数会制造虚假样本量；本报告以唯一代表数为准。\n")
L.append("## 4. 官方 E2 形象标准（详见 OFFICIAL_E2_FINDINGS.md）\n")
L.append("- 官网三版独立页：简约版/jyb、标准版/bzb、都市版/dsb；下载 16 张官方 SI 手册图（约28MB，含 2–3 万像素长图，已切 82 段）。")
L.append("- 品牌 VI：标准红 #e60012（PANTONE 3346C）/辅助蓝 #005dac（PANTONE 2175C）；标志「左邻·右舍」⊙；口号「亲如一家 Good Neighbors」。")
L.append("- 门头规范（B3）：红底白字吸塑灯箱主招牌（2000×1000mm）+彩钢条「白蓝白红白」色带（总高600mm）+350×700mm柱灯箱+防水飘檐。")
L.append("- **关键**：官方 SI 标准门头**无 PLUS、无英文副标**；PLUS 是真实变体（简约版品牌故事页夜景实拍 + 搜狐稿 P05_00 实证），非官方标准版，对应加盟/升级版本待品牌方确认（E3）。")
L.append("- 视觉版本（L1）≠加盟合同（L4）；官方未公布对应规则，franchise_type 默认 null。\n")
L.append("## 5. 缺口与下一步（诚实披露）\n")
L.append(f"- 设计文档目标 30–50 张可核验独立实拍；当前**唯一已核验正例 {len(pos_rep)} 张**，距 30 张下限尚差 {max(0,30-len(pos_rep))} 张。")
L.append("- 缺口主因：公开搜索结果约 80% 为素材站效果图/无关店；真门头稀疏且多转载。")
L.append("- 补齐路径：① 地图 POI 商户图（百度/高德门店相册，按实体门店去重）；② 品牌方授权门头形象图+门店台账（确立 E2/E3 真值）；③ 实地/街景采集。")
L.append("- 权利：全部 pending_review；进入训练前须逐张完成授权审查（设计文档 E2/E3 门禁）。")
L.append("- 标注：当前为视觉模型单轮核验，正式金标准须双人独立盲标+第三方裁决（§4.4）。\n")
L.append("## 6. 产物清单\n")
L.append("- registry_v0.csv / registry_v0.json：483 行完整登记表（25 字段，含 sha256/感知哈希/photo_family/权利状态）。")
L.append("- images/：511 Bing 候选 + 129 种子图（69 唯一）；official/：16 张官方 SI 图 + slices/ 82 段。")
L.append("- sheets/：24 页候选核验表 + 种子核验表 + positive_final.jpg 正例终核表。")
L.append("- OFFICIAL_E2_FINDINGS.md：官方形象标准证据。")
L.append("- bing_candidates.json / download_log.json / image_meta.json / seed_pages.json / verified_labels.json / seed_labels.json：可追溯中间产物。")
open("COLLECTION_REPORT.md","w").write("\n".join(L))
print("report written")
print("positive_rep",len(pos_rep),"neg_rep",len(neg_rep),"total",len(reg))
print("plus",[r["image_id"] for r in plus])
print("store_ids",dict(sids))