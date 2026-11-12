#!/usr/bin/env python3
# 视觉核验结果（vision-model 逐张核验，24 页 contact sheet）
# label: positive=左邻右舍便利店/超市真门头实拍; negative=负样本
# neg_type: same_name_unrelated / confusable_brand / competitor / render_stock / indoor / logo / map / uncertain
POS = {
 0:"红底白字左邻·右舍 No.0062 蓝红门头 店内货架",
 3:"蓝白+红招牌左邻·右舍 街边玻璃门",
 4:"红蓝撞色左邻·右舍 红灯笼 转角店",
 5:"招牌特写白字左邻右舍 英文小字 阳光光斑 偏模糊",
 7:"蓝门楣红字左邻·右舍 红卷帘门柱",
 8:"红砖楼 蓝白横幅左邻·右舍",
 9:"夜景 蓝底白字灯箱左邻·右舍 行人",
 11:"蓝横幅左邻·右舍 卷帘半开 街景",
 13:"蓝招牌左邻·右舍 居民楼底层 灯笼",
 14:"蓝门楣左邻·右舍 开放式门面",
 15:"蓝横幅左邻·右舍 门口防水布 似装修",
 17:"夜景左邻·右舍 图虫广西龙卷风水印",
 35:"红招牌白字左邻右舍超市 欢迎光临 自行车",
 36:"夜景深蓝宽幅左邻·右舍 排队人群",
 37:"白天红字左邻·右舍 蓝遮阳棚 卷帘半开 三轮车",
 140:"夜景左邻·右舍 蓝底红蓝色带 疑PLUS",
 155:"夜景红蓝横条左邻·右舍 壹球供应链水印",
 182:"夜景红蓝双色左邻·右舍 疑PLUS副标 超市货架",
 207:"白天红蓝色带左邻·右舍 No.0002 英文副标 无水印",
 208:"白天红蓝色带左邻·右舍 英文 无水印",
 209:"白天蓝底左邻·右舍 侧竖招 无水印",
 342:"白天深蓝底金色行书左邻·右舍 红小图标 电动车",
 465:"街角两层门头红字左邻·右舍 蓝底 促销堆头 三轮车",
 479:"玻璃幕墙底层红门头左邻·右舍 旁中国石化 街景",
}
NEG = {
 6:("competitor","美宜佳便利店"),
 16:("render_stock","驿彩UICOU 3D渲染"),
 18:("render_stock","黑底招牌效果图 疑渲染"),
 19:("render_stock","上好佳社员国潮设计稿"),
 20:("render_stock","小美便利店效果图"),
 21:("render_stock","生鲜超市效果图"),
 23:("logo","左邻右舍ZUOLIN YOUSHE房屋logo素材"),
 24:("same_name_unrelated","品味小厨餐馆"),
 28:("render_stock","他店效果图"),
 29:("logo","左邻·右舍Good Neighbors平面logo"),
 30:("render_stock","他店效果图"),31:("render_stock","他店效果图"),33:("render_stock","他店效果图"),34:("render_stock","他店效果图"),
 38:("competitor","7-Eleven夜景"),
 39:("render_stock","他店效果图"),40:("render_stock","他店效果图"),41:("render_stock","他店效果图"),42:("render_stock","他店效果图"),
 43:("indoor","店内实拍非门头"),44:("indoor","店内面包区"),
 50:("render_stock","他店效果图"),51:("confusable_brand","Let's购绿色门头"),53:("render_stock","他店效果图"),
 125:("competitor","全家FamilyMart"),
 132:("logo","左邻ZUOLIN生活超市logo"),
 154:("render_stock","灯箱广告素材"),
 173:("indoor","纯室内货架"),
 227:("uncertain","左邻右舍黑门头夜景 疑渲染 满月横幅"),
 243:("confusable_brand","佐邻佑邻"),
 265:("map","楼房街景无招牌"),
 276:("indoor","左邻右舍室内品牌展示墙"),
 310:("same_name_unrelated","左邻右舍社区火锅 夜景"),
 311:("same_name_unrelated","左邻右舍社区火锅 夜景"),
 312:("same_name_unrelated","左邻右舍黑竖招 新华网水印 疑社区服务"),
 313:("render_stock","室内文化墙效果图"),314:("logo","红色宣传海报logo"),317:("render_stock","室内文化墙效果图"),
 319:("same_name_unrelated","睦邻室内装饰墙"),
 321:("render_stock","效果图"),322:("indoor","室内背景墙"),323:("indoor","室内背景墙"),325:("render_stock","效果图"),326:("render_stock","效果图"),
 327:("same_name_unrelated","石碑砖墙标识"),328:("uncertain","红门头无清晰左邻右舍字"),
 329:("same_name_unrelated","舍下物靓汤土豆粉"),330:("same_name_unrelated","火锅店"),331:("render_stock","效果图"),
 333:("same_name_unrelated","门诊部黄门头"),334:("render_stock","效果图"),335:("same_name_unrelated","学生私房早餐"),
 337:("same_name_unrelated","左邻右舍便民服务中心石墙"),338:("render_stock","效果图"),339:("render_stock","自流平杰深行效果图"),340:("render_stock","效果图"),
 341:("logo","logo素材"),349:("render_stock","效果图"),350:("render_stock","效果图"),358:("render_stock","横幅"),365:("render_stock","素材"),
 374:("render_stock","左邻右舍卡通插画墙面"),
 394:("logo","心形logo设计稿"),395:("logo","书法字素材"),396:("uncertain","施工中无清晰招牌"),400:("logo","插画素材"),
 423:("competitor","罗森LAWSON"),424:("competitor","7-Eleven"),427:("competitor","快客粉橙招牌"),428:("competitor","全家FamilyMart"),429:("competitor","7-Eleven"),433:("competitor","全家FamilyMart"),
 446:("confusable_brand","美邻优MEILINYOU"),447:("confusable_brand","美邻优MEILINYOU"),
 448:("map","百度地图定位截图"),449:("indoor","超市内景"),450:("indoor","超市内景货架"),
 460:("confusable_brand","邻几GOOD SPOT红底白字"),461:("confusable_brand","邻几GOOD SPOT侧面"),
 468:("confusable_brand","好又多全选生活超市效果图"),469:("confusable_brand","合福超市横幅"),
 569:("confusable_brand","甄选超市真实街景实拍"),
}
# 批量效果图/素材负例（昵图网/千图/包图等素材站，他牌超市门头渲染）
RENDER_BATCH = list(range(53,125))+list(range(157,173))+list(range(197,207))+list(range(211,227))+list(range(244,265))+list(range(277,310))+list(range(343,349))+list(range(351,358))+list(range(359,365))+list(range(366,374))+list(range(375,394))+list(range(401,423))+list(range(430,446))+list(range(451,460))+list(range(462,465))+list(range(470,479))+list(range(480,495))+list(range(496,580))
import json
meta=json.load(open("image_meta.json"))
by_i={m["i"]:m for m in meta}
rows=[]
for i,feat in POS.items():
    if i in by_i: rows.append({"i":i,"label":"positive","neg_type":"","features":feat,**{k:by_i[i][k] for k in ("file","sha","w","h","cluster","url","purl","host","query")}})
for i,(nt,feat) in NEG.items():
    if i in by_i: rows.append({"i":i,"label":"negative","neg_type":nt,"features":feat,**{k:by_i[i][k] for k in ("file","sha","w","h","cluster","url","purl","host","query")}})
seen={r["i"] for r in rows}
for i in RENDER_BATCH:
    if i in by_i and i not in seen and by_i[i].get("is_rep"):
        rows.append({"i":i,"label":"negative","neg_type":"render_stock","features":"他牌超市/便利店门头效果图素材",**{k:by_i[i][k] for k in ("file","sha","w","h","cluster","url","purl","host","query")}})
        seen.add(i)
rows.sort(key=lambda r:r["i"])
json.dump(rows, open("verified_labels.json","w"), ensure_ascii=False, indent=1)
from collections import Counter
print("total_labeled", len(rows))
print("positive", sum(1 for r in rows if r["label"]=="positive"))
print("negative", sum(1 for r in rows if r["label"]=="negative"))