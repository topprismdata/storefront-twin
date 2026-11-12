# 左邻右舍全面门店数字孪生：行业案例与技术路线报告

<text color="secondary" size="xs">技术决策评审版 V1.0 · 2026年11月12日</text>

**研究对象：** 左邻右舍（驰骋控股）门店数字孪生项目  
**研究范围：** 行业案例、Sim-to-Real 域差、合成数据、组件架构、工具链、数据验证、权利治理  
**证据基础：** 《GPT_TWIN_RESEARCH_BRIEF.md》，区分项目实测、外部案例、研究推断和待验证假设

## 一、执行摘要与总体裁决

<row align="center" gap={2} wrap="wrap">
  <badge color="warning">总体裁决：REVISE</badge>
  <text color="secondary" size="xs">路线方向认可，当前验证方案尚不足以进入业务训练和部署</text>
</row>

**核心结论：全面门店数字孪生值得继续建设，但现阶段必须把“数字孪生资产建设”和“识别模型有效性”分开验收。**

行业案例能够支持从门头向全店扩展的架构方向，却不能证明当前合成门头数据已经具有实际识别价值。

尤其需要纠正三个潜在误判：

1. **AuPR 0.324 是风险信号，不是域差根因已经被证明的证据。** 需要先核实指标定义、正例比例和评估协议，再通过受控消融判断失败究竟来自域差、合成样本覆盖不足、探针表示能力、目标错位还是融合方式。
2. **视觉真实感评分从约 3 提升到 5.5–7/10，不等于识别能力提高。** 真实感属于感知质量指标；是否有利于真实样本排序，只能依靠独立验证。
3. **受控货架商品检测接近 99% 的结果不能外推到开放街景门头。** 场景结构、类别边界、背景变化、相机分布和数据治理条件均不相同。

### 当前项目证据快照

<box border radius="xl" padding={3} gap={3}>
  <grid columns={2} gap={3}>
    <grid-item>
      <text color="secondary" size="xs">E8-G 最佳管线 F1</text>
      <title size="xl" color="default" tabularNums>0.862</title>
      <caption>330 张评估集 · 30 正例</caption>
    </grid-item>
    <grid-item>
      <text color="secondary" size="xs">E10 VLM 证据层 F1</text>
      <title size="xl" color="default" tabularNums>0.93–0.95</title>
      <caption>探索性，配对 Bootstrap 差异不显著</caption>
    </grid-item>
    <grid-item>
      <text color="secondary" size="xs">合成探针 AuPR</text>
      <title size="xl" color="danger" tabularNums>0.324</title>
      <caption>报告为负贡献，需补评估协议</caption>
    </grid-item>
    <grid-item>
      <text color="secondary" size="xs">当前三维孪生 v1</text>
      <title size="xl" color="default" tabularNums>820</title>
      <caption>低多边形示意渲染</caption>
    </grid-item>
    <grid-item>
      <text color="secondary" size="xs">真实采集候选图</text>
      <title size="xl" color="default" tabularNums>2,542</title>
      <caption>尚未获得训练授权</caption>
    </grid-item>
    <grid-item>
      <text color="secondary" size="xs">唯一门店 POI</text>
      <title size="xl" color="default" tabularNums>1,178</title>
      <caption>浙江占 95%，存在地域偏置</caption>
    </grid-item>
  </grid>
  <divider color="subtle"/>
  <caption>全部为附件报告数据；其中 F1、AuPR 与真实感评分并非同一评价维度。<FileCite ref="turn0file0" line_range_start={6} line_range_end={12}/></caption>
</box>

### A–F 裁决总览

| 问题 | 裁决 | 工程决策 |
|---|---|---|
| A 行业案例与路线 | PARTIAL | 接受全店组件化方向，拒绝准确率外推 |
| B AuPR 根因与修复 | OPEN | 优先补证据、做负贡献归因和随机化消融 |
| C 工具选型 | PARTIAL | 近期 Blender，暂不启动全面 Omniverse 迁移 |
| D 架构与真值 | PARTIAL | 接受组件化，先冻结最小语义真值协议 |
| E 数据规模 | PARTIAL | 停止以合成数量代替真实域有效性验证 |
| F 治理边界 | PARTIAL | 认可硬约束，工程上实现可审计的强制阻断 |

**本次 REVISE 不否决数字孪生战略，而是要求在继续扩大合成生产线之前，关闭影响结论可信度的前置缺口。**

## 二、行业案例研究：哪些经验可以迁移？

### 2.1 四类案例的证据价值

| 案例 | 已报告的能力 | 对左邻右舍的价值 | 不能证明什么 |
|---|---|---|---|
| Kinetic Vision + NVIDIA | 合成生成、孪生仿真、边缘部署与反馈闭环；开发时间和成本约为物理环境方法的 10% | 验证闭环架构方向 | 不证明我们能够获得相同成本收益或精度 |
| 台湾 7-Eleven | 货架检测 mAP@50 99.41%；商品检测 95.7%；分类准确率 99.86% | 货架、SKU、陈列合规的直接参考 | 不证明开放门头品牌识别能达到 99% |
| Neurolabs ZIA | 1,200 张合成货架图、129 SKU、约 17,000 条标注 | 合成场景自动真值、结构与外观变化 | 不证明纯合成训练能够跨越复杂街景域差 |
| Clobotics / Infosys | 规模化货架视觉分析；Replicator、TAO、DeepStream 组合 | 全店扩展后的工程参考 | 不提供我们的门头任务准确率保证 |

来源：附件第二节。<FileCite ref="turn0file0" line_range_start={14} line_range_end={41}/> 其中 NVIDIA 技术博客、7-Eleven 论文与 Neurolabs 数据集说明已补充核对。<Cite refs={["turn771057view0","turn977419search0","turn771057view3"]}/>

### 2.2 特别重要：7-Eleven 的 99% 究竟是什么？

<box gap={2}>
  <row align="center" gap={3}>
    <AsyncImage query="convenience store planogram packed shelves products supermarket aisle photography" aspectRatio="5:4" maxWidth="156px"/>
    <box flex="1" gap={1}>
      **受控货架场景**

      <text color="secondary" size="sm">具有固定陈列结构、明确商品区域、可建立 SKU 目录和标准陈列序列。识别对象虽多，但结构先验较强。</text>
    </box>
  </row>
  <divider color="subtle"/>
  <row align="center" gap={3}>
    <AsyncImage query="China urban convenience store storefront street facade clutter parked scooters street photography" aspectRatio="5:4" maxWidth="156px"/>
    <box flex="1" gap={1}>
      **开放门头场景**

      <text color="secondary" size="sm">建筑形态、遮挡、招牌修改、竞品相似标识、拍摄距离、透视和光照都可能独立变化。</text>
    </box>
  </row>
</box>

7-Eleven 研究并不存在一个可以直接引用为“全店数字孪生准确率 99%”的统一指标。

其实际报告包含多个独立任务：

- 货架检测：mAP@50 = 99.41%。
- 商品检测：mAP@50 = 95.7%。
- 商品分类：准确率 = 99.86%。
- 未见商品少样本分类：Top-1 = 98.39%。

这些值的分母、任务对象和评估方式均不同。也不能据此推断端到端陈列合规的准确率。

另外，该论文描述了面向 7,000 多家门店的应用，但其具体模型数据采集来自少量实际门店。因此，应将**规模化应用范围、训练数据覆盖和模型测试结果**分别陈述。<Cite refs={["turn977419search0","turn977419search1"]}/>

对我们的真正启发不是直接采用 YOLOv8 或 ResNet101，而是：

**将复杂任务拆成具有结构先验、能独立验收的子任务，再建立可比较的空间表达。**

门头侧先解决品牌实体与经营场景判定；进入室内后，则采用货架定位、商品识别、空间排序、陈列规则比对的分层路线。

### A. 行业证据对“门头→全店”的支持程度

**裁决：PARTIAL**

已经可以确认组件化、合成数据、自动真值和分层识别具有行业实践依据，但尚无证据证明左邻右舍的开放门头识别能够直接继承受控货架的性能。

具体决策：批准架构层面的全店扩展设计，暂不批准根据行业数字制定本项目门头准确率目标。后续每个组件分别建立自己的基准、样本定义和验收指标。

---

## 三、B 问专题：AuPR 0.324 的根因诊断与修复

<badge color="danger">当前最高技术优先级</badge>

### 3.1 首先纠正一个关键统计解释

附件写道：合成探针 AuPR = 0.324，小于 0.5，因此属于负贡献，体现纯合成训练的 Sim-to-Real 域差。<FileCite ref="turn0file0" line_range_start={6} line_range_end={10}/>

这里有一个必须澄清的问题：

**AuPR/Average Precision 不能采用 0.5 作为通用有效性门槛。**

对于以正例比例为基准的 PR 曲线，随机排序的期望 Average Precision 通常接近正例比例，而非 0.5。

若 330 张评估集中的 30 张正例正是探针所用评估集，则：

\[
\pi=\frac{30}{330}\approx0.0909
\]

<grid columns={2} gap={2}>
  <grid-item>
    <box border radius="lg" padding={3} gap={1}>
      <text color="secondary" size="xs">正例比例参考值</text>
      <title size="xl" tabularNums>0.091</title>
      <caption>仅在相同评估集条件下成立</caption>
    </box>
  </grid-item>
  <grid-item>
    <box border radius="lg" padding={3} gap={1}>
      <text color="secondary" size="xs">已报告探针 AuPR</text>
      <title size="xl" tabularNums>0.324</title>
      <caption>不能仅凭低于 0.5 解释为反向预测</caption>
    </box>
  </grid-item>
</grid>

如果上述条件一致，0.324 反而表明探针可能具备高于随机排序的区分能力。

但这**不代表它对现有 E8-G 有正增益**。

这里必须分别测试三个命题：

| 命题 | 证据现状 | 判断 |
|---|---|---|
| H1：探针自身具有真实域排序信息 | 只有 AuPR 0.324 | 尚不能完整判断 |
| H2：探针加入融合后降低总体性能 | 附件报告“负贡献”，但缺少完整消融表 | 待独立复核 |
| H3：负贡献由 Sim-to-Real 域差主导 | 目前缺少定位性实验 | 未证实 |

**最重要的技术结论：不能把“探针有信息，但融合后损害效果”与“探针没有信息”混为一谈。** 两种情况的修复方式完全不同。

### 3.2 根因树：先定位，再增加合成资产

<box border radius="xl" padding={3} gap={2}>
  <svg viewBox="0 0 340 300" width="100%" xmlns="http://www.w3.org/2000/svg">
    <rect x="85" y="5" width="170" height="35" rx="8" fill="#334155"/>
    <text x="170" y="19" fontSize="12" fontWeight="600" fill="white" textAnchor="middle">AuPR 0.324</text>
    <text x="170" y="32" fontSize="9" fill="white" textAnchor="middle">融合负贡献待核</text>
    <path d="M170 40 V57 M43 57 H297 M43 57 V72 M127 57 V72 M213 57 V72 M297 57 V72" stroke="currentColor" strokeWidth="1.2" fill="none" opacity=".5"/>
    {#each [{x:3,t:"评估口径",s:"指标/泄漏"},{x:87,t:"融合冲突",s:"尺度/校准"},{x:173,t:"真实域差",s:"场景/视角"},{x:257,t:"探针设计",s:"标签/表示"}] as n}
      <rect x={n.x} y="72" width="80" height="48" rx="6" fill="#64748b" fillOpacity=".12" stroke="#94a3b8" strokeOpacity=".65"/>
      <text x={n.x+40} y="91" fontSize="11" fill="currentColor" textAnchor="middle" fontWeight="600">{n.t}</text>
      <text x={n.x+40} y="107" fontSize="9" fill="currentColor" opacity=".7" textAnchor="middle">{n.s}</text>
    {/each}
    <path d="M213 120 V142 M55 142 H285 M55 142 V159 M132 142 V159 M209 142 V159 M285 142 V159" stroke="currentColor" strokeWidth="1.2" fill="none" opacity=".4"/>
    {#each [{x:16,a:"几何",b:"相机/透视"},{x:94,a:"材质",b:"字体/反射"},{x:171,a:"环境",b:"遮挡/街景"},{x:248,a:"采样",b:"难负例"}] as n}
      <rect x={n.x} y="159" width="76" height="49" rx="6" fill="#d97706" fillOpacity=".10" stroke="#d97706" strokeOpacity=".5"/>
      <text x={n.x+38} y="178" fontSize="11" fontWeight="600" fill="currentColor" textAnchor="middle">{n.a}</text>
      <text x={n.x+38} y="195" fontSize="9" fill="currentColor" textAnchor="middle">{n.b}</text>
    {/each}
    <rect x="20" y="242" width="300" height="43" rx="8" fill="#059669" fillOpacity=".10" stroke="#059669" strokeOpacity=".6"/>
    <text x="170" y="258" fontSize="12" fontWeight="600" fill="currentColor" textAnchor="middle">实验定位 → 单因素修复 → 冻结集复核</text>
    <text x="170" y="273" fontSize="10" fill="currentColor" textAnchor="middle">禁止先扩大数据量，再以合成集分数证明成功</text>
    <path d="M170 208 V242" stroke="#059669" strokeWidth="1.4" strokeDasharray="4 3"/>
  </svg>
  <caption>工程诊断框架，属于本报告提出的验证假设，而非已经查明的项目根因。</caption>
</box>

我建议按以下顺序排查，而不是立即把精力投入更高精度的 Blender 渲染。

#### 第一步：确认失败到底发生在哪里

冻结 E8-G 当前模型输出和探针输出，不重新训练，先补以下表：

| 对照 | 目的 |
|---|---|
| CLIP-only | 确认通用视觉语义基线 |
| DINOv2-only | 确认视觉结构表征能力 |
| Synthetic probe-only | 测量探针自身真实域排序能力 |
| CLIP + DINOv2 | 得到无合成探针基线 |
| CLIP + DINOv2 + probe | 测量实际融合增益 |
| 去掉 probe 的 E8-G 复算 | 确认所谓“负贡献” |

所有对照必须使用相同 330 张评估集、相同标签、相同数据切分与固定的判定规则。

需要注意：如果融合权重或阈值曾经使用这 330 张样本选择，就不能把它再视作独立测试集；应将该结果标为探索性，并在新的冻结集上最终裁决。

主要报告：

\[
\Delta AP=AP_{\text{with probe}}-AP_{\text{without probe}}
\]

\[
\Delta F1=F1_{\text{with probe}}-F1_{\text{without probe}}
\]

使用门店组级重采样计算配对不确定区间，避免同一家门店多张近似照片被当成独立样本。对 AP 差异尤其要保留原始成对分数，并声明 Bootstrap 的具体采样单位和置信区间方法。

这里有一个重要区别：**AP 是排序指标，F1 是阈值决策指标。** 如果 AP 改善而 F1 下降，首先检查校准和阈值，而不是认定合成训练失败。

#### 第二步：诊断是否属于融合负迁移

如果 probe-only 明显优于随机排序，但融合效果不佳，优先检查：

- **分数尺度不一致：** CLIP、DINOv2 和 probe 的相似度不可直接假定同标尺。
- **错误相关性：** probe 在 CLIP 已经正确的样本上产生干扰，却没有补足 CLIP 的困难样本。
- **负例表示不足：** 合成品牌和真实竞品之间的距离没有学到有效边界。
- **权重或阈值过拟合：** 在 30 个正例的评估集上调优容易产生不稳定的选择结果。
- **证据目标错位：** 探针可能学习了招牌图案相似度，而最终任务要求品牌身份与真实经营场景同时成立。

具体工程实现：输出逐样本 `score_clip`、`score_dino`、`score_probe`、`score_fused`、`truth`、`store_group_id` 及错误类型；复算 probe 增加后每个样本的排名变化。

**先研究探针在哪些样本上有独立贡献。** 如果全部收益集中在极少数样本，就不应采用全局固定权重强制融合。

可以将 probe 暂时降级为诊断性证据，不参与线上决策；只有独立验证证明条件性增益后，再考虑条件化启用。

#### 第三步：验证 Sim-to-Real 域差，而不是直接假设它

目前 v2 已扩大相机范围，加入材质、磨损和传感器噪声。<FileCite ref="turn0file0" line_range_start={9} line_range_end={10}/>

问题在于这些修改是组合实施的。若最终效果改变，仍无法解释是哪种变化带来的增益。

建议使用受控消融：

| 实验臂 | 改动 | 可检验假设 |
|---|---|---|
| S0 | 复现 v1 | 确立基准 |
| S1 | 仅增加相机随机化 | 透视/尺度覆盖是否是主因 |
| S2 | S1 + 立面几何变化 | 模型是否依赖单一建筑模板 |
| S3 | S2 + 光照/材质随机化 | 纹理与照明是否导致域差 |
| S4 | S3 + 遮挡/竞品/环境 | 真实复杂背景与难负例是否关键 |
| S5 | S4 + 传感器扰动 | 后处理噪声是否有额外收益 |

建议各臂尽量固定总图量、类分布、随机种子、训练步数和骨干网络，只改变目标因素。这里的 S1–S5 是逐步累加实验，适合判断新增模块的边际价值；若要精确分离各因素交互，还需要追加单因素或小型因子实验。

应重点统计如下错误切片：

- 招牌角度：正面、斜角、大透视；
- 拍摄条件：白天、夜间、曝光异常；
- 场景复杂度：遮挡、反射、背景拥挤；
- 身份难度：标准门头、旧版门头、视觉相似竞品；
- 场景类型：真实经营入口、室内、地图、海报、纯 Logo。

其中真实数据的切片检查只能使用治理批准范围内的资产。不能因为准备进行消融实验而绕过权利门。

### 3.3 域随机化够不够？

结论是：**值得验证，但没有证据表明它单独足够。**

Tobin 等人的 2017 年研究支持一个重要思想：在特定视觉定位任务中，多样性较高的非照片级渲染也可能成功迁移到真实世界。其任务主要是机器人目标定位，并非便利店品牌身份识别。<Cite ref="turn771057view2"/>

因此，我们不应把“继续提升渲染真实感”视为唯一方向。

对于门头身份识别，另一个特别重要的因素是**判别语义的准确性**：

- 目标品牌标识的形状、比例、文字、配色需要准确且有合法使用依据。
- 竞品的相似圆环、红蓝颜色和同类招牌布局，必须在合成场景中形成困难负例。
- 真实经营入口与平面设计稿之间，必须有独立的场景结构边界。
- 合成随机化不能把这些身份关键特征随机破坏。

换句话说：

**随机化应该放大无关因素的变化，保留决定品牌身份的因素。**

这比统一增加照片级材质更有可能解决与当前任务有关的问题，但仍需要实验验证。

### 3.4 是否必须引入真实锚定？

需要区分两种完全不同的“锚定”。

| 锚定类型 | 当前建议 | 治理边界 |
|---|---|---|
| 真实域评估锚定 | 必要 | 仅使用已获相应授权的独立评估资产 |
| 真实照片参与训练 | 当前禁止 | G0/G1 未过不得实施 |
| 从真实照片制作背景或贴图 | 始终禁止 | 即使以后获训练许可也不自动解除该禁令 |
| 来自合法独立来源的抽象场景统计 | 可设计方案 | 实际使用前审查来源许可和数据用途 |
| 真实照片生成的 VLM 裁决标签 | 禁止作为训练真值 | 只能作为推理证据 |

附件引用的混合训练论文表明，在特定数据集上混合真实与合成样本可能改善域迁移，但其中的比例并不是跨任务通用最优值。特别是 60%–80% 合成比例不能直接成为左邻右舍的训练配置。<Cite refs={["turn977419view0","turn977419view1"]}/>

在 G0/G1 尚未通过之前，本项目的可行路径是：

**合成侧构建和离线自检 → 合法可用评估资产进行固定协议验证 → 根因归因 → 权利解锁后才决定真实数据训练策略。**

如果当前连既有 330 张评估集的持续复核权利都无法确认，则新的真实域实验同样应暂停，保留已有结果并标记其权利状态。不能以“只是研究”作为绕过使用权限的理由。

### 3.5 B 问最终裁决

**OPEN：尚不能判定域差是主要根因。**

建议先完成分数级审计和去探针融合消融，再决定是否投入下一轮合成数据生产。

修复方向采用以下条件决策：

| 诊断结果 | 下一步 |
|---|---|
| Probe AP 优于随机，但融合负增益 | 优先修复校准、融合权重和目标条件 |
| Probe 自身排序差，角度切片恶化 | 优先改善相机和几何分布 |
| 品牌相似竞品误判集中 | 优先增加合法构造的难负例 |
| 真实经营入口与 Logo 图混淆 | 增加独立的场景有效性判定，不让品牌探针单独裁决 |
| 所有受控消融均无稳定真实域增益 | 停止继续扩张该探针路线，将合成资产保留作渲染和系统测试用途 |

**核心验收条件：不是合成图越真实、AuPR 越接近 1 就批准，而是它必须在独立真实域评估上提供可复现、具有明确任务意义的增量价值。**

---

## 四、C 问：BlenderProc 与 Omniverse 的工具路线

**裁决：PARTIAL — 近期选型可以确定，中期迁移尚缺真实成本证据。**

### 4.1 工具选型结论

| 维度 | Blender / BlenderProc | Omniverse Replicator |
|---|---|---|
| 当前资产复用 | 高，已有 Blender 资产 | 需要适配 |
| 场景自动化 | Python 脚本、参数化渲染 | OpenUSD 与随机化工作流 |
| 自动真值 | 可生成分割、深度等 | 支持多种 Synthetic Data Annotator |
| 近期门头实验 | 优先 | 暂不需要 |
| 多组件全店场景 | 可实现，但需维护装配逻辑 | 有更强的 USD 场景生态潜力 |
| 迁移风险 | 小 | 中等，需实际 PoC |
| 商业与许可 | GPL 义务需审查 | NVIDIA 软件许可需审查 |

附件工具链：<FileCite ref="turn0file0" line_range_start={48} line_range_end={55}/>。补充说明：NVIDIA 2026 年 5 月的许可说明显示，Omniverse 可在相应条款下免费用于开发和生产；但免费使用不等于没有许可约束或企业支持费用。<Cite refs={["turn771057search1","turn771057search2"]}/>

### 4.2 推荐工程安排

近期保留 Blender 作为主生成器，按以下顺序改造：

1. 将几何、材质、相机、照明、标签生成拆成配置驱动模块。
2. 增加可重放随机种子、渲染配置快照、逐图 manifest。
3. 通过统一输出契约对接训练/评估平台，而不是让训练代码依赖 Blender 场景对象。
4. 在门头域差问题定位后，再选择一个货架微场景评估 Replicator。

**不要现在进行全项目 Omniverse 迁移。** 迁移不会自动解决合成数据的语义偏差、真实场景分布差异或模型融合负贡献。

建议中期仅做一次对照 PoC：使用相同的一个货架组件，比较两条工具链的标注一致性、批量渲染吞吐、重放稳定性与工程维护成本。这些是未来是否迁移的验收依据，而非预先假设 Omniverse 一定更快。

---

## 五、D 问：全面门店数字孪生架构

**裁决：PARTIAL — 架构方向成立，尚需冻结可审计的数据契约。**

### 5.1 建议的目标架构

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **StoreTwin 统一场景与实体层**

    <text color="secondary" size="xs">Store → Facade / Entrance → Aisle → Shelf → Product</text>
  </box>
  <grid columns={2} gap={2}>
    {#each [{a:"C1 门头招牌",b:"已实现 · 身份与场景"},{a:"C2 建筑立面",b:"下一阶段 · 几何和入口"},{a:"C3 室内货架",b:"下一阶段 · 结构和位置"},{a:"C4 陈列合规",b:"依赖 SKU + Planogram"},{a:"C5 环境",b:"光照、遮挡、道路和邻店"},{a:"规则与语义",b:"品牌、门店、SKU、关系"}] as c}
      <grid-item>
        <box border radius="md" padding={2} gap="2px">
          <text weight="medium" size="sm">{c.a}</text>
          <text color="secondary" size="2xs">{c.b}</text>
        </box>
      </grid-item>
    {/each}
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="tertiary" size="lg"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **场景装配 + 条件化域随机化**

    <text color="secondary" size="xs">geometry / materials / camera / light / occlusion</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="tertiary" size="lg"/>
  </row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
        **渲染适配器**

        <text size="xs" color="secondary">BlenderProc / Replicator</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
        **真值生成器**

        <text size="xs" color="secondary">bbox / mask / depth / IDs</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="tertiary" size="lg"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **资产登记、权利门、评估与证据追踪**

    <text color="secondary" size="xs">Provenance · Rights · Dataset Manifest · Evaluation</text>
  </box>
</box>

该架构是对附件 StoreTwin 草案的工程细化，不代表组件已经全部实现。<FileCite ref="turn0file0" line_range_start={57} line_range_end={59}/>

### 5.2 统一真值 Schema 必须分两层

附件提出 bbox、polygon、segmentation、depth 和 SKU 的统一 schema，这是正确方向，但仅有图像标注字段还不够。

建议分为：

**第一层：场景实体真值**

`store_id`、`scene_id`、`entity_id`、`parent_entity_id`、`entity_type`、`brand_id`、`sku_id`、`geometry_id`、`coordinate_frame`、`transform`、`visibility`

**第二层：渲染观测真值**

`image_id`、`camera_id`、`bbox_xyxy`、`mask_id`、`depth_uri`、`occlusion_ratio`、`label_source`、`rights_status`、`renderer_version`、`seed`

特别需要区分：

- `brand_id`：品牌身份。
- `store_id`：具体门店实例。
- `sku_id`：商品身份。
- `planogram_id`：标准陈列定义。
- `observation_id`：某次拍摄或渲染观测。

它们不能互相替代。

例如同一品牌可以有很多门店，同一商品可以出现在不同货架，同一门店也有不同时间状态。

这正是从“渲染门头”走向“全面门店数字孪生”时需要补充的语义结构。

### 5.3 室内组件优先级

我建议优先完成**空货架几何与虚拟货架装配**，而不是立即开展 471 类甚至更多 SKU 识别。

工程顺序为：

`C1 门头 → C2 入口/立面 → C3 空货架 → 商品占位与SKU实体 → C4 陈列规则`

C5 环境从第一阶段起作为可随机化的共用组件同步建设，不必独立等待到最后。

空货架阶段可以验证几何重建、视角遮挡、实例分割、虚拟货架拼接和实体坐标关系，不依赖真实商品图像训练权限。SKU 身份识别和合规判断则留给后续具有合法素材、目录和标准陈列数据的阶段。

---

## 六、E 问：数据量现实与最小可行验证

**裁决：PARTIAL — 可以制定可执行方案，但目前缺少足够的独立真实域证据。**

### 6.1 现有数据不可简单相加

| 数据 | 数量 | 可做的事情 | 不应做的事情 |
|---|---:|---|---|
| 3D v1 渲染 | 820 张 | 合成算法开发、场景验证 | 当作真实域精度证据 |
| 高德候选图 | 2,542 张 | 权利和质量审查范围内的资产盘点 | 直接作为可训练数据 |
| 唯一门店 POI | 1,178 家 | 来源、地域覆盖、重复性统计 | 等同于 1,178 个有效正例 |
| E8-G 评估集 | 330 张 | 在既有许可范围内复核 | 无限制重复调参并称独立测试 |
| 7-Eleven 商品检测 | 99,135 张 | 研究数据规模参考 | 直接制定我们需要的数据量 |

数据来源：附件第一、二节。<FileCite ref="turn0file0" line_range_start={6} line_range_end={31}/>

行业数据量远大于当前项目，并不意味着必须先积累同等数据量才能开展研究。真正重要的是实验能够区分不同工程方案，而不是单纯增加图片数量。

### 6.2 建议两层基准

**层一：合成侧验证基准（G0/G1 前可执行）**

完全由合法自建组件生成，包含已知真值，用于验证相机随机化、标注完整性、空间关系、几何遮挡、模型代码与端到端流程。

它的结果只能宣称合成域表现。

**层二：独立真实域验证基准（需相应权利审批）**

需要具备人工独立确认的品牌身份与场景类别标签，采用门店组隔离。保留完整的正例、竞品、难负例及四态裁决边界，并为 UNKNOWN 和 OUT_OF_SCOPE 明确评估规则。

对于 E10 的探索性改进，附件已经说明配对 Bootstrap 差异不显著，因此不能把 F1 0.93–0.95 宣称为确定优于 E8-G。<FileCite ref="turn0file0" line_range_start={6} line_range_end={9}/>

### 6.3 建议的最小实验矩阵

| 编号 | 工作包 | 交付 | 放行条件 |
|---|---|---|---|
| V0 | 现有实验取证 | 指标定义、样本 ID、逐图分数、融合配置 | 0.324 可独立复算 |
| V1 | 探针与融合消融 | 6 组对照及配对差异 | 明确负贡献发生位置 |
| V2 | 合成域随机化消融 | S0–S5 的版本化实验 | 确认有效因素而非只看真实感 |
| V3 | StoreTwin 最小组件 | 门头、立面、空货架及统一真值 | 场景/渲染/标注一致 |
| V4 | 真实域复核 | 独立许可与金标准审核 | 授权合法且无训练/测试泄漏 |
| V5 | 下一阶段决策 | Go / Revise / Stop | 基于真实域增量证据 |

V0、V2 的合成侧工作、V3 的合法自建资产可以先做；V1 中涉及既有真实分数的重新利用以及 V4 的真实数据处理，必须依授权范围执行。

如果尚无合法真实域评估条件，V0 先审计已有汇总结果，V2–V3 可以继续，但不宣称已经解决域差。

---

## 七、F 问：权利治理与工程强制控制

**裁决：PARTIAL — 治理原则正确，但还需要证明技术实施的门禁能够阻断违规路径。**

### 7.1 G0/G1 未过时的工程边界

| 行为 | 当前裁决 |
|---|---|
| 自建参数化门头与货架几何 | 允许 |
| 自建材质、程序化纹理、合成渲染 | 允许，遵守相关权利和许可 |
| 合成数据内部训练实验 | 限于合法自建合成资产和研究范围 |
| 合成图像自动生成分割、bbox、depth | 允许 |
| 真实候选图训练或微调 | 禁止 |
| 把真实门头照片贴到 3D 立面 | 禁止 |
| 把真实门头照片用作合成背景 | 禁止 |
| 将 VLM 裁决用作模型训练标签 | 禁止 |
| 真实域业务准确率宣称 | 禁止 |
| 生产系统部署 | 禁止 |

依据：附件第一节权利约束。<FileCite ref="turn0file0" line_range_start={11} line_range_end={12}/>

### 7.2 应落实为程序，而不是仅靠研发人员遵守

建议每份数据资产都强制具有以下元数据：

```yaml
asset_id: synthetic_scene_0001
source_type: self_created
rights_status: synthetic_train_only
provenance_id: scene_manifest_0001
label_source: renderer_ground_truth
allowed_uses:
  - synthetic_generation
  - synthetic_internal_training
  - synthetic_evaluation
real_photo_as_texture: false
real_photo_as_background: false
```

对于真实候选资产，默认 `rights_status: pending_review`。任何训练任务发现该状态都必须阻断，不允许通过修改实验配置绕过。

在数据加载、训练入口、模型注册和部署入口分别进行权限检查，记录 dataset hash、来源标识和审批证据。

还应将 VLM 裁决与金标准标签分开存储，并设置明确的不可逆混用限制，防止未来在数据汇总时无意把 VLM 预测当成监督真值。

需要特别说明：**G0/G1 通过之前不存在“先用真实照片训练，训练完不部署就合规”的例外。** 训练本身已经是被禁止的行为。

---

## 八、整体工程路线和里程碑

建议以 8 周作为本轮研究验证周期上限，不把全店数字孪生建设压缩为一次完成。

<box border radius="xl" padding={3} gap={3}>
  <text weight="medium">第一轮：8 周验证计划（拟议，不是项目已完成进度）</text>
  <box gap={2}>
    {#each [
      {w:"W1–W2",t:"证据清理与根因审计",d:"完成指标定义、AuPR复算、融合消融；同时检查所有数据使用权"},
      {w:"W3–W4",t:"受控随机化实验",d:"完成 S0–S5 合成侧实验，记录每次随机化和真值一致性"},
      {w:"W5–W6",t:"StoreTwin 最小语义组件",d:"实现 C1/C2/C3 的独立构造、装配与自动标注"},
      {w:"W7–W8",t:"独立验收与下一阶段决策",d:"完成可执行范围内的复核；真实域部分未获授权则保持阻断"}
    ] as p,i}
      <row gap={3} align="start">
        <box background="surface-secondary" radius="md" padding={2} width="77px" align="center">
          <text weight="medium" size="xs">{p.w}</text>
        </box>
        <box flex="1" gap={1}>
          <text weight="medium" size="sm">{p.t}</text>
          <text color="secondary" size="sm">{p.d}</text>
        </box>
      </row>
    {/each}
  </box>
</box>

推荐定义三个分别验收的成果：

**资产成果：** 一个可复现、可装配、能够输出合法合成数据和统一真值的 StoreTwin 原型。

**算法成果：** 一份明确回答合成探针为什么有无增益的实验报告。即使结果为“没有可靠增益”，也是可以接受的研究结论。

**治理成果：** 一个能够在代码和数据层面执行 G0/G1 阻断，并保留证据链的系统。

这三类成果不应该被合并成一个单一 F1 指标。

## 九、总体裁决及新增 P0

<row align="center" gap={2}>
  <badge color="warning">REVISE</badge>
  <text weight="medium">有条件认可技术方向，不批准扩大真实数据训练及业务部署</text>
</row>

本次仅提出三项新增 P0，其他建设事项按前文计划执行，不再扩大阻断范围。

| 新增 P0 | 必须关闭的缺口 | 关闭标准 |
|---|---|---|
| P0-01 | 合成探针 0.324 的统计解释及融合负贡献尚不清楚 | 复算 AP/AuPR、正例基准、去探针对照和逐样本排名变化；归因结论可独立复核 |
| P0-02 | 真实域数据和模型结果的权利边界尚未被技术门禁保证 | 数据使用范围审计完成；未授权训练、贴图、背景和部署路径均被自动阻断 |
| P0-03 | 尚无冻结、可追溯的真实域验证协议支撑 Sim-to-Real 结论 | 冻结分组、指标、判定规则和统计方法；获合法评估权利后执行独立验证，否则保持未验证状态 |

**P0-01 关闭不要求探针一定取得正增益。** 如果独立验证证明它无增益，明确停止该路线，同样是有效关闭。P0-03 也不要求以获得授权为名允许提前训练；无法获得权利时，应如实维持验证阻断。

### 最终研究判断

全面门店数字孪生不应该只是生产越来越逼真的门店效果图，而应该建立一个具有**门店实体、空间结构、商品身份、动态状态、规则约束和证据追踪**的可计算系统。

现阶段最值得投入的不是更昂贵的渲染器，而是：

**证明合成信息到底能解决什么真实问题，以及在什么条件下它反而造成损害。**

只有这个问题得到可复核的回答，下一阶段扩大三维门店组件、SKU 生成、陈列合规和全店场景理解，才有可衡量的技术基础。

---

## 参考资料与核验边界

本报告基于附件《GPT_TWIN_RESEARCH_BRIEF.md》第一至第六节编制，并补充核对部分一手来源。

- <Link url="https://developer.nvidia.com/blog/deploying-highly-accurate-retail-applications-using-a-digital-twin" title="NVIDIA：Kinetic Vision 数字孪生应用案例（2021）"/>
- <Link url="https://doi.org/10.1038/s41598-025-27773-5" title="Scientific Reports：7-Eleven 虚拟货架与陈列合规研究（2025）"/>
- <Link url="https://www.neurolabs.ai/blog-post/using-neurolabs-retail-specific-synthetic-dataset-in-production" title="Neurolabs：CPGDet-129 合成货架数据集（2026）"/>
- <Link url="https://arxiv.org/abs/1703.06907" title="Tobin 等：Domain Randomization（2017）"/>
- <Link url="https://arxiv.org/abs/2506.03449" title="合成图像与真实数据混合训练研究（2025）"/>
- <Link url="https://arxiv.org/abs/2506.24093" title="真实与合成混合训练策略比较（2025）"/>

**证据限制：** 我没有取得本项目的原始逐样本分数、融合权重、数据划分、真实照片使用许可文件或实验运行日志。因此，0.324 的精确根因、融合差异的显著性及治理实现状态，仍属于待独立验证事项，而非本报告已经确认的事实。

---

## 十、P0-01 终局与 v2 扩量判定（2026-11-12 本地实测补遗）

第二节要求的"原始逐样本分数"已取得；本节为**本项目实测**（330 评估集，AP 随机基线=正例比例 π≈0.091，阈值曾在 330 上选→探索性）：

### 探针融合消融（P0-01，回答 B 问）
| 系统 | AP | F1 | P/R | 判读 |
|---|---|---|---|---|
| CLIP 单独 | 0.659 | 0.635 | 0.491/0.900 | 弱 |
| DINO 单独 | 0.766 | 0.882 | 0.789/1.000 | 单模态最强 |
| 探针 v1 单独 | 0.324 | 0.414 | 0.429/0.400 | 远低于骨干（0.324≠随机 0.5，基线是 0.091） |
| 探针 v2 单独 | 0.276 | 0.429 | 0.333/0.600 | 同上更弱 |
| **CLIP+DINO 无探针** | **0.883** | **0.885** | 0.871/0.900 | **全部系统 F1 最高** |
| 完整融合 v1（0.4c+0.4d+0.2p） | 0.925 | 0.862 | 0.800/0.933 | 排序最好 |
| 完整融合 v2 | 0.902 | 0.857 | 0.923/0.800 | v2 更差 |

**消融结论**：探针在融合内 ΔAP +0.041（排序收益）但 ΔF1 −0.023（工作点损失）；v2 探针 ΔAP +0.019 / ΔF1 −0.028。**探针=混合证据，不再宣称净正贡献**；v1 探针 AP 0.324 的根因=低排序质量+阈值门控放大，非 bug。逐样本分数落盘 `vis_cache/probe_ablation{,_per_sample}.json`。

### v2 数据扩量判定（820 张合成/300 难负例，公平探针同规模对比）
- 探针侧：v2 线性 AuPR 0.319 vs v1 0.324，**Δ=−0.005，无增益**；MLP 0.254 vs 0.263 同向。
- 融合侧：AP 0.902 vs 0.925，F1 0.857 vs 0.862，**均劣于 v1**。
- **判定：v2 扩量路线关闭，v1（820 张基线）继续作为冻结权重**。合成扩量在当前域差距（低多边形示意图）下不产生收益——与第六节"最小可行验证"预期一致。

### 真实数据侧同期进展（候选级，权利门未过）
1178 店/2542 图采集 → 接触表核验 A 真门头 891（75.6%）→ 10% 从严抽检 precision 94.4% [87.5,97.6] → **估计真门店 ~841 家（780–869）**，≥40 家目标超额 ~21 倍；A 类 96.6% 集中浙江（结构局限）。详见 `data/gen_verify/VERIFICATION_REPORT.md`。