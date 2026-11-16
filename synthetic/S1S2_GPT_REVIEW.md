# 一、总体裁决

<row align="center" gap={2}>
  <badge color="success" size="lg">APPROVE</badge>
  **批准进入受控验证与 Demo 阶段**
</row>

但不等于生产级准入。此次升级在方法论上有实质进步，尤其是从 1:N 品牌辨识转向 1:1 品牌验证，以及真实负例带来的泛化提升。

我认为目前最重要的成果不是 95% 这个数字，而是系统开始具备三个能力：

1. 能够识别自身评估中的假完美结果。
2. 能够在未见品牌上进行一定程度的诚实拒答。
3. 能够区分视觉证据不足、文字识别冲突与目录外品牌，而不是简单地扩大品牌目录。

目前尚未证明的是：剩余 5% 是否具有可通过模型优化解决的可分性，以及验证模式的 FAR 是否在真实业务环境中足够稳定。

---

# 二、上轮五项条件落实核对

| 上轮条件 | 状态 | 评审意见 |
|---|---|---|
| ① FOREIGN 同框/碰撞测试，不扩短别名模糊 | <badge color="warning">PARTIAL</badge> | 本轮声称全部执行，但简报没有展示独立同框碰撞测试结果、测试样本和误判统计。 |
| ② 先 S1 后 S2，对照包含未见品牌 | <badge color="success">CLOSED</badge> | 执行顺序正确。瑞幸、库迪完全不参与训练，作为品牌留出验证对象。 |
| ③ 拉丁规则暂留但不得固化 | <badge color="success">CLOSED</badge> | 不仅没有固化，而且已完全移除原拉丁/CJK 硬规则，改由相似度门控。 |
| ④ 双成绩单：E25 可比 + Demo 决策 | <badge color="success">CLOSED</badge> | 两套指标分开呈现，没有把识别准确率与目录外拒答率混为一谈。 |
| ⑤ pending_review 不进训练/展示/生产 | <badge color="warning">PARTIAL</badge> | 明确了真实 pending_review 仅进入评测，但未分别提供展示和生产隔离的验证证据。 |

依据：简报 §1、§5 及双成绩单部分。<FileCite ref_id="turn0file0" line_range_start={4} line_range_end={5} /> <FileCite ref_id="turn0file0" line_range_start={40} line_range_end={49} />

**我的要求：**

①、⑤不是要求重新开发，而是补齐治理证据。不能因为简报写了全部执行，就把没有展示的验证结果视为已经通过。

---

# 三、A-F 逐项评审

## A. S1 校准陷阱修正与品牌留出协议

<badge color="success">CLOSED</badge>

**裁决：方法学方向正确，而且这次暴露问题的过程具有很高价值。**

三个关键判断：

- 发现 gallery 自身参与校准导致相似度饱和至 1.0，这是典型的数据泄漏问题。
- 改为非 gallery 样本，并按门店级 60/40 拆分，避免简单图片随机切分造成同店泄漏。
- 瑞幸、库迪完全留出，避免把已见品牌识别能力伪装成未知品牌拒识能力。

修正后 AUROC 0.981、已知 FPR 测试观测 0/165、未知品牌 Recall 64.4%，至少说明系统没有继续依赖原先的虚假完美成绩。<FileCite ref_id="turn0file0" line_range_start={6} line_range_end={14} />

**特别评价：**

我认为评估体系已经初步具备自检能力。

但必须区分：

- 发现评估错误：已经证明。
- 自动发现所有潜在评估错误：尚未证明。

后续应将 gallery 自匹配检查、门店级隔离、品牌级留出作为固定评估门禁。

## B. S2 负例探针的未知品牌泛化

<badge color="success">CLOSED</badge>

**裁决：可以认定为具备初步的一般性未知品牌边界能力，但不能认定为通用开放世界识别。**

关键证据：

<box border radius="lg" padding={3} gap={3}>
  <title size="2xl" color="success" tabularNums>88% → 93%</title>
  未见品牌留出泛化提升
  <divider color="subtle"/>
  <grid columns={2} gap={3}>
    <grid-item>
      <title size="xl" tabularNums>95%</title>
      <caption>目录外诚实率</caption>
    </grid-item>
    <grid-item>
      <title size="xl" tabularNums>5%</title>
      <caption>错进目录率</caption>
    </grid-item>
  </grid>
  <text color="secondary" size="xs">真实负例扩充后；瑞幸、库迪未参与训练。</text>
</box>

这里最有价值的不是 93%，而是实验对照：

- 合成负例：0pp 泛化收益。
- 真实负例：+5pp 泛化收益。

说明系统学习到的不是抽象的负例数量，而是真实照片中与目标目录竞争的视觉分布。<FileCite ref_id="turn0file0" line_range_start={28} line_range_end={38} />

但有一个重要边界：

瑞幸和库迪虽然是未见品牌，却不代表未见品类、未见商业场景、未见地域、未见拍摄条件。

因此我的定义是：

**具备初步的未见品牌拒识泛化能力，不具备已经证明的任意未知品牌泛化能力。**

这个边界不影响本轮通过。

## C. 1:1 验证模式与双分布阈值

<badge color="warning">PARTIAL</badge>

**裁决：架构转变正确，阈值方法合理，但 FAR 0% 还不能作为生产级安全保证。**

我非常认可这次范式转变。

原来的问题：

> 这家店究竟属于哪个品牌？

新的问题：

> 这张照片是否支持用户宣称的品牌？

这消除了大量不必要的 1:N 品牌竞争问题。

双分布阈值也比只使用 genuine 分布可靠得多。之前只用 genuine 分布时，出现 FAR 19%，甚至左邻右舍照片冒充美宜佳通过 32.5%，说明单边校准存在严重风险。

双分布校准后：

| 指标 | 结果 | 判断 |
|---|---:|---|
| ZUO 真宣称 TAR | 97.5% | 良好 |
| 冒充 FAR | 0%（0/100） | 初步通过 |
| 目录 TAR | 78.8% | 偏保守 |

<FileCite ref_id="turn0file0" line_range_start={21} line_range_end={26} />

我有两个保留意见：

**第一，0/100 不等于真实 FAR 为 0。**

100 个冒充样本没有出现误接受，只能说明当前测试中没有观察到误接受。统计置信区间仍然较宽。

建议扩大独立冒充测试，至少达到数百个样本，并按品牌、门店和拍摄条件分层。

**第二，目录 TAR 78.8% 是一个需要业务确认的代价。**

如果这是品牌真实性验证，宁可拒绝一部分真实品牌，也不能轻易接受错误品牌，78.8% 可以接受作为保守工作点。

但必须提供人工复核或重新拍摄通道，不能把 REJECTED 直接解释成品牌造假。

## D. 孪生合成负例实验及停止投入

<badge color="success">CLOSED</badge>

**裁决：负结果解释成立，停止当前合成负例扩产是正确决策。**

这次实验很有价值，因为它否定了一个看似合理的技术假设。

| 对照 | 留出泛化收益 |
|---|---:|
| 参数化合成负例 | 0pp |
| 真实照片负例 | +5pp |

合成图与真实照片的最近余弦相似度只有 0.31–0.35，而真实门店与同品牌 gallery 的相似度接近 1.0。

因此，模型确实可能学到渲染风格，而不是品类边界。<FileCite ref_id="turn0file0" line_range_start={28} line_range_end={38} />

**我的建议：不再投入高保真 Blender 渲染来扩充这个负例池。**

尤其不要因为提高渲染质量，就预设它能够解决真实照片的视觉混淆。

未来如果要重新研究合成数据，必须先证明合成样本与真实困难负例在目标嵌入空间具有足够的分布接近性。

目前没有这个证据。

## E. 剩余 5% 的归属与处理策略

<badge color="danger">OPEN</badge>

这是本轮最重要的未决问题。

我不建议简单地把剩余 5% 视为模型还有 5% 的提升空间。

简报给出的 14 张错误样本：

- 9 张：纯视觉路径，嵌入与目录高度重叠。
- 5 张：文字路径，OCR 错读撞别名或同框招牌。

<FileCite ref_id="turn0file0" line_range_start={52} line_range_end={53} />

这两类问题本质不同，应该分开处理。

## 四、对剩余 5% 的边际收益建议

我的明确建议是：

<badge color="success">建议采取选择性 S3，而不是继续扩产 S2</badge>

### 1. 9 张纯视觉困难负例：不建议继续无差别扩产

这 9 张的核心问题是：

真实照片本身就与目录内品牌具有高度视觉相似性。

继续增加普通茶饮、咖啡、快餐照片，很可能只是在增加容易样本，而不是解决真正困难的边界。

建议：

- 先对 9 张进行独立专家复核。
- 检查是否存在可辨识的品牌特征、门店环境、招牌文字。
- 对确实具有额外可见证据的样本，尝试 S3 本地 VLM。
- 对单图证据本身无法区分的样本，允许 UNKNOWN，不强迫模型给出品牌归属。

### 2. 5 张文字路径冲突：优先做低成本修复

这 5 张不应该直接交给 VLM 兜底。

先检查：

- OCR 是否存在字符误读。
- 短别名是否产生碰撞。
- 同框招牌是否导致品牌归属错误。
- OCR 文字是否与视觉主体属于同一家门店。

这些问题可能通过 OCR 候选重排、文字与视觉主体关联、别名冲突门控解决。

而且，这部分比纯视觉重叠更可能通过规则和证据关联获得低成本收益。

### 3. S3 应该是选择性验证，不是全面替换

我建议采用下面的分层架构：

<box border radius="lg" padding={3} gap={2}>
  <box background="rgba(22,163,74,0.09)" radius="md" padding={3} align="center">
    **S1 + S2 主识别系统**

    <text color="secondary" size="xs">正常样本直接决策</text>
  </box>
  <box align="center">
    <icon name="arrow-down" color="secondary" size="lg"/>
  </box>
  <box background="rgba(234,179,8,0.10)" radius="md" padding={3} align="center">
    **困难样本触发器**

    <text color="secondary" size="xs">高相似度 / OCR 冲突 / 同框歧义</text>
  </box>
  <box align="center">
    <icon name="arrow-down" color="secondary" size="lg"/>
  </box>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={2} align="center">
        **S3 本地 VLM**

        <caption>补充视觉与文字证据</caption>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={2} align="center">
        **UNKNOWN / 人工复核**

        <caption>证据不足时拒绝归属</caption>
      </box>
    </grid-item>
  </grid>
  <box align="center">
    <icon name="arrow-down" color="secondary" size="lg"/>
  </box>
  <box background="rgba(22,163,74,0.09)" radius="md" padding={3} align="center">
    **最终决策**

    <text color="secondary" size="xs">VERIFIED / REJECTED / UNKNOWN</text>
  </box>
</box>

注意：VLM 不应该拥有推翻主系统证据门禁的权力。它提供补充证据，而不是成为另一个不受约束的裁决者。

### 4. 边际收益的投入优先级

| 方向 | 边际收益预期 | 建议 |
|---|---|---|
| 继续扩充普通真实负例 | 低 | 暂停 |
| 继续扩充合成负例 | 极低 | 停止 |
| 9 张纯视觉困难样本专项验证 | 中，待证实 | 优先 |
| 5 张 OCR/别名冲突修复 | 较高的低成本机会 | 优先 |
| S3 全量接管 | 未知，成本较高 | 不做 |
| 选择性 S3 兜底 | 值得验证 | 推荐 |
| 接受不可判定样本 UNKNOWN | 高治理价值 | 必须保留 |

**我建议设立 S3 的明确准入门槛：**

以冻结的 14 张困难样本作为初始测试集，另外增加独立真实困难负例测试集。

建议至少满足：

- 困难负例错误接受率相对下降 50%。
- 已知品牌准确率下降不超过 0.5 个百分点。
- 不允许 VLM 把证据不足的样本强行判定为某个品牌。
- 推理成本和延迟必须低于业务允许的预算。

这些是我建议新增的验收门槛，并非简报已经取得的实验结果。

**最终判断：不要为了把 95% 提升到 97% 或 98%，而无限扩张负例池。应该把资源集中到剩余 14 张的错误机制上。**

---

## F. 治理：合成负例与 pending_review 隔离

<badge color="warning">PARTIAL</badge>

**裁决：治理方向正确，但证据链还没有完全闭合。**

我认可两个原则：

1. 合成负例可以进入训练池，但必须明确其合成来源，不得冒充真实门店证据。
2. 真实 pending_review 照片只用于评测，不得进入训练。

<FileCite ref_id="turn0file0" line_range_start={28} line_range_end={31} />

但我希望进一步明确三个隔离边界：

| 数据类型 | 训练 | 评测 | Demo 展示 | 生产 |
|---|---|---|---|---|
| 已审核真实照片 | 允许 | 允许 | 按权限 | 按权限 |
| 合成负例 | 允许 | 单独标记 | 不得冒充实拍 | 不得作为真实证据 |
| pending_review | 禁止 | 允许 | 禁止 | 禁止 |

尤其要避免一个容易被忽视的问题：

pending_review 照片参与评测后，评测人员的人工裁决结果是否会反向进入训练数据？

这需要明确版本隔离、人工复核和训练数据准入流程。

---

# 五、最终 A-F 裁决汇总

| 项目 | 裁决 | 核心结论 |
|---|---|---|
| A. S1 校准与留出协议 | <badge color="success">CLOSED</badge> | 假完美问题已纠正 |
| B. S2 未知品牌泛化 | <badge color="success">CLOSED</badge> | 初步开放世界拒识能力成立 |
| C. 1:1 验证模式 | <badge color="warning">PARTIAL</badge> | 架构正确，FAR 仍需扩大验证 |
| D. 合成负例零收益 | <badge color="success">CLOSED</badge> | 停止当前方向投入 |
| E. 剩余 5% | <badge color="danger">OPEN</badge> | 需要困难样本专项验证 |
| F. 数据治理 | <badge color="warning">PARTIAL</badge> | 原则明确，隔离证据不完整 |

# 六、最终决策与下一阶段

<box border radius="lg" padding={3} gap={2}>
  <row align="center" justify="between">
    **总体裁决**
    <badge color="success" size="lg">APPROVE</badge>
  </row>
  **批准 S1/S2/验证模式升级进入受控 Demo。**

  不批准将当前指标直接作为生产级品牌真实性保证。
  <divider color="subtle"/>
  **下一阶段只做四件事：**

  1. 补齐 FOREIGN 同框碰撞测试与 pending_review 隔离证据。
  2. 扩大验证模式的独立冒充测试。
  3. 专项处理剩余 14 张困难样本，验证选择性 S3。
  4. 固化双成绩单与评估数据隔离门禁。
</box>

**最后，我想强调一个战略判断：**

这个项目已经出现了一次重要的方向性转变。

早期系统的目标是尽可能准确地识别品牌；现在系统开始理解什么时候不应该识别、什么时候应该验证、什么时候必须拒绝给出结论。

这比单纯追求准确率更重要。

下一阶段不要再以「把准确率从 95% 做到 98%」作为主要目标，而应转为：

**把系统无法判断的边界弄清楚，把能够判断的证据用好，把不应该判断的情况可靠地拒绝掉。**

这才是一个可以进入真实商业场景的门头识别系统应该具备的能力。