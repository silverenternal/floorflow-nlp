# NE-002 神经元核心与既有实现边界

日期：2026-11-06

## 盘点结论

| 现有实现 | 当前职责 | 与基础神经元框架的关系 |
| --- | --- | --- |
| `NeuronModule` | 局部投影、fast/slow GRU 状态、门控及残差输出；显式接收/返回 `NeuronState` | 有价值的有状态候选，不是通用协议。固定 PyTorch Tensor 与 `[batch, sequence, channels]` 形状，诊断和特定残差结构属于实现细节。 |
| `CompetenceCell` | 面向具名能力缺口的低秩残差变换，可选 GRU 状态 | 任务/能力适配器；无显式状态返回，不应作为所有神经元必须实现的行为。 |
| `UnifiedCompetenceBank` / `UnifiedPlanner` | 同时路由能力、深度、并行 lane、生成模式与计算预算，并汇总生成专家 | 高层模型编排器，职责过多，不适合作为最小神经元或首个闭环。 |
| `ContextConditionedExpertFactory` | 从上下文和参数基底合成低秩专家，并选择 rank/width | 专家生成机制，可作为未来能力实现候选；它负责权重生成而非单元的基础行为。 |
| `CompactNLPDecoder` 与训练脚本 | 可选集成神经元、能力路由、专家和任务头；通用 NLP 训练入口耦合 tokenizer、数据和模型配置 | 可供后续接入或对照。首个核心闭环不依赖大型 NLP 数据管线、旧运行配置或 tokenizer 文件。 |

## 代码证据

- `NeuronModule.forward(parent_hidden, state=None)` 在一个序列调用中推进 fast/slow 状态；`CompactNLPDecoder.forward` 调用它时不传入或保留状态，因此当前集成路径不会跨模型调用继续状态。
- 神经元模块和能力银行均是可选配置项；模型构造默认关闭神经元模块。部分历史架构配置会启用它，但这些配置不构成已验证的独立神经元训练闭环。
- 现有 `NeuronModule` 的 CPU 前反向和参数梯度测试通过，适合作为有状态候选和第二阶段对照。
- 有状态、无状态、能力路由和生成专家目前混在不同的 NLP 模型实现内；没有一个独立、最小的行为协议供不同实现共享。

## 首个垂直切片决策

1. 在 `src/newera/neuron/` 建立与 `nlp_flow` 解耦的基础行为协议；协议采用结构化类型，不要求继承某个神经元基类。
2. 协议只暴露一次行为调用、值和可选显式状态；输入/输出类型保持泛型，不将序列长度、门控、局部状态结构、任务路由或学习算法设为公共要求。调用者通过不再传入旧状态实现重置；无状态实现返回空状态。
3. 下一步先做小型可训练 MLP 控制实现，再以现有 `NeuronModule` 作为单独候选对照。二者不强行兼容；若已有实现不满足新协议，后续通过显式实验适配器接入，不加旧 API fallback。
4. 首个端到端任务用确定性的合成序列任务和 CPU 训练；单独报告参数量、训练计算和推理时延。先验证接口和训练闭环，不以现有复杂 NLP 训练器或 `UnifiedCompetenceBank` 的规模代表神经元能力。
5. 能力契约与具体任务适配器留到 NE-008；动态生长、多模态/语言能力和高层编排不纳入最小公共接口。

## NE-003 / NE-004 实施结果

- `newera.neuron.Neuron` 已用 Python `Protocol` 定义 `step(inputs, *, state=None) -> NeuronResult`；输入、输出和状态类型均为泛型，不强制继承、Tensor 形状或 `nn.Module`。
- 状态由调用方持有；传入 `None` 表示开始新的状态生命周期，无状态实现返回 `None`。协议没有为所有实现增加空泛的 `reset`、学习或诊断方法。
- 无状态和有状态测试实现共用同一个结构化调用路径；MLP 控制实现进一步覆盖真实参数训练、反向传播及显式拒绝无效形状/非空状态。
- 两层 MLP 在 64 个固定合成样本上训练 80 步，MSE 从 2.05217 降到 0.006378；65 个参数，线性层前向约 48 MAC/样本。详见 [NE-004 基线](neuron_baseline.md)。
- 当前协议由 Python runtime structural check 和 CPU 梯度测试验证；没有声称已有静态类型检查或跨设备一致性验证。

## NE-005 有状态子模型候选

新增 `StatefulMLPNeuron`：每次对一个 `[batch, input_size]` 输入先编码，再用 GRU 更新调用方持有的状态，最后输出任务值。其状态是显式 Tensor；调用者可以在序列步骤间传递状态，也可以传 `None` 重置。它与无状态 MLP 控制实现通过同一个 `Neuron.step` 调用方，不共用内部基类。

CPU 测试覆盖多步组合、状态重置、形状/类型错误和全参数反向梯度。测试规模 `input=3, state=8, output=2` 下，候选有 482 个参数、约 424 线性 MAC/步骤；同输入输出与 8 维隐藏层的 MLP 控制有 50 个参数、约 40 MAC/样本。MAC 不含偏置、门控激活、数据搬运或运行时开销；候选显著更贵，后续必须做预算对齐比较。当前尚未做时延、内存或 MPS 上的候选评测，也没有泛化结论。

## NE-006 有序组合与状态隔离

新增 `SequentialNeuron`，用同一批输入按顺序调用结构化 neuron 实现，并返回按组件位置对应的状态 tuple。组合器要求子组件是 `nn.Module`，以便 PyTorch 收集参数；抽象协议本身不要求继承 `nn.Module`。无状态组件对应的状态槽保持 `None`，各有状态组件拿到独立状态；输入和先前状态都不被组合器就地改写。当前只实现有序串联，不引入并行路由。

CPU 测试覆盖两层混合组合、沿时间显式传递状态、传 `None` 整体重置、状态长度校验，以及批量运行与逐样本运行的等价性。测试规模下组合共有 541 参数、约 472 线性 MAC/样本/步；第二层有状态 Tensor 在 batch=4、state=8、float32 时占 128 bytes。以上不含框架开销；组合成本会随组件累加，后续仍需正式测量。

## NE-008 能力与任务适配边界

新增 `newera.capabilities.Capability[I, O]` 和独立的 `TaskAdapter[TaskInput, CapabilityInput, CapabilityOutput, TaskOutput]`。能力只约定 `execute`；适配器只负责 `encode`/`decode`。它们不依赖某个神经元类，不强迫所有能力共享序列、状态、设备或任务字段，也不把任务编码混入神经元实现。具体错误由实现按已声明的输入约束明确抛出；stateful 执行仍通过前述可选 neuron state 或能力自己的实验实现。

测试中的 double、square 两种能力使用同一任务适配器和调用路径；另有真实 MLP regression capability 冒烟。接口只表达替换边界，没有引入统一路由器、能力注册表或稳定版本承诺。

## NE-010 第二种能力

新增 `SequenceSumCapability`，用 `StatefulMLPNeuron` 逐步累计定长数值序列；它与 NE-009 的无状态回归能力使用同一 `Capability.execute` 边界，但有独立 task adapter 和模型实现。两个任务均能通过相同结构化替换协议运行，任务形状和状态语义由各自适配器/能力处理。

这只证明了两种合成任务可共享替换边界，不证明二者可互换、跨任务迁移或真实数据泛化。结果与成本见[NE-011 对照记录](../topics/neuron_budget_comparison/neuron_budget_comparison.md)。

## 已知限制

这只是对仓库中当前代码的职责划分，不意味着 `NeuronModule` 的内部结构已被实验证明有效，也不声称神经元全能性。下一步应让不同实现通过同一小型协议测试，再据证据调整协议。


## 可变度稀疏神经元图（NE-015）

输入嵌入继续作为输入适配器。显式设置 `enable_neuron_graph` 后，`DynamicNeuronGraphLayer` 运行在嵌入/任务元数据之后、Transformer 解码层之前。该开关默认关闭，以保留既有 2.3M 参数预算。节点当前对应 token 位置；这不是跨样本持久的神经元身份系统。

连接以稀疏边表保存。入边消息按有效边权求和，再除以 `sqrt(max(1, 总入边权重))`；有效加权度数同时进入 `log1p` 度数信号。零权边不影响聚合或度数。共享低秩候选更新和门控更新产生节点状态；残差门以 0.01 初始化，让主任务梯度从第一步即可进入图参数。连接的实际入度/出度没有 per-node 上限。可选 `neuron_graph_max_edges` 是整个 batch 的硬边预算，任何前向都会检查。

`NeuronGraphTopology.causal_candidates` 自动产生候选连接。默认每图搜索预算为 256 条；这是每次调用的新候选搜索预算，不是已建立边数或神经元度数限制。候选按有效 segment 分组，以三角索引从有效连接空间直接抽样或按 offset 切片，不物化全部连接；offset 到达候选空间末尾后返回空集，不回绕重复。设置预算为 `None` 或预算覆盖全部候选时才会完整枚举，成本随有效候选数平方增长。策略评分集合始终并入当前拓扑的有效边，因此候选预算不会让已有连接逃过保留、改型或删除评估。图消息传递、标准解码层和动态深度解码路径都按相同 segment ID 隔离 packed 样本。

训练数据打包器为每个来源样本生成 `segment_ids`，padding 位置为 `-1`。`batch_loss` 只对输入与目标均有效且 segment ID 相同的相邻 token 计算语言模型损失，避免跨样本拼接边界形成伪 next-token 目标。训练时策略 logits 经过 softmax 形成每类边的可微期望权重，语言模型损失可直接训练连线策略。模型输出 `neuron_candidate_edges` 与 `neuron_link_logits`；有候选动作标签时，传入 `neuron_link_targets` 并在 `batch_loss` 设置 `neuron_link_weight`，可加入监督交叉熵。硬拓扑更新通过调用者提供 mutations，或显式设置 `update_neuron_topology=True` 在调用边界执行；这两种硬更新都必须配置有限的 `neuron_graph_max_edges` 整图预算，避免无界拓扑增长。策略动作是保留、按关系类型添加/改型、删除。

拓扑复用必须提供稳定 `neuron_node_ids`：拓扑保存创建时的 ID，模型复用时逐位置精确比对；ID 缺失、重复或顺序/样本不匹配都会拒绝。padding ID 固定为 `-1`。`PackedTokenDataset` 可用调用者分配的 `node_identity_namespace` 生成可重复 ID；namespace 范围为 `[0, 32768)`，并须在可能共享拓扑的数据集之间唯一。ID 编码保留 24 位来源行序号和 24 位行内 token 偏移。`append_causal_nodes` 对已绑定拓扑要求调用者为新节点提供新的 ID，再重映射 batch 内边索引。这样拓扑只会在节点身份确实一致时复用；节点身份仍由调用方定义，仓库尚无跨任务的持久神经元注册表。

硬拓扑更新只允许在模型 eval 模式执行，策略按离散拓扑结果传播；训练模式保留可微候选边权，不允许同一前向混合硬更新与软边权。调用方应在优化步骤之间切换到 eval 模式更新拓扑，再回到 train 模式继续优化；`CompactNLPDecoder.forward` 强制检查此边界。

普通训练使用设备上的并行 `index_add_` 聚合。启用 PyTorch 确定性算法时，图层和节点 embedding 查表改用按索引分组求和路径；Apple MPS 上图层与完整紧凑解码器的确定性前后向均有测试。该确定性路径更慢，后续应单独测吞吐。实现与契约见 `src/newera/neuron/graph/`、`src/newera/nlp_flow/model/components/architecture/foundation/model.py`、`src/newera/nlp_flow/components/research/training/objectives/training_losses.py` 和 `project/tests/models_training/contract_scopes_stage/neuron/test_neuron_graph.py`。

探索性二阶模加任务未显示候选图优于无图基线；CPU 候选搜索在 32–512 长度上的预算扫描和端到端前向时延已记录，但没有自然语言能力结论、MPS 吞吐或峰值内存证据。图搜索预算、全图边预算和学习到的实际连接度数是三个独立量，应在正式实验中分别记录。
