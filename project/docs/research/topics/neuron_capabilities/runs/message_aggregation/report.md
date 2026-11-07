# 消息集合聚合器验证

## 输入契约与分组反例

`ModelNeuron.step` 的消息输入是 `[batch, contacts, message_size]`。每行是一条独立接触；行顺序不代表语义，重复行保留，集合无固定 top-k 上限。strength 与 edge type 按行对齐。分组/接触身份必须由 task adapter 编码为可观察 payload 特征，不能把样本内 group ID 等同于持久 neuron ID。

NE-024 的最小反例有四条消息，role 为 `[A,B,A,B]`、值为 `[1,2,3,4]`。group 配对 `[0,0,1,1]` 的 oracle 为 14；配对 `[0,1,1,0]` 的 oracle 为 10。两输入的全局线性投影后求和完全相同，因此全局加权求和不能区分；PyG Set Transformer 在 CPU/MPS 初始化时均能区分，整组行置换后输出不变。当前 topology 无需扩展平行边：任务表达的是一个接收神经元的多条输入消息，contact group 是 payload 语义；没有反例要求相同 source/target 的多个 topology edge。

完整契约见 `project/docs/research/topics/neuron_capabilities/specification/message_set_contract.md`。

## 实现与上游适配

- 在 `src/newera/neuron/cells/aggregation/set_transformer.py` 适配 PyG 2.8.1 `SetTransformerAggregation`，参数作为 `ModelNeuron.message_aggregator` 子模块注册，所有图位置共用同一份权重。
- adapter 支持单个变长集合、padded batch 与 mask；空集合返回零向量，padding 被排除，重复消息不去重。Padded batch 按有序 `index` 一次调用 PyG，跳过空 segment 并散射回原 batch 行。
- `ModelNeuron` 为实现 `forward_batched` 的聚合器走批量路径；其他聚合器仍按各样本分别处理。默认的全局加权求和路径不变。
- 额外验证了 context-gated sum residual：任务上下文可选择性开启 Set Transformer 交互，门控零初始化使其开始时等于原求和路径；所有注意力权重与门控参数都在同一个 ModelNeuron 下。
- CPU 和 Apple MPS 聚焦测试 23 passed；含消息排列、分组反例、mask、空输入、梯度、参数注册和共享模型路径。Ruff 通过。PyG 当前会产生 `torch.jit.script` 弃用 FutureWarning。

上游来源、MIT 许可、固定源码提交、安装方式及直接 adapter 的早期微型基准见 `project/docs/research/topics/neuron_capabilities/reuse/component_selection.md`。

## 训练对照

所有候选在同一个 ModelNeuron 里联合训练 set sum 与 grouped binding 两族；输入上下文给出任务族与可观察集合大小。组身份每样本随机重标，payload 使用 `sin/cos(π·group_id/2^k), k=0..3` 编码，role 使用 edge type。比较使用 MPS、六个 seed、每族 2,048 个训练样本、8 epoch、batch 32、每模型 1,024 更新，AdamW 1e-3。各候选共享同一训练样本与更新数。验证数据用于记录训练尺寸和外推尺寸结果；封存测试使用每族 512 样本，仅在候选与训练方案冻结后评估。

| 方案 | sealed set-sum MAE | sealed grouped-binding MAE | 参数 | MPS 前反向中位数（64 条消息） |
|---|---:|---:|---:|---:|
| 全局加权求和 | 0.625 | 1.485 | 7,081 | 0.415 ms |
| 全局求和 + 上下文门控 Set Transformer 残差 | 0.658 | **1.200** | 24,489 | 4.118 ms |

上下文门控候选在 grouped binding 上的 paired seed-relative MSE 改善为 **24.7%**，95% t 区间约 **3.9%–45.4%**；MAE 改善约 19.2%。在 set sum 上 MSE 平均增加约 10.9%，但区间跨零；MAE 约差 5.2%。集合行置换最大平均误差分别低于 `1.2e-6`。独立 Set Transformer 候选（无 sum residual）把 grouped-binding MSE 降低约 24.6%，但 set-sum MSE 增加约 78.5%，因此不选为通用默认。原始标量组 ID 版本及普通残差版本结果保留在各自子目录，不能与随机重标 Fourier 版本混为一组。

外推验证未达标：grouped binding 从训练最多 8 组外推到 16 组时，求和 MAE 3.07、上下文残差 MAE 3.33；set sum 从训练最多 64 项外推到 256 项时，两者 MAE 分别为 7.62 和 7.88。冻结目标为两族 MAE ≤0.05；两种实现都远未通过，不能声称模型已学会集合归约或分组绑定。

## 决策与未闭合项

全局求和仍是默认聚合，因其参数与时延更低且没有注意力带来的 set-sum 精度回退。Context-gated PyG 聚合保留为显式研究候选：它在训练分布内提供有统计证据但幅度有限的分组交互收益；约 3.46 倍参数、约 9.9 倍整层 MPS 前反向代价和失败的 16 组外推，尚不足以作为生产候选。

NE-024.1 消息契约和反例已完成；NE-024.2 上游组件适配、参数共享、空输入/mask 语义已完成。NE-024.3 已完成对照测量和候选选择，但预先冻结的能力阈值与规模外推均失败，因此聚合能力验收不通过。这个缺口记录为后续需要解决的明确问题：在不破坏全局归约的前提下，提升任意样本内 group identity 的配对学习和 16 组外推，同时显著收回注意力成本。在此期间，NE-026/027 依赖的“已验证非线性聚合能力”不可当作已具备。

完整六 seed 逐项数值、配置/源码指纹、成本和早期负结果见：

- `sum_vs_attention/confirmation_results.json`
- `sum_residual/raw_group_ids/results.json`
- `sum_residual/fourier_group_ids/confirmation_results.json`
- `contextual_sum_residual/fourier_group_ids/confirmation_results.json`
- `training_budget_diagnostic/results.json`

研究配置：`project/configs/models_training/learning/neuron_comparisons/functional/message_aggregation.json` 与 `sum_residual_aggregation.json`。

## PyG DeepSets 追加候选

NE-024.4 在冻结的相同 6 seed、每族 2,048 训练样本、8 epoch、1,024 updates 配置下加入 PyG `DeepSetsAggregation`：每条消息经 MLP 编码，按集合求和后再经全局 MLP；全局求和和 Set Transformer 保持同批比较。实现为 `DeepSetsMessageAggregator`，仍注册在单一 `ModelNeuron`，没有独立聚合模型。完整配置哈希与逐 seed 结果见 `training_budget_diagnostic/deep_sets_results.json`。

| 方案 | sealed set-sum MAE | sealed grouped-binding MAE | 16组 binding 外推 MAE | 参数 | MPS 前反向中位数（64条） |
|---|---:|---:|---:|---:|---:|
| 全局加权求和 | 0.6139 | 1.4985 | 3.0994 | 7,081 | 0.413 ms |
| PyG DeepSets | 0.7538 | 1.3634 | 3.0167 | 15,465 | 1.357 ms |
| PyG Set Transformer | 1.2574 | 0.7091 | 2.8476 | 24,393 | 3.982 ms |

相对求和，DeepSets 的 grouped-binding MSE 平均改善 10.1%（paired 95% t 区间 6.7%–13.5%），未达预注册的 20%；set-sum MAE 和 binding MAE 都远高于 0.05，16组外推也远高于 0.05。DeepSets 参数约 2.18 倍、MPS 前反向约 3.3 倍于求和；比 Set Transformer 快约 2.9 倍，但任务能力仍不合格。Permutation/空集/mask/重复接触和梯度语义测试在 CPU/MPS 通过，聚合实现能保持置换不变。

**决策：**DeepSets 是比注意力更便宜的可用实验组件，但不是通过能力验收的方案，不替换默认求和，也不解锁 NE-026/027。研究结果没有证明已学会分组绑定或可扩展归约；应在下一轮先诊断 8 epoch 是否造成欠拟合（只使用新训练/验证数据），若优化预算也不能显著降低误差，再转向更强的组身份交互表示。

## 更长训练预算与上下文门控 DeepSets

NE-024.5 的 validation-only 单 seed 预算曲线显示，8→32 epoch 时误差仍下降（完整记录：`training_budget_diagnostic/learning_curve/deep_sets/underfit_probe_results.json`）。据此冻结 32 epoch 后，用全新 seed base 64000 做六 seed 确认；逐 seed sealed test 在配置冻结后首次读取，结果见 `training_budget_diagnostic/extended_budget/deep_sets_extended_results.json`。DeepSets 的 grouped-binding sealed MAE 为 1.2153、16 组外推 MAE 为 3.0218，相对求和的 grouped-binding MSE 改善 19.9%（95% CI 10.4%–29.4%），未达到预注册的 20% 平均门槛。DeepSets 的 set-sum sealed MAE 0.4121，高于求和的 0.2183。参数为 15,465 对 7,081，64 条 MPS 前反向约 1.34 ms 对 0.41 ms。训练预算提升帮助了 group binding，但未完成验收。

之后实现上下文门控 DeepSets 残差：零初始化时与求和完全一致，由接收神经元 context 选择性开启交互。全新 seed base 74000 六 seed 结果在 `training_budget_diagnostic/extended_budget/contextual_deep_sets_confirmation_results.json`：grouped-binding MAE 1.2657，paired MSE 改善 15.1%（95% CI 6.1%–24.2%）；set-sum MAE 0.3507，高于求和 0.2451；16 组外推 MAE 3.204。参数 15,561，64 条 MPS 前反向约 1.42 ms。门控候选在零初始化时满足严格求和语义，但任务指标仍未通过，不能替换默认求和或声称完成分组绑定能力。

下一候选转向 PyG `LCMAggregation`：它以共享 GRUCell 对称合并构造可交换组合，并按树深归约，官方 PyG 文档将其描述为 O(log N) 深度。该实现当前锁定版本为 PyG 2.8.1；先检查可变长度批处理、空集合、重复接触、排列误差及 CPU/MPS 梯度，再决定是否进入训练对照。来源：[PyG 聚合 API](https://pytorch-geometric.readthedocs.io/en/stable/modules/nn.html)、[LCM 论文](https://arxiv.org/abs/2212.08541)。

## PyG LCM 兼容性筛查

PyG 2.8.1 `LCMAggregation` 在 CPU/MPS 小型适配试验中未通过消息集合置换不变性：同一组 3 条消息做置换后，随机初始化下最大输出差异分别约为 0.314（CPU）与 0.251（MPS）。PyG 实现对每一级二元 GRU 合并的左右方向求均值，保证单次二元合并对称；不同输入次序会改变树的配对和归约结构，因此整体集合输出仍会变化。该偏差违背 `ModelNeuron.step` 的无序消息集合契约。未为弥补上游行为另写排序或全排列逻辑；LCM 不接入项目，也不进入任务训练比较。

下一项考察 PyG `EquilibriumAggregation`。上游定义以对所有输入元素求和的能量项对潜变量做迭代优化，可直接作用于 index 分组，不需要对不等长集合补零；需实测 MPS 双重梯度、计算成本和 masked/empty 适配，再决定是否训练。官方资料：[PyG API](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.aggr.EquilibriumAggregation.html)、[论文](https://arxiv.org/abs/2202.12795)。

EquilibriumAggregation 的 MPS 筛查同样没有支持进入正式确认。首轮 8 epoch/512样本验证结果在 `training_budget_diagnostic/learning_curve/equilibrium/equilibrium_screen_results.json`；扩展到32 epoch、每族2048训练样本的新 seed 结果在 `equilibrium_extended_screen_results.json`。后者 set-sum 验证 MAE 为 0.4151，对比求和 0.1564；grouped-binding 验证 MAE 为 1.3650，对比求和 1.4397。聚合器参数 15,662 对 7,081；单 seed 32 epoch 耗时约 165.7s 对 142.6s。分组绑定提升不足以抵消 set-sum 回退与成本，没有开展封存测试。筛查后移除适配代码，不进入 NE-024 正式候选。

## PyG EquilibriumAggregation 筛查结论

PyG 官方 `EquilibriumAggregation` adapter 在 CPU/MPS 通过语义、置换和梯度检查，但训练/验证筛查没有胜出。使用正式样本量与32 epoch的新 seed 时，set-sum 验证 MAE 为 0.4151（求和控制 0.1564），grouped-binding 为 1.3650（求和 1.4397）；参数 15,662（求和 7,081），单 seed 训练 165.7s（求和 142.6s）。没有查看 sealed test，候选实现已移除。完整数据见 `training_budget_diagnostic/learning_curve/equilibrium/equilibrium_extended_screen_results.json`。

下一项是 PyG `SetTransformerAggregation` 的多 seed pooling：移除全量消息 encoder self-attention，采用四个 learned seed query 对全部消息 cross-attention，再在 seed slots 之间解码。这将保持输入可见性和置换不变，并把消息交互改为 O(N·4)；项目仅复用 PyG 实现并适配输出形状。上游构造参数见[官方 API](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.aggr.SetTransformerAggregation.html)。

## PyG 多 seed pooling 筛查

PyG SetTransformerAggregation 的零 encoder block、四个 learned pooling seed、一个 seed 间 decoder 适配在 CPU/MPS contract tests 通过，复杂度避免了全量消息 self-attention。新 seed 87139、32 epoch、每族2048训练样本的 validation-only 筛查结果见 `training_budget_diagnostic/learning_curve/multi_seed_pool/screen_results.json`：候选 grouped-binding MAE 1.4965、set-sum MAE 0.4784，均未形成优势；DeepSets 同批 binding MAE 为1.1413、求和 set-sum MAE为0.3237。参数23,209、训练149.5s，也高于求和控制7081参数/142.9s。未读取 sealed test，筛查后移除 adapter。

后续候选改用 Set Transformer 原作者官方 PyTorch 实现的 ISAB，保留每个消息与固定诱导点交互的机制来控制复杂度。官方仓库：[juho-lee/set_transformer](https://github.com/juho-lee/set_transformer)，论文描述其诱导点注意力将全量 self-attention 的二次计算降为对集合大小线性。

## 作者官方 ISAB 筛查

从作者官方 MIT repo（commit `73432c640ac78140496d6738416c54d32c686d65`）适配 ISAB/PMA，并在 MAB 中加入 key mask。许可与修改清单见 `reuse/set_transformer_source.md`。CPU/MPS 置换、mask、空集、变长和反向 contract tests 通过。32 epoch 新 seed 94121 的 validation-only 结果（`training_budget_diagnostic/learning_curve/induced_set/`）：16 inducing/2 blocks 的 binding MAE 1.4182、set-sum 0.7850；32 inducing/1 block 为 binding 1.1943、set-sum 0.4925；同批 DeepSets 为 binding 1.1102、set-sum 0.2358。参数分别29,897和21,193；MPS训练时间167.5秒和156.5秒，DeepSets 138.4秒。未读取 sealed test。ISAB 单独替换求和没有验证收益。下一步仅验证上下文门控残差能否保持求和同时引入交互；若不优于 DeepSets，则移除 adapter。

ISAB 上下文门控残差在同一 seed 的32 epoch validation-only筛查也未形成足够优势：grouped-binding MAE 1.0821，略好于 DeepSets 的1.1102；set-sum MAE 0.3373，差于 DeepSets 的0.2358；参数21,289，训练157.6s，对比 DeepSets 15,465/139.3s。未读 sealed test；按预注册决策移除 ISAB adapter。上游 clone、MIT许可和修改记录保留于 `reuse/set_transformer_source.md`。

下一项以 PyG DeepSetsAggregation 保持上游算子，只将 MLP expansion 从2提高至8；使用已查看验证集作一次容量筛查，不调 sealed test，失败就删掉宽变体。

## DeepSets 容量扩展筛查

PyG DeepSets MLP expansion 从2提高至8后，seed94121 的32 epoch validation binding MAE 从1.1102降到0.9267，但set-sum MAE从0.2358升到0.4109，参数升至40,425；训练时间未见显著变化（140.4 vs 139.6s）。上下文门控宽残差把set-sum MAE恢复到0.2426，但binding MAE为1.3094，paired to sum 的binding MSE收益约10.7%；参数40,521。结果文件分别为 `training_budget_diagnostic/learning_curve/deep_sets/wide_screen_results.json` 和 `deep_sets/contextual_wide/screen_results.json`。未读取 sealed test，宽上下文候选不进入正式确认。

下一项在相同32 epoch预算复核 full Set Transformer 与上下文求和残差；如果注意力仍无法兼顾任务或成本过高，则停止这一组通用聚合候选，保持求和默认和明确的能力缺口。

## Full Set Transformer 延长训练验证筛查

seed94121、32 epoch、每族2048训练样本的 validation-only 对照见 `training_budget_diagnostic/learning_curve/full_attention/screen_results.json`。full Set Transformer 的 set-sum/binding验证 MAE分别0.4414/0.1481；上下文求和残差为0.1030/0.2884；DeepSets为0.2358/1.1102；求和为0.1782/1.4480。context residual相较求和两族都有改善，但两者仍未达到0.05。参数24,489；MPS前反向 median：入度64为3.94ms、256为4.02ms；求和约0.89/0.87ms。更长训练预算的判断只看训练/验证数据，不读取sealed test。

## MPS 重复性诊断（NE-024.16）

固定 seed 94121、相同训练/验证配置的两个 32 epoch validation-only 结果中，求和控制基本重复，而上下文 Set Transformer 明显分叉：此前筛查的 set-sum/binding MAE 为 0.1030/0.2884，本轮为 0.2872/0.8437。延长预算的本轮曲线在 128 epoch 达到 set-sum MAE 0.0309，却伴随 binding MAE 回升至 0.7098，不能据此认定双任务收敛。未读取 sealed test。

在 MPS 上开启 `torch.use_deterministic_algorithms(True)` 并执行注意力前向、反向和优化器更新，反向触发 `index_put_with_accumulate_mps does not have a deterministic implementation`。适配层已绕过 PyG `to_dense_batch` 的 `scatter_reduce_mps` 路径，但注意力的反向算子仍不支持确定性，因此无法靠当前算子路径保证逐 seed 重复训练。单 seed MPS 曲线只保留为探索性观察，不作为能力交付或模型选择的正式证据。NE-024.17 将检查可复现 CPU 评估和共享模型条件输出通道；在解决前不启动正式确认，也不解封 sealed test。

## 确定性 CPU 曲线与共享输出通道筛查（NE-024.17）

使用 `torch.use_deterministic_algorithms(True)`，对固定 seed 94121、每族2048训练样本/256验证样本做 CPU validation-only 对照；同一个 `ModelNeuron` 根据输入 context 的任务族选择其共享 `output_projection` 的输出通道，没有增加独立模型或任务专属参数。配置与结果：`training_budget_diagnostic/learning_curve/full_attention/cpu_deterministic/`。另以小样本固定配置重复运行两次，sum 和 contextual attention 两种候选的训练/验证指标逐项相同。

32 epoch 时，全局求和的 set-sum/binding 验证 MAE 为0.0557/1.4523；上下文 Set Transformer 残差为0.1093/1.1162。绑定验证 MSE 从4.1515降至3.0620（约26.2%），但两任务绝对 MAE 均未过0.05。延长至128 epoch后，残差候选 set-sum/binding MAE为0.0851/0.7075；binding仍远未达标，求和控制为0.0321/1.4557。运行成本约82秒；求和约19秒。模型解码任务族、较长预算和确定性 CPU 训练均没有弥合能力缺口。此筛查未读取 sealed test，结果不支持把候选用于正式能力交付；求和仍为默认聚合器。

下一步先对任务表示和聚合器结构做针对性设计复核：binding输入是否提供足够稳定的组关系信号、通用注意力是否以合适深度/交互表达该关系，以及开源可复用的分组/关系聚合实现。设计筛查通过前不进行新 sealed test 确认。

## 组关系输入与开源聚合复核（NE-024.18）

当前 grouped-binding 合成器把随机重标号后的组 ID 编成 `sin/cos(id * π / 2^k)`，`k=0..3`。该映射整体周期为32：ID 0与32、1与33等组 ID 的8维特征近乎完全相同（最大数值差约 `2.1e-14`）。因此它无法区分任意大的组集合，现有16组外推也没有覆盖别名边界。失败不能只归因于学习能力，输入表示本身存在确定性碰撞。

PyG `GraphMultisetTransformer` 是可直接适配的官方多代表元注意力池化候选：将元素汇总成 k 个代表元、在代表元间做 self-attention、再池化；但官方接口要求分组 index 排序，且它本身不消除上游组身份编码碰撞。[PyG GraphMultisetTransformer API](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.aggr.GraphMultisetTransformer.html)。

最小下一候选是把组身份作为 `ModelNeuron.step` 的可选整数关系元数据传入，由同一共享模型内部按组标识执行关系聚合；不再将整数身份压进周期性连续位置特征。必须先验证置换不变、组 ID 重标号不变、任意有效 ID 无别名、重复接触保留、空组/变长 mask 语义和反向传播，然后才做 validation-only 训练比较。

## 精确组身份与分组聚合实现（NE-024.19）

`ModelNeuron.step` 新增可选 `incoming_group_ids`（long `[batch, contacts]`）；组感知聚合器可要求该字段。新组件 `GroupRoleDeepSetsAggregator` 在每个整数身份组内按 edge type 汇总消息，再通过共享 MLP 编码各组并求和。身份只用于相等比较，不转换成浮点、不按最大 ID 分配表，也不排序输入；实现内存/运算为 O(N²)，当前无固定入度上限。消息数据去掉周期正余弦身份特征，改用 opaque int64 标签；set-sum 每条接触自成一组，binding 每对同组异角色。

CPU/MPS contract、重标号、排列、空集、padding、重复项和反向测试合计45项通过。`torch.use_deterministic_algorithms(True)` 下 MPS 训练与验证可完整运行；发现原来按样本 gather 任务输出通道的反向本身会触发 `scatter_reduce_mps`，已改为布尔 mask 选择再求和，确定性模式下通过。固定配置 MPS 两次训练的 train/validation/extrapolation 指标完全相同。

32 epoch、seed94121、每族2048样本的验证结果见 `training_budget_diagnostic/learning_curve/full_attention/cpu_deterministic/relation/final/`。训练组数2/4/8的 grouped-binding MAE为0.0458，set-sum为0.0429；求和控制分别为1.4485和0.1261。以16组作外推时，关系候选MAE为0.2467，仍未过0.05。分组候选13,321参数，MPS前反向 median约2.44ms@64、2.52ms@256；求和7,081参数，约1.38ms/1.55ms。新结构在训练范围内学会配对，但外推差距仍显著，未解封 sealed test，也不能声称通过能力验收。

## 训练组数扩展筛查（NE-024.20）

确定性 CPU、seed94121、32 epoch，把训练/常规验证的 binding 组数扩展到2/4/8/16，并在32组做独立 validation-only 外推，结果见 `.../relation/final/wide_group_counts/`。GroupRoleDeepSets 的 set-sum/binding 验证 MAE为0.0934/0.0602，16组范围的 binding MSE从求和6.2078降至0.01445，但两项绝对 MAE与32组外推MAE 0.4278均未通过。扩大训练组数并未保持此前2/4/8范围的低误差，表现出任务覆盖面之间的泛化/优化冲突。参数和成本与首轮相同（13,321；CPU约0.26ms@64、0.48ms@256）。未读取 sealed test。

## 图拓扑组身份贯通（NE-024.21）

`NeuronGraphTopology` 新增可选 `edge_group_ids`，`LinkMutation` 新增可选 `group_id`；节点重映射、上下文筛边、追加因果节点、删除/改型/改强度边都会保留或显式设置关系身份。`ModelNeuronGraph` 将静态组身份和候选边 `candidate_group_ids` 展开成每个目标神经元的消息组，并转交同一 `ModelNeuron.step`；关系聚合器遇到缺少身份的真实/候选边会报错。候选身份可为共享向量或按 batch 给出。CPU/MPS 图路径测试通过，且确定性算法模式下 MPS 两步图更新及反向通过。

验收测试包含直接 `ModelNeuron.step` 与图执行输出一致、候选边组 ID、超大 int64 标签、拓扑重映射/过滤/扩展/边变更的身份保留。NE-024.21完成。NE-024.22的 context 条件组编码筛查同样未同时达标：set-sum/binding MAE为0.1149/0.0546，32组外推0.4127；相对不带 context 的版本无实质改善。结果见 `.../relation/final/wide_group_counts/contextual/results.json`。

## 对照初始化公平性复核

复核发现 `SharedAggregationModel` 先构造候选聚合器、再初始化 `ModelNeuron` 核心，导致候选模块的随机初始化消耗改变了后续核心层初值。单 seed 对照因此有初始化混杂，尽管各候选固定 seed 可重跑。现改为先初始化完全相同的 ModelNeuron 核心，再挂载候选聚合器；新增逐参数断言证明相同 seed 下核心权重完全一致。此前单 seed候选对照仅作探索线索，不再用于决定候选；后续复核采用相同核心初值。

公平初始化后，2/4/8/16训练范围的pairwise角色交互验证 set-sum/binding MAE为0.1138/0.0492、32组外推0.3935；未满足两任务和外推阈值。公平初始化的求和残差预算曲线见 `.../pairwise/residual/budget/combined/results.json`：128 epoch时训练范围验证 MAE为0.0314/0.0307，优于求和控制0.0755/1.6535，但32组外推MAE仍为0.9147。更长预算解决了训练范围任务冲突，却没有解决规模泛化。

## 64组外推边界与容量归一化复核（NE-024.25）

训练/验证覆盖到32组、外推64组（128条消息）的确定性CPU曲线见 `.../wide_group_counts/pairwise/residual/budget/combined/wider/`。公平初始化下，求和残差在128 epoch的训练范围 set-sum/binding MAE为0.0619/0.1131，64组外推binding MAE为0.8878；32/64/128预算均没有通过。另将入度 context 从固定 `log1p(64)` 分母改为配置的 `log1p(capacity)`，重复曲线没有改善（0.062/0.113/0.888），但修正了容量改变时的归一化错误。分组组数扩大后关系候选仍无法泛化；未解封 sealed test。

## 64组规模外推的隐藏容量筛查（NE-024.26）

把共享 `ModelNeuron` 与逐组关系编码器 hidden size 从32增至64，在同样公平初始化、容量128、训练组数至32、64组外推的确定性CPU筛查中没有改善。hidden64的128 epoch结果：训练范围 set-sum/binding 验证 MAE为0.0794/0.0787，64组外推binding MAE为1.516；hidden32对应0.0619/0.1131和0.8878。hidden64参数升至59,849，前反向约0.37ms@64、0.73ms@256（hidden32为15,593参数，约0.32/0.54ms）。扩大隐藏容量没有通过能力门槛，且成本更高。关系组件虽在小范围任务可学，但尚不能满足更宽组数泛化；继续放宽容量/预算没有依据。

## 有界状态对无界加法任务的结构限制（NE-024.27）

基础 `ModelNeuron` 将聚合输入经 `tanh` 后送入 GRUCell，再只由隐藏状态生成输出。GRU 的候选态经过 `tanh`，更新态是候选态与旧态的 sigmoid 凸组合；从零初态开始，逐维状态保持在 `[-1, 1]`。因此固定 hidden size 的状态读出对所有入度有统一有限上界，不能精确实现随着消息条数无界增长的纯求和。这不是聚合器表达力可以修补的问题。

在 `ModelNeuron` 增加零初始化 `aggregate_output_projection` 直读支路：聚合器输出除以 `sqrt(degree)` 用于稳定状态驱动；直读支路乘回同一归一化因子，将未归一化聚合量映射到输出。零初始化令它开始时严格不改变既有输出。新增 CPU/MPS 解析合同测试：固定一份权重，对3条和128条全1消息分别精确输出3和128，且状态仍有界。此设计提供了无界加法输出路径；是否改善实际多任务/关系外推需单独训练验证。

## 聚合直读头训练验证（NE-024.28）

公平初始化、hidden32、容量128、训练组数2/4/8/16/32、64组外推的确定性CPU曲线见 `.../wide_group_counts/readout32/results.json`。zero-init直读头将set-sum训练范围验证MAE降至0.0217。求和残差候选128 epoch的set-sum/binding验证MAE为0.0343/0.0499，较无直读头时分别0.0619/0.1131；64组外推从0.8878改善至0.2746，但仍显著高于0.05。候选15,890参数，CPU前反向约0.34ms@64、0.55ms@256。结构修复改善未归一化加和和训练范围任务，但尚未解决未见组数外推。

## 组数覆盖64后的长预算趋势（NE-024.29）

公平初始化、hidden32/capacity256、训练与验证组数2/4/8/16/32/64、128组外推的确定性CPU结果见 `.../readout32/wider_coverage/`。求和残差在128 epoch的训练范围 set-sum/binding MAE为0.1393/0.0844、128组外推0.3260；延长至256 epoch后分别降至0.0028/0.0222，外推降至0.1406。误差仍在明显下降，尚未收敛至外推阈值，因此冻结同一数据/配置再延长到512 epoch作为最后一轮预算趋势判断。未读取 sealed test。

## 512 epoch 最终预算结果（NE-024.30）

同一容量256、组数训练到64、128组外推切分的求和残差候选，在512 epoch没有继续改善：256 epoch的训练范围验证 set-sum/binding MAE为0.0028/0.0222、128组外推0.1406；512 epoch变为0.0053/0.0280、外推0.1581。训练误差轻微回升，外推恶化，说明这一配置在256 epoch附近已到最佳附近；更长预算不支持跨越外推缺口。停止本切分的预算扩展，不访问sealed test。

## 原16组外推预算收口（NE-024.31）

公平初始化、hidden32、aggregate直读头、训练binding组数2/4/8，以16组作独立 validation-only 外推。结果见 `.../readout32/holdout16/`：16组外推MAE由32 epoch的0.142降至64的0.114、128的0.078；延长至256后反升至0.165。256 epoch时训练范围binding验证MAE0.0481、set-sum0.0047，显示继续拟合训练范围并未改善未见16组。排列差异低于1e-5，但原16组绝对误差门槛未通过。停止该变体的预算调参，保留关系能力缺口。

## 低秩双线性交互通过原16组验证筛查（NE-024.32）

`GroupRoleDeepSetsAggregator` 的同通道 Hadamard 角色乘积替换为角色独立线性投影后逐维相乘（低秩双线性交互），保留零初始化求和残差和完整ModelNeuron核心共享。公平初始化、训练binding组数2/4/8、16组外推的128 epoch确定性CPU结果见 `.../readout32/holdout16/bilinear/final_results.json`：set-sum/binding验证MAE为0.01135/0.00893，16组外推MAE为0.01364、MSE0.0002905、最大绝对误差0.04689；16组排列最大偏差 `9.54e-7`，训练范围排列偏差 `4.77e-7`。CPU前反向0.370ms@64、0.587ms@256，参数17,938。

开启 `torch.use_deterministic_algorithms(True)` 后，MPS两次同seed训练更新的输出、loss和完整state_dict逐项完全相同；MPS median成本2.674ms@64、2.744ms@256（求和基线2.754/2.740ms），256入度p95 4.179ms。该候选达到当前validation能力门槛，允许冻结配置进入六个全新seed确认；验证屏幕未读取sealed test。

## 六seed正式确认未通过（NE-024.33）

按冻结配置 `training_budget_diagnostic/formal_confirmation/config.json` 完成6个全新seed（260000–260005）的同共享模型确认，结果见同目录 `results.json`。封存测试 set-sum/binding 平均MAE为0.0238/0.0240，排列误差约`1.8e-6`/`7.2e-7`；binding paired MSE平均改善99.94%（95% t区间约[99.80%,100.07%]）。但16组 validation 外推MAE均值0.0594，seed范围0.0069–0.2206，未满足0.05；封存测试也有样本最大绝对误差超过0.05。按冻结规则正式确认失败。封存指标只登记，不用于修改或重训当前候选；后续若继续，必须使用新validation配置与新seed。

## 新seed样本量与外推稳定性诊断（NE-024.34）

为查明16组外推seed方差是否可由更多训练样本缓解，冻结低秩bilinear候选、hidden32、公平初始化及128 epoch，仅比较每任务1024与2048条训练样本；使用全新seed 270000–270003，所有报告均为validation-only。配置和逐seed结果位于 `training_budget_diagnostic/formal_confirmation/validation_robustness/`。本轮不生成、不读取六seed正式确认中的sealed split。

16组外推MAE在1024样本下均值为0.1294、seed标准差0.0912、最大值0.2649；2048样本下均值降至0.0730，但标准差仍为0.0966、最大值0.2138，只有2/4个seed达到0.05。训练样本翻倍降低平均误差，却未消除seed不稳定，也未满足预注册门槛。NE-024.34按失败分支完成，不启动第二次正式确认；NE-024仍未完成，本候选停止继续调参，能力报告不得把该实验结构宣称为通过的通用关系聚合器。

## 高入度分组聚合的线性 workspace 路径（NE-024.35）

审查发现精确组关系聚合器原先对所有入度构造 `N×N` 相等矩阵。现保留有界稠密快路径，并在 pairwise workspace 超出预算时按 opaque group ID 稳定排序，使用分段前缀和；自定义反向将梯度按段长度广播，并以唯一索引写回原顺序，避免 MPS 确定性模式下不支持的累加 scatter。标签值仍不用于embedding、索引或分配；分段路径临时 workspace 随消息数线性增长，未设置最大入度。

CPU/MPS pair-workspace预算分别为 `256²` 和 `3000²` 个配对项。前反向中位耗时（ms，8轮测量、丢弃前2轮；每轮重新建叶子张量）如下，展示旧稠密参考和排序分段路径的成本：CPU N=64 为0.120/0.231、256为0.333/0.262、1024为1.189/0.563、4096为19.903/1.281；MPS N=64为0.917/4.551、256为1.269/4.507、1024为1.620/2.702、2048为6.297/7.945、4096为17.928/5.460。常见范围保留稠密核；高入度切换以限制内存并改善高规模延迟。

稠密与分段路径输出及消息梯度、参数梯度对照通过；CPU 16,384个唯一组和MPS 4,097入度确定性反向通过。相关神经元/聚合测试129项通过。此项只解决可扩展性和工程成本，不修复已停止候选的seed外推不稳定，NE-024仍未通过能力验收。

## 显式二阶角色绑定读出筛查（NE-024.36）

结构动机参考两类一手工作：Deep Sets 将无序集合函数写成逐项编码后求和的形式；2024 年关于向量/张量不变函数的结果进一步研究可识别多重集的和分解，但给出的普适表示维数仍依赖集合大小上界，不能据此声称有限模型已解决无界关系泛化。[Deep Sets](https://arxiv.org/abs/1703.06114)；[Tabaghi & Wang, ALT 2024](https://proceedings.mlr.press/v237/tabaghi24a.html)。角色-填充值绑定常用张量积表达，本轮据此推导出一个适配本任务的候选：每组先算角色低秩投影逐元素乘积，直接线性读出，再跨组求和，去掉逐组通用MLP。这个映射到当前任务的结构选择是设计推论，不是上述论文对本模型性能的保证。[Tensor Product Decomposition Networks](https://arxiv.org/abs/1812.08718)。

冻结验证配置位于 `training_budget_diagnostic/formal_confirmation/structural_candidates/exact_bilinear_group_sum/config.json`，四个新seed为280000–280003，每任务2048训练样本、hidden32、128 epoch、训练组数2/4/8，独立validation外推16组；只与同seed公平初始化 `global_sum` 比较，未生成或读取sealed数据。结果见同目录 `results.json`。候选参数10,578；四seed的训练范围 set-sum/binding MAE均小于0.05，binding validation MSE相对求和改善均超过99.9%，排列偏差低于`4e-6`。但16组外推MAE逐seed为0.1085、0.00427、0.0519、0.00279，均值0.0419、标准差0.0499，2/4 seed未达0.05。故冻结门槛失败，不启动正式sealed确认，不据结果调配置；该新候选也停止。NE-024仍未满足通用关系能力验收。

## 完整双线性角色张量筛查（NE-024.37）

在低秩二阶结构 seed 方差失败后，将角色交互替换为 PyTorch `nn.Bilinear` 完整双线性层；输入、共享 `ModelNeuron`、零初始化求和残差和 128 epoch 训练均固定。四个全新 validation seed 290000–290003 与同 seed 求和基线配对，配置/结果见 `training_budget_diagnostic/formal_confirmation/structural_candidates/full_bilinear_group_sum/`。训练范围 set-sum/binding MAE逐seed为0.00208/0.00170、0.01138/0.00217、0.00347/0.00592、0.00195/0.00361；16组外推MAE为0.00286、0.00577、0.01493、0.01411，四seed均低于0.05。binding validation MSE较求和基线改善均超过99.998%，最大排列差低于`2.4e-6`。模型参数40,274。

CPU训练验证四seed总耗时约半小时；聚焦CPU/MPS语义和梯度测试已通过，包含高入度线性workspace路径。按用户授权在内网RX 7800 XT（ROCm 7.2.53211，PyTorch 2.15.0）实测，确定性算法开启。完整模型40,274参数，入边数64/256/1024/4096的前后向中位时延为3.46/4.45/8.81/16.53 ms，p95为3.47/4.48/8.86/16.58 ms，完整原始结果见`full_bilinear_group_sum/rocm_cost/results.json`。NE-024.37按冻结validation标准完成；此结果仅是所测硬件/宽度的成本证据，不表示无界关系泛化已通过。sealed test未访问。

## 完整双线性六 seed 正式确认（NE-024.38）

NE-024.37全新四 seed validation全部通过后，冻结六个新seed 300000–300005、hidden32、每族2048训练/256验证与sealed样本、128 epoch、确定性ROCm，比较`full_bilinear_group_role_sum`和`global_sum`。原始冻结配置与逐seed完整结果见`training_budget_diagnostic/formal_confirmation/structural_candidates/full_bilinear_group_sum/formal_confirmation/`。

在RX 7800 XT（PyTorch 2.15.0、HIP 7.2.53211）上运行完毕。完整双线性候选validation/sealed平均MAE：set-sum 0.00962/0.00950，grouped-binding 0.00468/0.00470；16组validation外推平均MAE 0.01329。paired sealed binding MSE相对求和改善99.9987%，95%区间下界99.9973%；最大排列差3.82e-6。64/256/1024/4096入边的前后向中位成本跨seed中位数为3.73/4.72/9.13/17.10 ms，40,274参数。

该候选未通过严格最大误差门槛：set-sum validation 最大绝对误差0.06877、sealed最大绝对误差0.12146；16组validation外推最大误差0.09667，均超过冻结0.05。binding验证与sealed最大误差≤0.03556。按预注册规则停止该候选，不从sealed数值调参或重训。均值能力和配对收益良好，但NE-024尚不满足完整验收。

## 标准化 L1 训练目标 validation-only 筛查（NE-024.39）

为检验MAE误差尾部，仅改变全新候选的训练目标：保持full bilinear结构、样本量、初始化与更新预算不变，将按`sqrt(入边数)`归一化的MSE改为同尺度L1；对照global sum使用相同目标。使用新seed 310000–310003，仅生成训练与validation集，不访问sealed数据。冻结配置位于`training_budget_diagnostic/formal_confirmation/structural_candidates/full_bilinear_group_sum/robust_objective_validation/config.json`。

L1筛查未改善误差尾部：四seed set-sum MAE为0.13565/0.02438/0.05162/0.07055，三seed超过0.05；最大误差最高0.31429。grouped-binding MAE为0.02303/0.00657/0.01752/0.00664，16组外推MAE为0.04751/0.03584/0.03611/0.01358；最大误差仍最高0.17804。结果见同目录`robust_objective_validation/results.json`，文件确认未生成或读取sealed测试集。按预注册规则停止该损失候选，不依据本批validation继续调整。

## 标准化 L1 目标筛查结论（NE-024.39）

L1筛查未改善误差尾部：四seed set-sum MAE为0.13565/0.02438/0.05162/0.07055，三seed超过0.05；最大误差最高0.31429。grouped-binding MAE为0.02303/0.00657/0.01752/0.00664，16组外推MAE为0.04751/0.03584/0.03611/0.01358；最大误差仍最高0.17804。结果见同目录`robust_objective_validation/results.json`，未生成或读取sealed测试集。按预注册规则停止该损失候选，不依据本批validation继续调整。

## 精确原始消息加和直通候选（NE-024.40）

本轮保持full bilinear角色分组交互，额外给ModelNeuron增加无参数的`Σ(message × strength)`直通输出，并在启用该结构时将普通隐藏态输出projection置零初始化。这样线性加和从初始前向就精确保留，binding仍依靠共享隐藏态与full bilinear分支学习。CPU smoke已核对模型输出逐位等于加权原始消息和；冻结validation配置位于`training_budget_diagnostic/formal_confirmation/structural_candidates/raw_sum_bilinear/validation/config.json`，fresh seed 320000–320003，不触及sealed数据。

四个seed全部在内网RX 7800 XT上完成；每个seed均记录PyTorch 2.15.0、HIP 7.2.53211、实际设备名和同步矩阵乘法探针checksum。独立GPU采样日志中card1共2803次采样，平均忙碌率97.19%、中位98%；card0始终为0%。每个模型40,274参数，训练中位约349.45秒/seed；前后向成本中位数按64/256/1024/4096条入边为4.27/5.32/9.24/17.14 ms。完整逐seed数据和GPU采样见该validation目录的`results.json`与`gpu_activity.log`。

候选未通过冻结门槛。Set-sum validation MAE逐seed为0.000396/0.110701/0.000354/0.000052，最大误差最高0.303429；seed 320001单独即超过MAE与最大误差阈值。Grouped-binding MAE为0.001572/0.003938/0.005983/0.016592，最大误差最高0.050481；16组外推MAE为0.005689/0.007301/0.015236/0.070194，最大误差最高0.106091，seed 320003外推失败。排列偏差≤2.39e-6。sealed数据未生成或读取；按冻结规则停止候选，不据该批validation继续调参。

这次结果也暴露了设计上的关键点：只在初始化时将普通输出projection置零，不能保证训练后可学习分支不会污染精确加和通道。CPU smoke证明的只是初始化语义，不能证明训练全过程的不变量。若后续重新提出该方向，必须先在接口/结构上隔离加和输出与可学习关系输出，再单独设计新验证方案；本轮不追加实验。NE-024整体关系能力验收仍未通过。

## 隔离求和与关系输出候选（NE-024.41，validation-only）

针对NE-024.40暴露的输出污染和NE-024.38的外推尾部，冻结一个不同的结构候选：ModelNeuron输出通道0以`torch.where`直接选择原始加权消息和，隐藏状态与所有可训练读出都无法写入该通道；输出通道1只选择关系聚合读出，不混入GRU状态。组关系路径仍然通过共享消息投影、角色嵌入、opaque group-id分组和共享双线性映射计算。full bilinear逐组映射去除bias，使缺少交互信号的组不会凭常数项贡献输出，避免该项随组数累积。通道以外的普通状态/输出路径保持现状；模型仍只有一份共享权重。

验证前本地契约检查已确认：任意训练后输出/聚合projection权重下，通道0逐位等于加权原始消息和；该通道对消息的梯度正好等于strength，其他输入通道梯度为零；空集合输出零；修改GRU状态读出权重不会影响通道1；无bias候选的架构签名与有bias候选不同，防止检查点误认为结构兼容。聚焦聚合器测试35项通过，Ruff与`newera-gate --stage explore`通过。全新seed 330000–330003，配置冻结于`training_budget_diagnostic/formal_confirmation/structural_candidates/isolated_sum_relation/validation/config.json`，完整ROCm结果和采样日志在同目录。

四seed均在RX 7800 XT（PyTorch 2.15.0/HIP 7.2.53211）完成，并逐seed通过同步GPU矩阵预检；562次card1忙碌采样均值97.45%、中位98%，card0的562次采样始终为0%。Set-sum原始输出通道MAE约2.44e-7–3.00e-7、最大误差≤2.39e-6，精确聚合不变量达标。关系通道失败：binding validation MAE逐seed0.002108/0.000141/0.074064/0.001019，最大误差最高0.414745；16组外推MAE为0.007006/0.001758/0.243687/0.003367，最大误差最高0.814643。seed 330002同时突破训练范围和16组外推门槛。模型40,242参数，训练耗时350.24/351.88/349.58/351.50秒；前后向成本中位数按64/256/1024/4096条消息为5.65/6.38/10.27/17.87ms。`results.json`确认sealed数据未生成或读取。按预注册规则停止该候选，不按同批validation调整。

因此，结构隔离修复了求和通道泄漏，却没有解决跨seed关系学习不稳定；移除full bilinear的逐组bias也没有保证稳定泛化。新候选需要重新审查角色投影中的冗余、初始化/梯度尺度和共享组求和的可辨识性，再设计全新seed的冻结验证；当前结果不用于改参。

## 无偏置因子化角色乘积候选（NE-024.42，validation-only）

NE-024.41失败seed的训练MAE接近零但validation误差明显上升，说明问题不是简单的欠拟合。复核模型输入后发现：role ID已经用于按角色拆分组摘要，却仍额外以可训练embedding加到每条消息；该项可能让每组输出依赖角色接触计数而非消息内容。新候选将role embedding从ModelNeuron消息路径移除，让edge type只用于聚合分组；每个组分别汇总两角色的共享线性投影，对应向量逐通道相乘，再通过无bias线性层映射并对组求和。此rank-H双线性映射适合乘性交互且参数量低于完整H×H×H张量。关闭sum residual和所有逐组bias，关系readout也无bias，因此无有效组时输出严格为零；通道0继续直接输出strength加权原始消息和，关系通道绕开GRU状态。

现有grouped relation核心保留其CPU/MPS/ROCm确定性分段求和路径。新增契约覆盖无bias因子化的行置换/group label重标、有限梯度、无edge embedding的ModelNeuron集成，以及训练后readout bias变化不污染空关系输出。配置冻结于`training_budget_diagnostic/formal_confirmation/structural_candidates/factorized_role_product/validation/config.json`，fresh seed 340000–340003；聚焦聚合测试40项、Ruff与工程gate通过。正式GPU运行必须先匹配RX 7800 XT和HIP 7.2.53211并通过同步GPU运算探针，禁止回退；validation失败不触及sealed，也不按同批数据复调。

四个validation seed均通过冻结门槛，且每seed都记录RX 7800 XT、HIP 7.2.53211和同步矩阵乘法探针。card1共379次忙碌采样，均值95.73%、中位97%；card0共379次，始终0%。训练binding MAE为0.00000030/0.00000031/0.00000029/0.00000030，训练最大误差≤5.73e-6；validation binding MAE为0.003478/0.0000053/0.0000019/0.0000340，最大误差≤0.018871；16组外推MAE为0.007089/0.0000108/0.0000041/0.0000682，最大误差≤0.023692。Set-sum MAE≤3.20e-7、最大误差≤3.82e-6；置换偏差≤2.15e-6。参数10,386；四seed候选训练120–126秒，前后向成本中位数按64/256/1024/4096项为1.427/1.406/1.803/3.329ms。相同seed的full-bilinear对照有binding与16组外推失败seed。完整原始结果及采样日志在`factorized_role_product/validation/`。sealed仍未读取；该候选只获得进入正式确认的资格，不代表NE-024已完成。

## 因子化角色乘积六seed正式确认（NE-024.43）

由于NE-024.42四个全新validation seed的训练/验证两族、最大样本误差、16组外推和置换指标全部通过，冻结新的独立正式确认配置`training_budget_diagnostic/formal_confirmation/structural_candidates/factorized_role_product/formal_confirmation/config.json`。候选为`factorized_role_product_group_sum`，公平对照为`global_sum`；使用seed base 350000的六个新seed、2048训练与256 validation/sealed样本每任务、hidden32、128epoch和确定性ROCm。新sealed split只在此冻结运行中生成并读取一次；所有指标和paired improvement区间完成后封存结论，不因结果再调参。
