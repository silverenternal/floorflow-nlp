# 神经元图层验证 v5：稳定身份与拓扑操作

## 目标与配置

本轮评估稳定节点身份在任意槽位置换下是否保持行为、4/8/12/16 神经元池变化、add/remove/retype/strength 操作、双边组合任务，以及 MPS 训练和推理成本。每个样本随机安排身份槽和 readout，padding 身份为 `-1`。训练只使用检索/组合任务 MSE，不向优化器提供边或操作标签。配置、运行器、正式原始结果分别见同目录的配置 `project/configs/models_training/learning/neuron_comparisons/graph/graph_validation_identity_permutation.json`、脚本 `project/scripts/experiments/training/neurons/graph/run_neuron_identity_evolution.py` 和 JSON 结果。

按预注册规模运行六个种子、两个任务、每模型 800 步，设备为 MPS。参数和 MAC 对照均先将节点按稳定 ID scatter 到规范槽位；因此 MLP 不会因随机槽位而受到额外惩罚。首轮未规范化控制的原始结果保存在 `preliminary_unmatched_control_results.json`，仅供追踪实验设计更正，禁止用于结论。

## 实验结果

我们先运行了身份规范化控制下的软候选权重训练，随后增加 straight-through 硬动作前向和软概率反向梯度，并在同一数据种子上复跑。软训练结果保存在 `soft_action_baseline_results.json`，直通结果在 `identity_evolution_results.json`。

| 指标 | 软候选：检索 / 组合 | 直通硬动作：检索 / 组合 | 门槛 |
|---|---:|---:|---:|
| 参数匹配 MLP 相对 MSE 差 | -99.83% / -35.74% | -44.72% / -16.12% | 均值 ≤ -20%，95% CI 上界 < 0 |
| 线性 MAC 匹配 MLP 相对 MSE 差 | -99.82% / -34.14% | -44.43% / -14.24% | 同上 |
| edge F1 | 0.341 / 0.250 | 0.343 / 0.252 | ≥ 0.80 |
| strength MAE | 0.125 / 0.266 | 0.324 / 0.335 | ≤ 0.20 |
| 置换 MSE 相对漂移 | 0 / 0 | 0 / 0 | ≤ 0.02 |
| 身份边集合 Jaccard | 1 / 1 | 1 / 1 | ≥ 0.95 |

软训练的结构操作 recall：检索 remove=0、retype=0、strength=0.995；组合 add/remove/retype 均为 0、strength=0.549。直通训练的 recall：检索 remove=0.018、retype=0、strength=0.252；组合 add=0.040、remove=0.052、retype=0、strength=0.034。直通动作偶有高精确率的 remove，但整体 recall 和 F1 仍远低于门槛。

因此，节点槽位置换不影响输出或稳定身份边集合；任务预测质量与硬拓扑学习却没有同时达标。软候选训练的相对 MSE 很好，但离散图不正确；直通硬动作令训练看到离散决策，却明显损害预测质量且未学会足够的操作。两种训练模式都未通过结构综合门槛，不能宣称动态拓扑已学成或指标刷满。

## 规模和成本

软训练评估中，4 到 16 个记忆神经元的 edge F1 约为检索 0.320–0.351、组合 0.239–0.255；池规模变化下执行稳定，但结构精度仍差。MPS 图层平均训练约 32.82 秒/800 步，参数与线性 MAC 对照约 8.55 秒/800 步。直通训练图层约 33.41 秒/800 步。

参数量：图层 7,040，参数匹配 MLP 7,053（误差 0.185%）；线性 MAC：图层 79,168，MAC 匹配 MLP 79,173（误差 0.0063%）。MAC 仅按 Linear 输出元素数乘输入宽度统计，不包含 scatter、归约、逐点操作、优化器和内存流量，因此不代表总 FLOPs。MPS 单样本前向 4–16 个候选边的中位数约 10.6 ms。主机 RSS、MPS allocated 和 driver allocation 用 2 ms 采样，可能漏掉短时峰值。

## 实现

- `LinkMutation` / `TopologyController` 支持 `set_strength` 并校验目标边、有限非负强度。
- `LinkProposalPolicy` 可选预测连接强度；预测值用于软候选权重及硬 add/retype/strength 更新。默认关闭，维持既有参数布局。
- `expected_type_weights(..., straight_through=True)` 提供硬动作前向和软概率反向梯度，训练器通过配置启用；没有加入边/操作监督。
- 对照 MLP 按 stable ID 规范化输入；训练批次无放回抽样，避免重复样本造成批次边索引重映射错误。
- 神经元模型配置增加 `neuron_graph_predict_strengths`。

## 证据限制与下一步

两轮方法迭代复用了相同种子对应的测试样本；因此这里的对照和改进过程属于探索性证据，不能作为独立确认性测试。合成任务不支持自然语言或通用能力结论；池规模只覆盖 4–16，MPS 内存是采样估计，MAC 不是总 FLOPs；身份注册表仍由调用方负责持久化。

下一项应从训练集/验证集解决软任务目标与离散连线决策之间的错位，再用全新种子和测试样本作一次性确认；结构验收阈值保持不变。验证：MPS 共运行 24 个模型训练实例（两种策略 × 两任务 × 六种子）；全量测试 `595 passed / 21 skipped`；Ruff、`uv lock --check`、JSON 与 `newera-gate --stage explore` 均通过。测试日志中 `dynamic_stack.py` 的 Python 3.16 弃用警告是既存代码告警。
