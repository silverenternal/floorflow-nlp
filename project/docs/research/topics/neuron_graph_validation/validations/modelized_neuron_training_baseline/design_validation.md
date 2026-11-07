# NE-030：训练与评估基线闭环

日期：2026-11-07

## 目的与范围

此记录验收训练预算、指标语义、控制组、检查点与小型运行闭环。它不是神经元能力通过报告。正式的六种子确认训练未运行；完整配置每个模型有 1,200 个优化步、每步累积 24 个独立图样本，只有确认数据与预算后才应启动。

历史实验内容未覆盖。修正后的实现使用 `project/configs/models_training/learning/neuron_comparisons/topology_evolution/modelized_neuron_training_baseline.json` 与本目录中的诊断记录。

## 实现结果

- `ModelNeuronGraph` 目前每次处理一个图。`batch_size` 现在通过逐样本梯度累积实现；一个优化步先对指定数量的独立样本反传，再按实际窗口样本数平均梯度。结果记录请求/有效 batch、优化步和消费样本数，不将其称为图批处理。
- 共用训练器处理累积、样本抽样、有限损失检查和梯度连接诊断。任务适配器只编码输入、解码输出，不实现检索答案算法。
- 检查点保存模型与图级参数、优化器、Python/NumPy/PyTorch RNG、样本抽样 RNG、身份注册表、调用方状态以及配置身份；写入复用项目已有原子保存和 RNG helper。配对模型使用相同的采样种子和样本序列。独立训练样本之间状态会重置，因此训练检查点的 `node_state` 明确为 `null`。
- 已有检查点可从 `--resume` 恢复训练；`--evaluate-checkpoint` 可单独载入指定共享神经元检查点，在验证集选择阈值并输出测试预测与动作记录。
- 输出使用原子 JSON 替换；训练、验证、测试阶段逐步报告进度。设备由配置指定；MPS 不可用时直接报错，不切换到 CPU。
- 边 precision/recall 与动作 precision/recall 记录计数分子、分母；分母为零时结果是 `null`。两空图的 Jaccard 为 1，同时独立记录空图比例。参考误差不大于配置 epsilon 时，相对误差为 `null`。
- 对照组包括零预测、训练标签均值、解析标签 oracle 上界，以及参数/MAC 匹配 MLP。oracle 直接使用生成标签，只是上界，不是推理模型。

## smoke 结果

CPU 与 MPS 各运行一组 smoke：单个种子、每任务 16 个训练样本、4 个验证样本和4个测试样本、3 个优化步、有效 batch 2。两个设备均完成训练、验证阈值选择、测试、结构统计、checkpoint 写入和重新载入。CPU 上另运行了梯度累积等价、单样本过拟合和中断续训对照；指定 checkpoint 的独立评估输出 4 个预测和动作列表。

CPU 诊断的单链检索 MSE 为 0.11910，训练均值控制为 0.00400，零预测为 0.00417。模型预测边数为 0，期望边分母为 8、边 recall 为 0；原图与置换图均为空，故 Jaccard 为 1，但空图比例为 1.0。单样本 probe 的初始损失很低，5 步后变高。此任务显然没有学会有效检索。

双链组合的模型 MSE 为 0.12984，训练均值控制为 0.05070，零预测为 0.05050。单样本 probe 损失下降，且 20/20 参数张量在首个微批次上有有限且非零梯度；这不代表跨样本泛化或路由结构正确。该任务在 smoke 中也未优于常数控制。

这些小样本值高度受样本抽取影响，只用于验证测量和恢复链路。它们不支持能力、性能优势或统计显著性主张；单链塌缩和控制组差距移交 NE-022 定位。

## 复现

CPU smoke：

```sh
uv run python project/scripts/experiments/training/neurons/run_modelized_neuron_graph_validation.py \
  --config project/docs/research/topics/neuron_graph_validation/validations/modelized_neuron_training_baseline/smoke/experiment_config.json \
  --output project/docs/research/topics/neuron_graph_validation/validations/modelized_neuron_training_baseline/smoke/results.json
```

MPS smoke 使用 `mps_smoke/experiment_config.json`，配置明确指定 `mps`。两个 smoke 目录保存实际配置、结果、身份注册表与 checkpoint。

单个 checkpoint 可单独评估：

```sh
uv run python project/scripts/experiments/training/neurons/run_modelized_neuron_graph_validation.py \
  --config project/docs/research/topics/neuron_graph_validation/validations/modelized_neuron_training_baseline/smoke/experiment_config.json \
  --output /tmp/newera-checkpoint-evaluation.json \
  --evaluate-checkpoint project/docs/research/topics/neuron_graph_validation/validations/modelized_neuron_training_baseline/smoke/results_checkpoints/single_link_retrieval/seed_901/modelized_neuron_graph.pt \
  --task single_link_retrieval --seed 901
```

## 检查记录

- `uv run pytest project/tests/other/neuron_model_evaluation -q`：6 passed。
- Ruff 对共用 runner、实验入口和新增行为测试通过。
- `uv run newera-gate --stage explore`：通过，findings 为空。
- CPU/MPS smoke 都显式完成；本记录不包含完整六种子性能或能力结果。
