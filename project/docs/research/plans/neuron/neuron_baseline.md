# NE-004 最小 MLP 行为基线

日期：2026-11-06

## 控制实现

`newera.neuron.MLPNeuron` 是无状态、两层全连接控制实现，逐步调用 `step(inputs, state=None)` 并返回 `NeuronResult`。输入最后一维必须等于 `input_size`；它不吞掉外部状态，收到非空状态时明确报错。实现参数可训练，可作为后续候选的共同接口与能力下限对照。

## 可重复的 CPU 训练 smoke

复现命令：

```sh
uv run --no-sync python - <<'PY'
import torch
from newera.neuron import MLPNeuron

torch.manual_seed(31)
x = torch.randn(64, 2)
y = 0.5 * x[:, :1] - 1.2 * x[:, 1:2] + 0.4
neuron = MLPNeuron(2, 16, 1)
optimizer = torch.optim.Adam(neuron.parameters(), lr=0.03)
for _ in range(80):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(neuron(x).value, y)
    loss.backward()
    optimizer.step()
print(torch.nn.functional.mse_loss(neuron(x).value, y).item())
PY
```

| 项 | 记录 |
| --- | --- |
| 参数量 | 65 |
| 数据 / 优化步 | 64 个合成样本 / 80 步；共处理 5,120 个样本 |
| MSE | 初始 2.05217；80 步后 0.006378 |
| 单样本前向线性计算 | 48 MAC（`2×16 + 16×1`），不含 bias、tanh 和其他算子 |
| 本轮前向线性计算代理 | 245,760 MAC；这是训练计算的前向部分，不包含反向、优化器或框架开销 |
| 设备 | CPU |

该结果仅验证接口、可训练性和测试数据上的过拟合闭环，不是泛化或能力证据；合成映射本身就是本轮训练目标，没有独立验证/测试划分。

## NE-007 训练、验证与中断恢复

`project/scripts/experiments/training/neurons/capabilities/run_neuron_smoke.py` 提供 CPU 端到端命令。训练集 64 条、验证集 32 条，由不同固定种子生成；没有测试集，也不把验证误作最终泛化结果。检查点包含 schema 版本、模型配置、模型和优化器状态、累计步数及 Torch CPU RNG 状态，并通过临时文件替换保存。`--steps` 表示本次新增步数，`--resume` 从已有检查点继续。

自动化恢复测试比较 80 步连续运行与 35 步保存后再恢复 45 步：最终模型 SHA-256、训练/验证 MSE 完全一致。这个小任务验证了本项目当前使用状态的保存恢复，不代表通用训练框架或跨设备精确复现。

## NE-009 首个任务能力

`RegressionCapability` 用 `RegressionTaskAdapter` 将结构化样例映射为 Tensor，并将输出解码回预测值；训练目标和模型保持分离。可运行入口为 `project/scripts/experiments/training/neurons/capabilities/run_regression_capability.py --seed 31 --steps 80`。固定运行记录：CPU，训练 64 条、验证 32 条，65 个参数，48 线性 MAC/样本/前向；训练 MSE 从 2.21951 降到 0.006104，验证 MSE 为 0.020641，验证 MAE 为 0.083852。无测试集，不能据此声称泛化能力；验证集只作 smoke 检查。

## NE-010 第二种能力

`SequenceSumCapability` 使用 `StatefulMLPNeuron(1, 8, 1)` 逐时间步处理长度为 5 的序列。入口 `project/scripts/experiments/training/neurons/capabilities/run_sequence_capability.py --seed 41 --steps 250` 在 128 个训练样例和 64 个独立种子验证样例上完成 CPU 训练/评估：训练 MSE 1.64461 → 0.000338，验证 MSE 0.000668、MAE 0.016631；457 个参数，约 400 MAC/时间步或 2,000 MAC/序列。样例由同一合成求和规则生成，验证结果只说明此合成任务可拟合。

## 测试

`project/tests/models_training/contract_scopes_stage/neuron/test_neuron_baseline.py` 验证协议可替换调用、梯度、状态拒绝、输入维度错误，以及固定种子的合成映射可学习。当前 smoke 只在 CPU 执行。
