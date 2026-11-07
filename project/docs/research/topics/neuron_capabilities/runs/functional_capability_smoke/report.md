# 首轮功能能力 smoke 记录

此记录验证任务生成、独立 oracle、共享模型训练调用、CPU/MPS 执行和留出规模评估链路，不是正式能力结论。

- 配置：`project/configs/models_training/learning/neuron_comparisons/functional/functional_capability_suite.json`
- 实现：`project/scripts/experiments/training/neurons/capabilities/functional/run_functional_capability_suite.py`
- 模型：单份共享参数 `ModelNeuron(input=3, hidden=32, message=1)`；每个样本持有独立短暂状态。
- 每族训练 8 个样本、一次优化更新；验证与外推各 8 个样本。样本由固定种子域生成；封存测试未读取。
- CPU/MPS 两种设备均完成运行，结果分别见 `cpu_results.json` 与 `mps_results.json`。
- 集合归约、绑定和回忆三族均未达到配置中的目标误差。样本量远低于正式配置，因此不能据此判断模型不可表达或推断泛化表现。

需要正式运行后再区分：表示瓶颈（如顺序扫描隐藏状态无法稳定保留分组对应关系）与优化/训练不足。外推评估只作为诊断，封存测试仍未使用。
