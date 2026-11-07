# DynamicNeuronGraphLayer 指标契约 v1

## 目标

指标要回答三件事：图是否改善目标任务、学到的连接是否有作用、这些连接付出了多少运行成本。节点隔离与因果约束是正确性门禁，不是可相互抵消的分数。

## 必报指标

| 面向 | 指标 | 计算口径 |
| --- | --- | --- |
| 任务质量 | 验证集交叉熵、next-token accuracy、相对无图差值 | 同种子、同数据摘要、同训练步数；图模型另报参数量，不把未匹配参数的差值解释为连线机制的因果收益。 |
| 连线效用 | 原策略、度数匹配随机因果图、关闭图消息 | 同一个训练后模型评估。随机图保留每个目标节点的入度，并在其因果前缀内重采样来源；关闭消息通过将图残差门设为零。 |
| 结构规模 | 有效节点数、边数、活跃边数、边密度、加权边质量、加权入度中位数/P95/P99/最大值、硬边预算使用率 | 分位数只在有效节点上计算，索引为 `floor(q * (n - 1))`；密度分母为有效 segment 内全部可能的过去到未来节点对；边预算未配置时不输出使用率。 |
| 表示状态 | 分图层输入、图层输出、最终解码表示的归一化有效秩；原始与尺度归一化 Dirichlet energy | 有效秩为奇异值熵指数除以 `min(节点数, 特征维数)`；原始能量随激活尺度变化，跨层优先看相对能量 `Σw||hᵤ-hᵥ||² / Σw·(||hᵤ||²+||hᵥ||²)/2`。均作为辅助诊断，不单独判定过平滑。 |
| 长距离传递 | 目标 token loss 对图层输入节点表示的梯度范数，按距离分桶 | 固定验证样本，对目标所在 segment 的所有较早位置采样，报告距离桶 `1–3, 4–7, 8–15, 16–31, 32–63, 64+` 的样本量、平均和中位范数；这是局部敏感度诊断，不等于完整因果归因。 |
| 策略行为 | keep/add/retype/remove 动作频率、动作熵、平均置信度、训练前后共有候选动作一致率 | 仅对共有候选计算稳定率，避免候选集合变化把动作变化误算为策略不稳定。 |
| 实际成本 | 候选搜索中位耗时、模型前向中位数/P95、tokens/s、参数字节数、采样的阶段 RSS、整进程高水位、设备内存快照/采样峰值 | psutil 每 2 ms 采样主机 RSS；阶段峰值可能漏掉更短的尖峰。MPS 在 warmup 后以 2 ms 采样 `current_allocated_memory` 与包含 allocator cache 的 `driver_allocated_memory`；采样值是稳态窗口峰值估计，不是精确硬件峰值。profiler CPU 分配总和、参数字节数和进程 RSS 都不能冒充 MPS 设备内存峰值。 |
| 正确性 | padding、跨 segment、跨 batch、未来信息隔离 | 使用扰动测试：改变隔离区域输入后，未受影响的有效节点输出保持一致。不能将正确性失败与任务分数做平均。 |

## 稳定身份与拓扑生命周期

`node_ids` 在每个独立 graph 内必须唯一；相同神经元池在 batch 的不同 graph 中可以复用同一组 ID。`NeuronGraphTopology.remap_node_ids` 按身份重排节点，丢弃已删除节点及其关联边，新加入节点保持断连，调用方随后可显式添加连接。因果图重排若会反转边方向则拒绝。跨进程身份注册表与持久化由调用方负责。

## 解释规则

- 首要结果是配对验证质量差值，并同时给出种子结果、参数量、时延和内存口径。
- 随机度数匹配图若与策略图相当，说明收益尚未证明来自学到的边身份。
- 有效秩下降或 Dirichlet energy 变化只触发进一步检查；过平滑文献指出单一平滑能量不足以涵盖真实条件。
- 梯度范数接近零表示所采样目标对这些输入节点不敏感，不足以单独断言图发生过挤压。
- 原始 Dirichlet energy 不应跨层直接比较，因为各层激活范数可能不同；同时查看相对能量、有效秩和验证任务表现。
- 多次训练的边集合 Jaccard 仅允许用于稳定 node ID 完全匹配、且候选暴露可比的拓扑。当前验证 runner 通过固定验证输入比较训练前后的共有候选动作，不将无稳定语义的 token 位置伪装成跨数据集身份。
- 单步 smoke 只验证度量管线可执行，不构成质量结论。当前实验仍是合成递归任务筛查。

## 开源实现复用记录

已克隆并检查 [GSLB](https://github.com/GSL-Benchmark/GSLB)（NeurIPS 2023）与 [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 源码。GSLB 评估框架支持效果、鲁棒性和复杂度维度，但其 Dirichlet 实现输入稠密邻接矩阵，并用矩阵元素数归一化；不能直接代表本项目有向稀疏边表的加权能量。PyG 的通用图工具不承载本项目的 segment 身份、动作策略和因果硬约束。两者都不适合作为本层运行依赖或直接复制度量实现。

图指标使用项目既有 PyTorch 边表与张量运算。为跨平台采样 RSS，实验开发依赖新增 BSD-3-Clause 的 [psutil](https://github.com/giampaolo/psutil)；只调用官方 `Process.memory_info().rss`，没有复制源码。源码版本：psutil `412d3e2d18734d3838bafc12f0795d54ee0a2b4f`（当前 lock 解析版本 7.2.2）。MPS memory counter 口径见 [PyTorch MPS API](https://github.com/pytorch/pytorch/blob/main/torch/mps/__init__.py)：current allocated 排除 allocator cache，driver allocated 包含 cache。GSLB `375cc1c20314360a927527f8d8528773fe1b0abc` 与 PyG `d1f2f8642387b6e0563d6aee227c39130746c738` 仅审查，没有移植不匹配实现。

## 研究依据

- [Rethinking Oversmoothing in GNNs: A Rank-Based Perspective](https://arxiv.org/abs/2502.04591)：有效秩为平滑能量提供互补诊断。
- [Understanding Oversquashing in GNNs through Effective Resistance](https://arxiv.org/abs/2302.06835)：从有效电阻研究长距离信息传输受限；当前有向层先采用任务敏感梯度采样，不把无向电阻直接套用到有向图。
- [GSLB: Graph Structure Learning Benchmark](https://proceedings.neurips.cc/paper_files/paper/2023/file/60bc87f3cf5257579435d92ec12c761b-Paper-Datasets_and_Benchmarks.pdf)：按效果、鲁棒性和复杂度共同评估结构学习。
- [Learning Discrete Structures for GNNs](https://proceedings.mlr.press/v97/franceschi19a.html)：结构学习应以验证目标检验学习到的图。
- [Revisiting the Necessity of Graph Learning and Common Graph Benchmarks](https://arxiv.org/abs/2412.06173)：需要无图控制，避免把可由特征模型解决的任务误归功于图。
