# 开源聚合组件适配记录

## 选择

采用 PyTorch Geometric 的 `torch_geometric.nn.aggr.SetTransformerAggregation`，适配器位于 `src/newera/neuron/cells/aggregation/set_transformer.py`。它将同一集合内所有消息做内容交互，再用 PMA 学习查询汇聚；支持可变消息数，不复制神经元权重。`ModelNeuron` 可显式接收该 adapter，并将全部聚合参数注册在单个模型下；默认加权求和路径保持为现有基线。PyG 源码版本 2.8.1，仓库提交 `6bc98104b5a3472176a2bec4187a65f9fc429213`。PyG 项目和包使用 MIT 许可。

上游源码通过临时 clone 阅读，路径 `/tmp/newera_pyg_source`；未复制上游源文件。适配只使用导入的聚合 API。安装由 `uv add 'torch-geometric>=2.8,<3'` 管理，当前 lock 解析到 2.8.1；安装增加了纯 Python 包及传递依赖，不引入 C++/CUDA 扩展。

具体上游实现：`torch_geometric/nn/aggr/set_transformer.py` 的 `SetTransformerAggregation.forward`，调用 `to_dense_batch`、Set Attention Blocks 和 PMA，输入需传 `index`/`ptr` 及 `dim_size`。NewEra 将单个 message set 编为一个排序好的 index segment。

官方 Set Transformer 仓库也已克隆检查（提交 `73432c640ac78140496d6738416c54d32c686d65`，MIT，`modules.py` / `models.py`）。其官方 MAB/ISAB/PMA 可复用，但 PyG 已提供与图消息汇聚契合的完整 API、mask 分组和更新的 PyTorch attention 实现，故不另外复制 MAB 堆栈。CLRS 不是聚合候选；其 BFS 数据候选及推迟理由记录在能力矩阵。

## 边界语义与兼容性

- **变长输入**：接受 `[items, channels]`；集合长度由当前张量本身确定，不设固定 top-k。
- **mask**：只选择 mask 为真的行，padding 不参与汇聚。
- **空输入**：上游对零元素集合会在 attention mask reshape 处抛 `RuntimeError`。NewEra adapter 显式定义空消息集合的汇聚结果为 `[1, channels]` 全零，并跳过上游调用。
- **重复接触**：重复行原样传给注意力；不会去重，重复项按多重集合参与计算。
- **排列**：非空集合在 CPU/MPS 上与输入排列无关（数值容差 `1e-5`）。
- **设备/梯度**：不捕获异常后转 CPU；输入、模块参数留在调用方设备；CPU 和 MPS 均通过前向、反向和有限梯度检查。
- **PyG 的 index 要求**：单集合适配时 `index` 全零，天然有序。多集合/图层整合由 NE-024 负责；当前 adapter 不构造重复的节点或神经元模块。
- **上游 warning**：PyG 2.8.1 在 PyTorch 2.15 导入时触发 `torch.jit.script` 弃用 `FutureWarning`。不影响当前路径，但应在升级依赖时复查。

## 测试和成本测量

`project/tests/other/neuron_model_evaluation/test_set_transformer_adapter.py` 在 CPU 与 MPS 覆盖上游等价、梯度、排列、重复执行确定性、空集合、mask padding、重复消息、输入校验，以及聚合器参数确由单个 `ModelNeuron` 持有；结果 13 passed（合并相同测试文件运行时计数）。运行环境：PyTorch 2.15.0、PyG 2.8.1、Apple MPS。

32 通道、4 heads、1 个 encoder/decoder block，固定单集合，热身后 20 次前向+反向的中位耗时如下；只作为当前机器上的微型组件量级参考，非图训练吞吐承诺：

| 消息数 | CPU (ms) | MPS (ms) | 输入张量大小 |
|---:|---:|---:|---:|
| 1 | 0.431 | 4.509 | 128 B |
| 16 | 0.520 | 3.235 | 2 KiB |
| 64 | 0.591 | 3.333 | 8 KiB |
| 256 | 1.292 | 3.563 | 32 KiB |

测量方法：5 次热身、20 次前向+反向，使用中位耗时；输入为固定宽度 32、batch 内只有一个集合。MPS 分配增量读取 `torch.mps.current_allocated_memory()`，包含输入和梯度；CPU 读取进程 RSS 增量，会受 Python/PyTorch allocator 与惰性初始化影响。

MPS 分配器观测的循环前后增量约为 3.5 KiB（16项）至 49 KiB（256项），相应输入+梯度张量本身为 4 KiB 至 64 KiB；该差值会受 MPS 缓存/热身影响。CPU 进程 RSS 包含 PyTorch 本身，按消息数的 RSS 增量噪声较大，因此不拿作组件峰值内存结论。后续性能比较应在 NE-024 中用整层、相同 batch 和相同消息特征基准测峰值内存。

## 结果与决定

依赖和 adapter 已落地，PyG Set Transformer 是 NE-024 的可选聚合候选。它目前尚未替换 `ModelNeuron` 的默认加权求和；是否成为默认路径要由 NE-024 的能力准确度、分组反例和整层 MPS 成本决定。本记录不代表其已通过能力任务阈值。
