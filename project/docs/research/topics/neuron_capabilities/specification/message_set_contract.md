# 神经元入边消息集合契约

`ModelNeuron.step` 的 `incoming_messages` 是 `[batch, contacts, message_size]`。每一行代表一个消息接触，集合次序没有语义，重复行保留为重复接触；集合长度由当前输入确定，不使用固定 top-k。

- `incoming_strengths` 和 `incoming_edge_types` 按同一 `[batch, contacts]` 位置对齐。
- 影响计算的 contact/group 身份必须编码在消息 payload 中；关系类别由 edge type 表示。任务 adapter 负责把可观察的组标识编码成 payload 特征，不得用 oracle 标签或隐藏目标。
- 空集合是合法输入，聚合值定义为零；有效消息全排列时聚合语义不变。
- 当前图拓扑不允许同一有向节点对重复建边。分组任务中的消息行表达一对多的输入接触集合，无需把任务 group id 混为拓扑 edge ID；只有出现同一 source/target 需要两条独立拓扑关系的反例，才扩展拓扑边身份。
- 输入带有 group/role/value 的绑定任务用于反例检验：保持各 role 的值和 group-id 边际分布不变，只改变 role 与 group 的配对关系，oracle 输出随之变化。全局线性投影后求和无法区分这类输入；集合内交互算子必须保留该信息。

聚合器的所有可训练参数仍注册在唯一 `ModelNeuron` 下，并在所有 neuron 位置共享。状态和稳定身份仍由调用方管理，group ID 只在单次消息集合内解释。
