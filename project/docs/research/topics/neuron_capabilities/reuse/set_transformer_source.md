# Set Transformer 上游复用记录

- 上游：Juho Lee 等人的作者官方 PyTorch 实现，https://github.com/juho-lee/set_transformer
- License：MIT；版权声明及完整许可文本保存在 `src/newera/neuron/cells/aggregation/upstream_licenses/LICENSE.set_transformer.txt`。
- 本地 clone：`.data/upstream/set_transformer/`（本地忽略，不提交）。
- 固定源码 commit：`73432c640ac78140496d6738416c54d32c686d65`。
- 采用范围：仅在研究分支中复制 MAB、ISAB、PMA 的核心张量结构；只复制所需模块。
- 适配范围：MAB 增加键 mask，支持不等长 padded batch；ISAB 清零 padding query 输出；PMA 使用有效消息 mask；试验 adapter 对空集合返回零向量，并用稳定输入输出形状封装；其训练/验证精度与成本未胜出，运行代码已移除。
- 不变式：不排序消息、不去重、不截断；固定 inducing point 数只约束内部摘要槽位，所有输入消息仍进入每个 ISAB 的 cross-attention。
