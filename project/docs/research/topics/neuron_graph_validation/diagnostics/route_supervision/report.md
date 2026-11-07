# 路由训练修复对照与确认

## 设计与验证集选型

同一 `single_link_retrieval` 训练/验证切分（训练 32、验证 64，seed 92001/92002），同一初始化 seed，5 个条件各 500 次更新；正式选择只看验证集。参数匹配 MLP 为 10,945 个参数，图模型为 11,102（差 1.4%）。sealed test 未参与选型。

- `production_control`：任务损失 + L1 边稀疏惩罚 + 最后 15% STE 硬前向训练。
- `task_only`：仅任务损失、软候选路由；无结构标签、稀疏惩罚或 STE。
- `sparsity_only`：任务损失 + L1 稀疏惩罚。
- `structure_supervised`：任务损失 + 仅训练时可见的类别平衡 oracle 动作交叉熵。
- `hard_calibrated`：任务损失，最后 30% 使用 STE 硬前向。

| 条件 | 验证 MSE | edge F1 | 空图/64 | 结论 |
|---|---:|---:|---:|---|
| production control | 0.1081 | 0.281 | 0 | 较 task-only 略差 |
| task-only | **0.1037** | 0.279 | 0 | 选为候选 |
| sparsity-only | 0.1239 | 0.266 | 1 | 稀疏惩罚无益 |
| structure-supervised | 0.1529 | 0.244 | 17 | 结构标签伤害任务质量 |
| hard-calibrated | 0.1207 | 0.278 | 0 | STE 校准无收益 |

验证集上 task-only 的 MSE 比训练均值控制低 23%、比参数匹配 MLP 低 30%。删边干预平均使预测变化 0.103、MSE 增加 0.0145，约 51% 的删除使误差恶化。故保留 task-only 作为训练候选，但不宣称拓扑策略已学会。

## 六 seed 封存确认

frozen config `confirmation_config.json` 使用六个新 seed（7601、7611、7621、7631、7641、7651），每 seed 训练集 256、验证集 128、sealed test 128，分别训练 1,200 次更新。阈值只用各自验证集的预先冻结候选集合选择；每个 sealed test 在冻结后评估一次。相同训练预算/样本用于图模型、参数匹配 MLP 与 Linear MAC 匹配 MLP。运行于 MPS，Torch 2.15.0。

| 模型/控制 | 平均 sealed-test MSE | 相对图模型改善 | paired 95% CI（相对 MSE 差） |
|---|---:|---:|---:|
| 零预测控制 | 0.1230 | -32.4% | [-40.5%, -24.3%] |
| 训练均值控制 | 0.1229 | -32.3% | [-40.2%, -24.4%] |
| 参数匹配 MLP | 0.1594 | -47.3% | [-56.1%, -38.5%] |
| Linear MAC 匹配 MLP | 0.1438 | -42.2% | [-48.0%, -36.4%] |

图模型平均 MSE 为 **0.0832**。相对四类控制的 paired 相对误差改善均超过预设 20%，95% CI 上界均低于 0；因此该小型合成检索任务通过“学到任务且优于控制”的质量判据。此结论只适用于当前任务、样本规模和实现预算，不外推为通用能力。

## 结构动作与 hard rollout

- 平均 edge F1：0.337；身份置换后边 Jaccard：1.000；置换 MSE drift：0.000；空图比例：0.000。
- 结构动作 recall：remove 0.000、retype 0.000、set-strength 0.995；add 在该数据分布无正例，故为 N/A。平均 strength MAE：0.142。结构判据未通过。
- 每 seed 两条 8 步 hard rollout（共 96 transitions）：平均 step MSE 0.130、平均 edge F1 0.132、平均硬边数 2.0、平均 identity-edge churn 1.417。策略几乎只学会 set-strength，hard 路由的目标边识别与跨步稳定性仍不足。

## 决策与交接

1. 保留 task-only 软候选训练作为当前任务损失训练基线：它稳定优于常数与匹配容量控制。
2. 不把当前策略作为已解决的动态拓扑/硬路由方案：edge F1、remove/retype recall 和 hard rollout 均不达标。NE-026 继续研究结构操作与持续通信；NE-024 的聚合器能力验证仍独立推进。
3. 该任务有可辨识的目标边，因此 edge/action 指标作为结构判据；对结构等价任务仍需用任务效用和干预评估，不强求唯一拓扑。
4. 原结果、六 seed 确认、checkpoint、阈值选择、逐 seed 操作和成本数据见同目录 `results.json`、`confirmation_results.json` 与 `checkpoints/`。

完整源代码指纹、阈值、逐 seed 结果、pool generalization、MAC 与运行限制见 JSON。
