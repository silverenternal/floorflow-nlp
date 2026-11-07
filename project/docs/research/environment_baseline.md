# NE-001 本地环境与测试基线

日期：2026-11-06

## 环境

- 机器：Apple Silicon，`arm64`，macOS 27.2
- Python：3.13.15；环境由项目根目录 `.venv` 管理
- uv：0.12.23
- PyTorch：2.15.0；MPS 已构建且可用，MPS 张量运算成功
- NumPy：2.5.4；pytest：9.2.0；tree-sitter：0.26.0
- 完整开发与可选预训练依赖：`uv sync --extra dev --extra pretrained`
- 项目锁文件只针对 macOS arm64；安装前可检查 `uv lock --check`

## 可复核命令和结果

| 命令 | 结果 |
| --- | --- |
| `uv run --no-sync python -c 'import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'` | `True True`；另在 MPS 上执行乘法，得到 `[3.0, 6.0]` |
| `uv run --no-sync pytest -q src/newera/gates/test_gates.py project/tests/other/general/cache_dry_ensemble/test_nlp_neuron_module.py` | 12 passed |
| NE-003 至 NE-006 协议、基线、候选与组合测试 | 聚焦测试通过；见 `project/tests/models_training/contract_scopes_stage/` |
| CPU neuron smoke：`NeuronModule(16, 4)`，输入形状 `(2, 3, 16)`，对均方损失反向传播 | 输出形状正确、损失有限、16/16 个参数获得梯度；489 个参数 |
| `uv run --no-sync pytest -q`（当前全量测试） | 551 passed、21 skipped、0 failed |

21 项跳过都有明确 pytest 原因：缺失的旧 `.data` tokenizer/parser/corpus 文件、跨机器不稳定的 parser 原生库 SHA、已移除的根目录 `nlptodo.json`/schema/Rust 生成 crate，或依赖这些冻结产物的旧 NLP 实验报告。相关测试保留为可见跳过项；它们不表示功能已验证。本轮清除了只检查仓库尺寸/目录、旧路线图、内部文件拆分、静态覆盖表、历史冻结状态和审计/冻结报告本身的测试；相应目录限制仍由统一门禁直接检查。训练/验证隔离、语料来源、运行预检、状态恢复和模型行为测试保留。其他定位到的重组问题已更新导入路径、Python 子进程命令和测试替身。根目录增加标准 `uv.lock` 后，已将它记为唯一工具元数据例外，不计入五项目录门禁；根目录协议和门禁同步更新。

## 边界

这份基线记录的是当前 PyTorch 环境、聚焦测试和旧 NLP 套件的显式跳过项；新协议和两个最小神经元实现另有独立 CPU 测试。MPS 仅验证了一个张量运算，尚未验证 MPS 上的神经元训练或模型端到端运行。
