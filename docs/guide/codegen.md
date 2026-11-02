# Codegen 使用指南

Codegen 把训练 YAML 中配置的模型替换和并行策略的patch写入模型源码，并同时生成 diff，方便直接查看真实训练的模型代码。

## 1. 环境准备

### 1.1 安装当前代码

在 HyperParallel 仓库根目录执行：

```bash
pip install -e .
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
```

第二条命令应输出当前仓库中的 `hyper_parallel/__init__.py`，而不是其他工作区或旧安装目录。

### 1.2 在 YAML 中指定 Hugging Face 模型

用户只需要维护 Codegen YAML，在 YAML 的 `model.pretrained_model_name_or_path` 中指定模型来源即可，例如：

```yaml
model:
  pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
```

该字段可以填写：

- Hugging Face Hub 上的模型名称，例如 `Qwen/Qwen3-30B-A3B`；
- 已下载到本机的 Hugging Face 模型目录，例如 `/models/Qwen3-30B-A3B`。

Codegen 会通过这个地址读取 Hugging Face 模型自带的结构配置和模型源码。模型目录中的 `config.json` 是Hugging Face 模型资源的一部分，不属于 HyperParallel 配置，也不需要用户手工编写。

模型源码分为两种情况：

1. **Transformers 已内置该模型**：安装支持该模型的 Transformers 版本，然后在 YAML 中填写 Hub 名称或本地模型目录。
2. **模型依赖自定义 remote code**：本地模型目录要包含模型资源所引用的
   `modeling_*.py`、`configuration_*.py` 及其相对导入文件；YAML 的 `model` 中需要设置
   `trust_remote_code: true`。

生成时不需要准备模型权重、tokenizer、训练数据、checkpoint 或 NPU/GPU 运行环境。

### 1.3 准备 YAML

YAML 至少要指定 Hugging Face 模型来源、TP/CP/EP/PP 等并行度，以及需要应用的模块或 forward 替换规则。建议复制最接近的内置 recipe 后修改，不要从空文件开始编写。

## 2. CLI 命令

Codegen的产物会随着训练启动自动生成，也可以单独生成：

```bash
python -m hyper_parallel.codegen --verbose generate \
  --config hyper_parallel/models/qwen3_moe/recipes/qwen3_moe_codegen.yaml
```

## 3. Codegen Bundle

```text
generated/
├── __init__.py
├── codegen_meta.json
├── modeling_<Model>_gen_npu.py
└── modeling_<Model>_gen_npu.py.diff
```

- `modeling_*_gen_npu.py`：已经写入模块替换和条件 forward 的完整模型源码。
- `modeling_*_gen_npu.py.diff`：生成源码相对于原始 modeling 源码的差异，适合代码审查。
- `codegen_meta.json`：记录模型来源、YAML 签名、边界模块和文件校验信息，供 `check` 使用。
- `__init__.py`：使生成目录可以作为 Python 包加载。

如果官方模型本身由多个 modeling 文件组成，bundle 也会保留多个文件及对应 diff。例如 Kimi K3 的顶层模型和语言模型位于不同源码文件中，它们共同组成一个 bundle，并不代表生成了两个独立模型。

## 4. 常见问题

### 生成是否需要 NPU/GPU？

不需要。Codegen 使用 CPU/meta 模型读取结构，不执行训练计算，也不建立 HCCL/NCCL 通信。

### 为什么不能用 `torchrun` 生成？

生成只需要一个 Python 进程。多进程会同时写入同一个 bundle，因此 CLI 会要求改用单进程命令。

### 是否需要下载完整模型？

不需要。YAML 指向 Hugging Face Hub 时，Codegen 会读取模型结构和源码；指向本地模型目录时，该目录只需能够被
Transformers 正常识别。自定义 remote code 模型还需要对应的 Python 源码。两种情况都不会加载模型权重和训练数据。

### `check` 提示找不到 `codegen_meta.json`

先确认 `generate` 成功，再确认 `check` 使用了相同 YAML。已注册模型的产物默认位于
`hyper_parallel/models/<model_name>/generated/`。

### `check` 提示 hash drift

说明生成文件被修改、缺失，或目录中混入了旧产物。请备份并移走整个 `generated/` 目录后重新运行 `generate`，
不要只删除其中一个文件。

### 为什么一个 bundle 有多个 modeling 文件？

部分官方模型会把顶层模型和子模型拆在不同源码文件中。Codegen 保留这种依赖关系，并为每个被修改的 modeling 文件
生成对应 diff；它们仍属于同一个 bundle。
