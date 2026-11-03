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

Codegen 和 Trainer 使用同一份训练 YAML。用户只需要维护训练配置，并在
`model.pretrained_model_name_or_path` 中指定模型来源，例如：

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

### 1.3 准备训练 YAML

这份 YAML 同时用于原生训练、CLI 生成、Trainer 自动生成和 generated model 训练，不需要再准备单独的
Codegen 配置。YAML 至少要指定 Hugging Face 模型来源、TP/CP/EP/PP 等并行度，以及需要应用的模块或
forward 替换规则。建议复制最接近的 `recipes/train.yaml` 或 training demo 后修改，不要从空文件开始编写。

## 2. CLI 命令

使用标准 `scripts/train_lm.py` 启动训练时，global rank 0 会在独立的单进程 sidecar 中自动生成产物；
生成失败会输出原因，但不会中断原生训练。也可以不启动训练，单独生成：

```bash
python -m hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/qwen3_moe/train_parallel_online.yaml
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

这里指不能使用 `torchrun -m hyper_parallel.codegen generate` 直接启动 CLI，否则每个 rank 都会尝试写入同一个
bundle。正常使用 `torchrun ... scripts/train_lm.py` 启动训练不受影响：Trainer 入口只让 global rank 0 启动一个
单进程 Codegen sidecar，其余 rank 不执行生成。

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

### DeepSeek-V4.1 训练时报 Multicore 或 MegaMoE 动态库错误

这不是 Codegen 专属错误：原生模型和生成模型都会使用 DeepSeek-V4.1 的 Multicore/MegaMoE native 扩展。请确认编译与
训练使用同一套 Python、PyTorch、torch_npu 和 CANN 环境，并执行一次严格的干净构建：

```bash
./build.sh --multicore on --strict on --clean
source /usr/local/Ascend/cann/set_env.sh
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
```

如果日志出现 `undefined symbol`，或加载了与当前 Python ABI 不同的 `.so`（例如 cp311 环境加载 cp313 产物），应清理旧构建并
在当前训练环境重新编译；不要通过关闭 MegaMoE 或修改模型结构绕过。完整环境要求见[安装指南](../installation.md)。

### DeepSeek recipe 报 planner boundary 不存在

带 `params`、`in_src`、`in_dst`、`out_src` 或 `out_dst` 的契约必须匹配模块边界，例如
`model.layers.*.engram`；只有 method replacement 才匹配 `model.layers.*.engram.forward`。如果把参数契约写到
`.forward`，planner 会提示真实模块没有对应 boundary。建议从仓库内置 recipe 复制后修改，并保留这两类规则的层级区别。
