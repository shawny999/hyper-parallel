# Codegen 生成模型训练对照脚本

这些脚本用于开发自测：通过同一训练 YAML 和同一组覆盖参数，比较原生 Trainer 与 Codegen 生成模型的实际训练计算。它们不是生成模型训练的产品入口，不改变原生 Trainer API。

本目录只包含验证脚本及使用说明。运行时生成的模型、初始状态、日志和报告应放在仓库外或 `.tmp/` 下，不应提交。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `compare_bundle_training.py` | 主入口：生成 bundle，执行原生参考、原生重复、生成模型三个阶段，并比较结果 |
| `run_forward_range.py` | Qwen3-MoE TP2/CP2/EP2 专项入口：增加完整 forward 覆盖及 TP/CP/EP 数值负对照 |
| `train_bundle.py` | torchrun 工作进程，解析统一 YAML，选择原生或生成模型 |
| `artifact_training.py` | 测试专用产物加载器，核对构造结构，恢复布局所需声明，并防止回退到原生 forward |
| `training_evidence.py` | 保存、回放初始状态和随机状态，采集输入、梯度和参数摘要 |
| `forward_range.py` | 记录产物 forward、并行边界及模块输出的实际执行情况 |
| `generation_worker.py` | 调用公共 Codegen CLI，保存生成快照和独立证据 |
| `native_worker.py` | 执行标准训练脚本，记录各 rank 的训练进度、sidecar 和退出情况 |
| `common.py` | 子进程管理、空卡轮询、文件哈希和数值比较工具 |
| `test_artifact_training.py` | 产物加载、状态回放和对照机制的 CPU 自检 |
| `test_forward_range.py` | forward 来源、覆盖统计及数值扰动机制的 CPU 自检 |

## 环境与适用范围

- 分布式训练入口面向 Linux Ascend 环境，需要 PyTorch、torch_npu、Transformers 和 HyperParallel 对应依赖；`npu-smi` 应位于 PATH。
- 从实际被测源码目录执行 `python -m pip install -e .`，并检查 `hyper_parallel.__file__` 指向该目录。
- 首先使用目标 YAML 和有效覆盖参数确认原生训练可运行。资源目录、tokenizer 和数据集由使用者准备，脚本不下载模型权重。
- 使用 eager、PP1，固定 seed，并开启 `training.enable_full_determinism`；关闭数据加载 worker 和后台预取。
- 主对照入口默认运行 8 个 rank、20 步，默认 `atol=rtol=0`。维度裁剪通过模型配置或 YAML 覆盖参数统一传给所有阶段。
- 完整数值负对照入口针对 Qwen3-MoE TP2/CP2/EP2 编写，不能直接视为任意模型、任意拓扑的通用负对照。
- 外部加载器依赖当前 family 和构建私有接口；PP、多样化 method companion 状态，以及复合模型的 experts 配置需单独适配。脚本拒绝不支持的路径，不静默改用原生模型取得一致结果。

## 一、环境准备

在 129 等 Ascend 主机执行，根据实际安装路径加载 CANN 环境，然后设置：

```bash
cd /path/to/hyper-parallel
python -m pip install -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'

SOURCE="$PWD"
PACK="$SOURCE/scripts/codegen_validation"
PYTHON=python
ASSETS=/path/to/cropped-qwen3-moe-assets
DATA=/path/to/training-data.jsonl
OUTPUT_ROOT=/path/to/codegen-test-output
mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$PACK${PYTHONPATH:+:$PYTHONPATH}"
```

`ASSETS` 应是与目标 YAML 匹配的模型配置及 tokenizer 目录。短步测试建议使用可在目标拓扑上运行的裁剪配置，例如让 head 数量、专家数量等满足 TP/CP/EP 的整除约束。`DATA` 使用 YAML 所要求的数据格式。

## 二、同 YAML 的原生与生成模型对照

使用全新输出目录执行主入口；主入口是单进程调度器，自行启动 torchrun 子进程，不要再用 torchrun 启动它。

```bash
"$PYTHON" "$PACK/compare_bundle_training.py" \
  --source "$SOURCE" \
  --config "$SOURCE/examples/training_demo/qwen3_moe/train_parallel_online.yaml" \
  --entry scripts/train_lm.py \
  --output "$OUTPUT_ROOT/comparison-r1" \
  --nproc-per-node 8 --steps 20 --atol 0 --rtol 0 \
  --model.config_path="$ASSETS" \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path="$ASSETS" \
  --dataset.data_path="$DATA" \
  --training.seed=42 --training.enable_full_determinism=true \
  --training.global_batch_size=8 \
  --compile.enabled=false --checkpoint.save_ckpt=false \
  --dataloader.num_workers=0 --dataloader.use_background_prefetcher=false \
  --accelerator.tp_size=2 --accelerator.cp_size=2 \
  --accelerator.ep_size=2 --accelerator.pp_size=1 \
  --accelerator.sequence_parallel=true --accelerator.loss_parallel=true \
  --fsdp_config.dp_shard_size=4 --fsdp_config.edp_shard_size=4
```

标准 LM 默认入口为 `scripts/train_lm.py`。主入口可以通过 `--entry` 指定 VLM 标准入口，但需先满足上面的加载器适配条件。

执行顺序：

1. 调用公共 CLI，生成本次待测 bundle。
2. 原生训练记录初始参数、持久 buffer 和随机状态。
3. 原生重复运行回放同一初始状态，检查确定性；通过后才进入生成模型阶段。
4. 从生成源码构造模型，执行同一训练 YAML 和覆盖参数；恢复必要构造声明，不用原生 forward 覆盖待测计算。
5. 逐 rank 比较初始 schema、参数哈希、输入、每步 loss、最终参数，并核验产物执行来源和阶段覆盖。

`common.py` 会在每个 `train_bundle.py` torchrun 启动前查询本机 NPU 占用，每 60 秒检查一次，直至所有卡空闲。该逻辑应运行在目标训练服务器上，不会停止其他任务，也不提供排他式设备预留。

## 三、forward 覆盖与数值负对照

此入口需要一个由使用者填写的资源配置 JSON。下面是配置格式示例，不包含验证结果：

```bash
cat > "$OUTPUT_ROOT/resources.json" <<EOF
{
  "optional_pythonpath": "",
  "models": {
    "qwen3_moe": {
      "yaml": "examples/training_demo/qwen3_moe/train_parallel_online.yaml",
      "entry": "scripts/train_lm.py",
      "overrides": ["--model.config_path=$ASSETS"],
      "training_overrides": [
        "--dataset.model_assets.tokenizer.pretrained_model_name_or_path=$ASSETS",
        "--dataset.data_path=$DATA",
        "--training.seed=42",
        "--training.enable_full_determinism=true",
        "--training.global_batch_size=8",
        "--compile.enabled=false",
        "--checkpoint.save_ckpt=false",
        "--dataloader.num_workers=0",
        "--dataloader.use_background_prefetcher=false"
      ]
    }
  }
}
EOF

"$PYTHON" "$PACK/run_forward_range.py" \
  --source "$SOURCE" \
  --manifest "$OUTPUT_ROOT/resources.json" \
  --output "$OUTPUT_ROOT/forward-range-r1"
```

该入口固定使用 8 rank、20 步和 TP2/CP2/EP2/PP1。在正向对照后，分别创建 bundle 的独立副本，对 TP、CP、EP 计算加入数值扰动，并确认实际训练中出现目标模块输出、梯度和 loss 差异。原始 bundle 保持不变。

## 四、CPU 自检

```bash
PYTHONPATH="$SOURCE:$PACK${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m pytest -q \
  "$PACK/test_artifact_training.py" \
  "$PACK/test_forward_range.py"
```

自检不启动多卡训练。分布式依赖在实际训练工作进程中加载，安装相应测试依赖后执行。

## 输出与判定

| 运行时输出 | 用途 |
| --- | --- |
| `bundle/` | 本次生成的待测源码 |
| `initial-state/` | 测试用初始权重、buffer 及随机状态 |
| `native/`、`native-repeat/`、`generated/` | 各阶段逐 rank 训练和执行来源记录 |
| `native-repeat-gate.json` | 原生重复确定性门槛 |
| `comparison-report.json` | 原生与生成模型对照判定 |
| `positive/`、`negative-{tp,cp,ep}/` | 完整数值验证的正向及负对照运行目录 |
| `report.json` | forward 覆盖与数值负对照的汇总判定 |
| `*.log`、`*.availability.json` | 子进程日志和空卡轮询记录 |

只有全部 rank 正常退出、完成指定 optimizer step、通过数值和执行来源断言，才算对照通过。每次运行更换输出目录；缺 rank、提前退出、构造失败或原生重复不确定，均不能作为生成模型数值通过。

不要将本目录运行产生的任何产物、报告、日志、权重或数据集加入 Git。
