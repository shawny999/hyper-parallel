# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Training entry point for a generated modeling bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence

from hyper_parallel.codegen.meta import load_codegen_meta
from hyper_parallel.models._generated.build_overlay import generated_model_build
from hyper_parallel.trainer.config.parser import load_training_config
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer

logger = logging.getLogger(__name__)


def _load_generated_module(model_file: str | Path) -> ModuleType:
    """Load one generated modeling file with its bundle as a Python package."""
    path = Path(model_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"generated model file does not exist: {path}")
    init_path = path.parent / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(
            f"generated bundle is missing __init__.py beside {path.name}"
        )
    digest = hashlib.sha256(str(path.parent).encode("utf-8")).hexdigest()[:12]
    package_name = f"hyper_parallel_generated_{digest}"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(path.parent)],
    )
    if package_spec is None or package_spec.loader is None:
        raise ImportError(f"cannot create package spec for generated bundle {path.parent}")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)

    module_name = f"{package_name}.{path.stem}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot create module spec for generated model {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def _resolve_model_class(module: ModuleType, architecture: str, requested: str | None) -> type:
    """Resolve the generated model class without a model-family allowlist."""
    class_name = requested or architecture
    model_class = getattr(module, class_name, None)
    if isinstance(model_class, type):
        return model_class
    suffixes = (
        "ForConditionalGeneration",
        "ForImageTextToText",
        "ForSequenceClassification",
        "ForCausalLM",
    )
    suffix = next((value for value in suffixes if architecture.endswith(value)), None)
    candidates = [
        value
        for name, value in vars(module).items()
        if (
            isinstance(value, type)
            and value.__module__ == module.__name__
            and suffix is not None
            and name.endswith(suffix)
        )
    ]
    if requested is None and len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"generated model class {class_name!r} was not found in {module.__file__}; "
        f"candidates={[candidate.__name__ for candidate in candidates]}"
    )


def _trainer_type(config: object) -> type:
    """Select the same text or VLM Trainer family used by standard entries."""
    target_path = getattr(getattr(config, "model", None), "_target_path", "")
    if "ImageTextToText" in target_path:
        return VLMTrainer
    return TextTrainer


def main(argv: Sequence[str] | None = None) -> None:
    """Load a generated model and execute training from the standard YAML."""
    parser = argparse.ArgumentParser(description="Train a generated HyperParallel model")
    parser.add_argument("--model-file", required=True, help="generated modeling_*_gen_npu.py")
    parser.add_argument("--model-class", help="optional generated class name override")
    parser.add_argument("config_file", help="standard HyperParallel training YAML")
    args, overrides = parser.parse_known_args(argv)

    model_path = Path(args.model_file).resolve()
    meta = load_codegen_meta(str(model_path.parent / "codegen_meta.json"))
    if meta is None:
        raise RuntimeError(
            f"generated bundle metadata is missing or invalid: {model_path.parent / 'codegen_meta.json'}"
        )
    architecture = meta.model_class
    if not architecture:
        raise RuntimeError("generated bundle metadata does not declare model_class")

    module = _load_generated_module(model_path)
    model_class = _resolve_model_class(module, architecture, args.model_class)
    config = load_training_config(args.config_file, overrides)
    trainer_class = _trainer_type(config)
    with generated_model_build(model_class, architecture):
        trainer = trainer_class(config)
    logger.info(
        "Training with generated model class %s from %s",
        model_class.__name__,
        model_path,
    )
    trainer.train()


if __name__ == "__main__":
    main()
