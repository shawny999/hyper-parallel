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
"""The generated-model launcher stays separate from modeling source files."""

from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("transformers")

# The optional-dependency gate must run before importing the training stack.
# pylint: disable=wrong-import-position
from scripts import train_generated


class FutureForCausalLM:
    """Generated model class used by launcher tests."""


def test_generated_module_loader_supports_relative_bundle_imports(tmp_path: Path) -> None:
    bundle = tmp_path / "generated"
    bundle.mkdir()
    bundle.joinpath("__init__.py").write_text("", encoding="utf-8")
    bundle.joinpath("configuration_future.py").write_text("VALUE = 7\n", encoding="utf-8")
    model_file = bundle / "modeling_Future_gen_npu.py"
    model_file.write_text(
        "from .configuration_future import VALUE\n\n"
        "class FutureForCausalLM:\n"
        "    marker = VALUE\n",
        encoding="utf-8",
    )

    module = train_generated._load_generated_module(model_file)  # pylint: disable=protected-access

    assert module.FutureForCausalLM.marker == 7


def test_generated_launcher_uses_standard_yaml_and_scoped_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events = []
    model_file = tmp_path / "modeling_Future_gen_npu.py"
    model_file.write_text("class FutureForCausalLM:\n    pass\n", encoding="utf-8")
    module = ModuleType("generated_future")
    module.__file__ = str(model_file)
    FutureForCausalLM.__module__ = module.__name__
    module.FutureForCausalLM = FutureForCausalLM
    config = SimpleNamespace(model=SimpleNamespace(_target_path="HyperAutoModelForCausalLM.from_config"))

    @contextmanager
    def overlay(model_class: type, architecture: str):
        events.append(("overlay", model_class, architecture))
        yield

    class FakeTrainer:
        def __init__(self, trainer_config: object) -> None:
            events.append(("init", trainer_config))

        def train(self) -> None:
            events.append(("train",))

    monkeypatch.setattr(
        train_generated,
        "load_codegen_meta",
        lambda _path: SimpleNamespace(model_class="FutureForCausalLM"),
    )
    monkeypatch.setattr(train_generated, "_load_generated_module", lambda _path: module)
    monkeypatch.setattr(
        train_generated,
        "load_training_config",
        lambda path, overrides: events.append(("config", path, overrides)) or config,
    )
    monkeypatch.setattr(train_generated, "generated_model_build", overlay)
    monkeypatch.setattr(train_generated, "TextTrainer", FakeTrainer)

    train_generated.main([
        "--model-file",
        str(model_file),
        "train.yaml",
        "--training.train_iters=1",
    ])

    assert events == [
        ("config", "train.yaml", ["--training.train_iters=1"]),
        ("overlay", FutureForCausalLM, "FutureForCausalLM"),
        ("init", config),
        ("train",),
    ]
