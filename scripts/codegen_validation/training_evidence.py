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
"""Fixed initial state, RNG and actual input/forward evidence for local A/B training."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import inspect
import os
from pathlib import Path
import random
import sys
from types import TracebackType
from typing import Any, Optional
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from common import code_fingerprint, declared_stages, dump

from hyper_parallel.trainer.base import BaseTrainer


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Inspect local shards without a collective or serializing a DTensor runtime object."""
    for _ in range(3):
        convert = getattr(value, "to_local", None)
        if not callable(convert):
            break
        value = convert()
    if value.is_meta:
        raise ValueError("The model has not been materialized")
    return value.detach().cpu().contiguous()


def tensor_digest(value: torch.Tensor) -> dict:
    """Fingerprint the complete local value, including BF16 and scalar tensors."""
    data = local_tensor(value)
    return {"shape": list(data.shape), "dtype": str(data.dtype),
            "sha256": hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def rng_state() -> dict:
    """Capture all RNGs that the already-initialized training worker can use."""
    numpy_state = np.random.get_state()
    result = {"python": random.getstate(), "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
              "torch": torch.get_rng_state(), "devices": {}}
    for backend in ("cuda", "npu"):
        device = getattr(torch, backend, None)
        initialized = getattr(device, "is_initialized", lambda: False)
        if device is not None and initialized():
            result["devices"][backend] = device.get_rng_state()
    return result


def restore_rng(state: dict) -> None:
    """Restore random streams after model construction, before data and optimizer setup."""
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"])
    for name, value in state["devices"].items():
        getattr(torch, name).set_rng_state(value)


def input_digest(value: Any) -> Any:
    """Fingerprint actual post-preforward model inputs, failing on opaque unsupported values."""
    if isinstance(value, torch.Tensor):
        return tensor_digest(value)
    if isinstance(value, dict):
        return {str(key): input_digest(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [input_digest(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported model input fingerprint type: {type(value).__name__}")


def model_state(model: nn.Module) -> tuple[dict, dict[str, torch.Tensor]]:
    """Keep alias/layout metadata and canonical local parameters/buffers outside the bundle."""
    schema, tensors, identities = {}, {}, {}
    for kind, named in (("parameter", model.named_parameters(remove_duplicate=False)),
                        ("buffer", model.named_buffers(remove_duplicate=False))):
        for name, value in named:
            key = kind + ":" + name
            canonical = identities.setdefault(id(value), key)
            owner_name, _, leaf = name.rpartition(".")
            owner = model.get_submodule(owner_name) if owner_name else model
            persistent = kind == "parameter" or leaf not in owner._non_persistent_buffers_set
            local = local_tensor(value)
            schema[key] = {"canonical": canonical, "shape": list(value.shape), "local_shape": list(local.shape),
                           "dtype": str(value.dtype), "requires_grad": value.requires_grad, "persistent": persistent,
                           "placements": [str(item) for item in getattr(value, "placements", ())]}
            if key == canonical:
                tensors[key] = local.clone()
    return schema, tensors


class TrainingEvidence:
    """Observe native Trainer execution without replacing its loop, loss or optimizer."""

    def __init__(self, directory: Path, initial: Path, action: str, generated: Optional[type] = None) -> None:
        """Choose explicit record/replay mode and a test-owned initial-state directory."""
        self.directory, self.initial, self.action, self.generated = directory, initial, action, generated
        self.rank = int(os.environ.get("RANK", "0"))
        self.report = {"rank": self.rank, "inputs": [], "artifact_forward_calls": {},
                       "artifact_stage_expectations": {}, "artifact_stage_calls": {}, "native_stage_fallback_calls": {},
                       "initial_state_policy": "native fixed weights; constructor RNG ordering is not the loss oracle"}
        self.stack = ExitStack()
        self.model = None
        self.profile_before = None
        self.profile_enabled = False
        self.stage_owners = {}

    def __enter__(self) -> "TrainingEvidence":
        """Install scoped observers; model-build state replay occurs before optimizer construction."""
        original_build = BaseTrainer._build_model
        original_preforward = BaseTrainer.preforward

        def build(trainer):
            original_build(trainer)
            self.model = trainer.model
            if self.generated is not None:
                if not isinstance(self.model, self.generated):
                    raise RuntimeError("Trainer did not build the generated entry class")
                self.collect_stage_owners()
            schema, tensors = model_state(self.model)
            path = self.initial / f"rank-{self.rank}.pt"
            world = int(os.environ.get("WORLD_SIZE", "1"))
            if self.action == "record":
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError(f"Refusing to reuse an old initial state: {path}")
                torch.save({"world_size": world, "schema": schema, "tensors": tensors, "rng": rng_state()}, path)
            else:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["world_size"] != world or saved["schema"] != schema:
                    raise ValueError("Generated/native initial parameter, alias or shard schemas differ")
                for key, entry in schema.items():
                    if not entry["persistent"] and key == entry["canonical"]:
                        if tensor_digest(tensors[key]) != tensor_digest(saved["tensors"][key]):
                            raise ValueError(f"Derived nonpersistent buffer differs before training: {key}")
                with torch.no_grad():
                    named_state = (("parameter", self.model.named_parameters()), ("buffer", self.model.named_buffers()))
                    for kind, named in named_state:
                        for name, value in named:
                            key = kind + ":" + name
                            entry = schema[key]
                            target = value.to_local() if callable(getattr(value, "to_local", None)) else value
                            target.copy_(saved["tensors"][entry["canonical"]].to(device=target.device))
                restore_rng(saved["rng"])
            actual_schema, actual = model_state(self.model)
            self.report["initial_schema"] = actual_schema
            self.report["initial_hashes"] = {key: tensor_digest(value) for key, value in actual.items()}
            self._flush()

        def preforward(trainer, batch):
            result = original_preforward(trainer, batch)
            rng = rng_state()
            rng["torch"] = tensor_digest(rng["torch"])
            rng["devices"] = {key: tensor_digest(value) for key, value in rng["devices"].items()}
            self.report["inputs"].append({"call": len(self.report["inputs"]) + 1,
                                          "value": input_digest(result), "rng": input_digest(rng)})
            if self.generated is not None and not self.profile_enabled:
                self.profile_before = sys.getprofile()
                if self.profile_before is not None:
                    raise RuntimeError("Disable other Python profilers during artifact comparison")
                self.profile_enabled = True
                sys.setprofile(self._profile)
            return result

        self.stack.enter_context(patch.object(BaseTrainer, "_build_model", build))
        self.stack.enter_context(patch.object(BaseTrainer, "preforward", preforward))
        return self

    def collect_stage_owners(self) -> None:
        """Require selected parallel methods to belong to the actual generated source."""
        module = sys.modules[self.generated.__module__]
        root = Path(module.__file__).resolve().parent
        for fqn, child in self.model.named_modules(remove_duplicate=False):
            declaration = getattr(type(child), "parallel_module_spec", None)
            if not callable(declaration):
                continue
            owner = next(base for base in type(child).__mro__ if "parallel_module_spec" in vars(base))
            for name in sorted(declared_stages(inspect.getsource(declaration))):
                function = inspect.unwrap(getattr(child, name))
                code = getattr(getattr(function, "__func__", function), "__code__", None)
                if code is None:
                    raise ValueError(f"Parallel stage has no executable code: {fqn}.{name}")
                owned = Path(code.co_filename).resolve().parent == root
                if not owned:
                    native_owner = next(base for base in type(child).__mro__ if name in vars(base))
                    native = inspect.unwrap(getattr(native_owner, name))
                    if not native_owner.__module__.startswith('hyper_parallel.components.modules.'):
                        raise ValueError(f'Unexpected native stage owner: {fqn}.{name}')
                    if getattr(native, '__code__', None) is not code:
                        raise ValueError(f'Native stage identity differs: {fqn}.{name}')
                key = f"{fqn}.{name}"
                self.report["artifact_stage_expectations"][key] = {
                    "class": owner.__name__, "method": name, "code_sha256": code_fingerprint(code),
                    "source_form": 'artifact' if owned else 'native_import', 'source_file': code.co_filename}
                self.stage_owners.setdefault((id(child), name), []).append(key)

    def _profile(self, frame, event, _argument):
        if event != "call":
            return
        keys = self.stage_owners.get((id(frame.f_locals.get("self")), frame.f_code.co_name), ())
        if not keys and frame.f_code.co_name not in ("forward", "_forward_impl"):
            return
        module = sys.modules.get(self.generated.__module__)
        owned = Path(frame.f_code.co_filename).resolve().parent == Path(module.__file__).resolve().parent
        if owned and frame.f_code.co_name in ("forward", "_forward_impl"):
            name = f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_firstlineno}:{frame.f_code.co_name}"
            counts = self.report["artifact_forward_calls"]
            counts[name] = counts.get(name, 0) + 1
        for key in keys:
            expected = self.report["artifact_stage_expectations"][key]["code_sha256"]
            if not owned and frame.f_code.co_name == 'forward':
                if self._record_native_delegate(frame, key, expected):
                    continue
            entry = self.report['artifact_stage_expectations'][key]
            correct_owner = owned if entry.get('source_form', 'artifact') == 'artifact' else (
                frame.f_code.co_filename == entry['source_file'])
            matches = correct_owner and code_fingerprint(frame.f_code) == expected
            counts = self.report["artifact_stage_calls"] if matches else self.report["native_stage_fallback_calls"]
            counts[key] = counts.get(key, 0) + 1

    def _record_native_delegate(self, frame, key, expected):
        """Accept an inherited forward only when the artifact body is its caller."""
        child = frame.f_locals.get('self')
        native_codes = {getattr(getattr(owner, 'forward', None), '__code__', None)
                        for owner in type(child).__mro__[1:]}
        ancestor = frame.f_back
        while ancestor is not None:
            if (ancestor.f_locals.get('self') is child
                    and code_fingerprint(ancestor.f_code) == expected
                    and frame.f_code in native_codes):
                delegates = self.report.setdefault('native_delegate_calls', {})
                delegates[key] = delegates.get(key, 0) + 1
                break
            ancestor = ancestor.f_back
        return ancestor is not None

    def _flush(self):
        dump(self.directory / f"comparison-rank-{self.rank}.json", self.report)

    def __exit__(self, exc_type: Optional[type], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> bool:
        """Restore observers and keep final parameter evidence only after a successful run."""
        if self.profile_enabled:
            sys.setprofile(self.profile_before)
        try:
            if self.model is not None and exc_type is None:
                _, tensors = model_state(self.model)
                self.report["final_hashes"] = {key: tensor_digest(value) for key, value in tensors.items()}
            self._flush()
        finally:
            self.stack.close()
        return False
