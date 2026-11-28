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
"""Test-only loading of the actual generated model through the unchanged Trainer build."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Iterator
from unittest.mock import patch

import torch
from torch import nn

from common import code_fingerprint
from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.models._generated.build_overlay import generated_model_build
from hyper_parallel.models._transformers import model_builder
from hyper_parallel.models.module_build import construct_family_model


def _entry_class(module: ModuleType, architecture: str) -> type:
    classes = {id(value): value for value in vars(module).values() if inspect.isclass(value)
               and value.__name__.replace("_", "") == architecture.replace("_", "")}
    if len(classes) != 1:
        raise ValueError(f"Expected one class matching {architecture!r} in {module.__name__}: {list(classes)}")
    return next(iter(classes.values()))


def load_bundle(path: Path) -> tuple[type, type, dict]:
    """Load a complete bundle without changing it or relying on an installed generated package."""
    candidates = [path] if path.is_file() else sorted(path.glob("modeling_*_gen_npu.py"))
    entries = []
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8")
        source = re.search(r"^#\s+source\s*:\s*(\S+)", text, re.MULTILINE)
        architecture = re.search(r"^#\s+architecture\s*:\s*(\S+)", text, re.MULTILINE)
        if source and architecture:
            entries.append((candidate.resolve(), source.group(1), architecture.group(1)))
    if len(entries) != 1:
        raise ValueError("Specify the main generated modeling file with its source/architecture provenance header")
    entry, source_name, architecture = entries[0]
    digest = hashlib.sha256(entry.read_bytes()).hexdigest()
    package_name = "_hp_artifact_training_" + digest[:12]
    package = ModuleType(package_name)
    package.__path__ = [str(entry.parent)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + ".modeling_artifact", entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source_module = importlib.import_module(source_name)
    generated = _entry_class(module, architecture)
    if generated.__module__ != module.__name__:
        raise ValueError("The requested architecture is imported, not defined by the generated source")
    report = {"entry": str(entry), "sha256": digest, "architecture": architecture,
              "generated_class": f"{generated.__module__}:{generated.__name__}", "metadata_only_reference": True}
    return generated, _entry_class(source_module, architecture), report


def _graph_schema(model: nn.Module) -> dict:
    def tensors(values):
        return {name: (tuple(value.shape), str(value.dtype), bool(value.requires_grad)) for name, value in values}

    aliases = {}
    for name, value in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(value), []).append(name)
    return {"modules": sorted(name for name, _ in model.named_modules(remove_duplicate=False)),
            "parameters": tensors(model.named_parameters(remove_duplicate=False)),
            "buffers": tensors(model.named_buffers(remove_duplicate=False)),
            "aliases": sorted(sorted(names) for names in aliases.values() if len(names) > 1)}


def _method_functions(value):
    if isinstance(value, (staticmethod, classmethod)):
        return (value.__func__,)
    if isinstance(value, property):
        return tuple(method for method in (value.fget, value.fset, value.fdel) if method is not None)
    return (value,) if inspect.isfunction(value) else ()


def verify_component_source(target: nn.Module, source: nn.Module, artifact_root: Path) -> None:
    """Allow an independently emitted class only after checking native inheritance and all method bodies."""
    if type(target) is type(source):
        return
    if type(target).__name__ != type(source).__name__:
        raise ValueError("Generated component class name differs from native")
    if type(source) in type(target).__mro__[1:]:
        if type(target).__bases__ != (type(source),):
            raise ValueError('Thin component must directly inherit its native class')
        _verify_thin_component(target, artifact_root)
        return
    actual = {owner.__name__: owner for owner in type(target).__mro__}
    owners = _component_owners(type(source))
    if ([owner.__name__ for owner in type(target).__mro__]
            != [owner.__name__ for owner in type(source).__mro__]):
        raise ValueError("Generated component inheritance differs from native")
    for owner in owners:
        if owner is not type(source) and actual[owner.__name__] is owner:
            continue
        for name, declaration in vars(owner).items():
            _verify_method_declaration(actual[owner.__name__], name, declaration, artifact_root)



def _component_owners(source_type):
    """Select algorithm ancestors whose method bodies require source verification."""
    return [owner for owner in source_type.__mro__
            if owner is source_type or owner.__module__.startswith("hyper_parallel.components.modules.")]


def _verify_thin_component(target, artifact_root):
    """Restrict thin wrappers to artifact-owned forward dispatch."""
    allowed = {'forward', '_forward_impl', '_hp_boundary_form'}
    for name, value in vars(type(target)).items():
        if name in allowed:
            if inspect.isfunction(value) and Path(value.__code__.co_filename).resolve().parent != artifact_root:
                raise ValueError('Rewritten method is not artifact-owned')
            continue
        if name not in {'__module__', '__doc__', '__qualname__'}:
            raise ValueError('Unexpected component override: ' + name)


def _verify_method_declaration(owner: type, name: str, declaration, artifact_root: Path) -> None:
    expected = _method_functions(declaration)
    if not expected:
        return
    value = vars(owner).get(name)
    if type(value) is not type(declaration):
        raise ValueError(f"Generated method descriptor differs: {owner.__name__}.{name}")
    rendered = _method_functions(value)
    if len(rendered) != len(expected):
        raise ValueError(f"Generated method is missing: {owner.__name__}.{name}")
    for left, right in zip(rendered, expected):
        if Path(left.__code__.co_filename).resolve().parent != artifact_root:
            raise ValueError(f"Generated method falls back to native source: {owner.__name__}.{name}")
        if code_fingerprint(left.__code__) != code_fingerprint(right.__code__):
            raise ValueError(f"Generated method body differs: {owner.__name__}.{name}")
        if (left.__defaults__, left.__kwdefaults__) != (right.__defaults__, right.__kwdefaults__):
            raise ValueError(f"Generated method defaults differ: {owner.__name__}.{name}")


def _verify_component_options(target: nn.Module, source: nn.Module, fqn: str) -> None:
    for name in inspect.signature(type(source).__init__).parameters:
        value = getattr(source, name, None)
        if isinstance(value, (str, int, float, bool)) and getattr(target, name, None) != value:
            raise ValueError(f"Generated constructor option differs at {fqn}.{name}")
    for name, value in vars(source).items():
        if name.startswith("_hp_") or name == "_is_hf_initialized":
            continue
        if type(value) in (str, int, float, bool) and getattr(target, name, None) != value:
            raise ValueError(f"Generated component scalar state differs at {fqn}.{name}")
        if is_dataclass(value) and asdict(value) != asdict(getattr(target, name)):
            raise ValueError(f"Generated component configuration differs at {fqn}.{name}")


def restore_build_metadata(generated: nn.Module, reference: nn.Module, context: Any) -> None:
    """Transfer native construction declarations only, never parameters, buffers or executable methods."""
    model_builder._tie_model_weights_before_parallelism(reference)
    model_builder._tie_model_weights_before_parallelism(generated)
    if _graph_schema(generated) != _graph_schema(reference):
        raise ValueError("Generated constructor graph/state/alias schema differs from native prepared model")
    targets = dict(generated.named_modules(remove_duplicate=False))
    artifact_root = Path(inspect.getfile(type(generated))).resolve().parent
    for fqn, source in reference.named_modules(remove_duplicate=False):
        target = targets[fqn]
        info = get_family_build_info(source)
        if info is not None:
            if info.layout is not None:
                raise ValueError("Only pre-layout family declarations may be restored")
            verify_component_source(target, source, artifact_root)
            _verify_component_options(target, source, fqn)
            target._hp_family_build_info = info
        if hasattr(source, "_hp_reset_after_materialization"):
            target._hp_reset_after_materialization = source._hp_reset_after_materialization
    generated._hp_module_build_context = context
    generated._hp_model_adapter_spec = reference._hp_model_adapter_spec
    generated._hp_components_prepared = True
    generated._hp_family_weight_mapping = reference._hp_family_weight_mapping


@contextmanager
def artifact_model_build(path: Path, report: dict) -> Iterator[type]:
    """Instantiate generated constructors and reuse native planning/materialization/runtime binding."""
    generated_class, source_class, loaded = load_bundle(path)
    report.update(loaded)
    original_prepare = model_builder._maybe_apply_module_replacements
    original_methods = model_builder._apply_model_method_replacements

    def prepare(model, distributed_setup, low_precision_config, load_base_model):
        if not isinstance(model, generated_class):
            raise RuntimeError("The YAML model factory bypassed the generated class; refusing a native fallback")
        context = model._hp_module_build_context
        if getattr(distributed_setup.mesh_context, "pp_size", 1) > 1:
            raise NotImplementedError("This local artifact comparison currently requires pp_size=1")
        dtype = next(model.parameters()).dtype
        default_dtype = torch.get_default_dtype()
        try:
            with torch.random.fork_rng(devices=[]), torch.device("meta"):
                torch.set_default_dtype(dtype)
                reference = construct_family_model(source_class, deepcopy(model.config), context=context)
                reference, mapping = original_prepare(
                    reference, distributed_setup, low_precision_config, load_base_model)
        finally:
            torch.set_default_dtype(default_dtype)
        restore_build_metadata(model, reference, context)
        report["constructed_from_artifact"] = True
        report["restored_family_declarations"] = sum(
            get_family_build_info(child) is not None for child in model.modules())
        return model, mapping

    def methods(model, distributed_setup, low_precision_config):
        if isinstance(model, generated_class):
            if getattr(distributed_setup, "method_replacements", None):
                raise NotImplementedError(
                    "Method factory companion-state restoration is not implemented by this test loader")
            return None
        return original_methods(model, distributed_setup, low_precision_config)

    with generated_model_build(generated_class, loaded["architecture"]):
        with patch.object(model_builder, "_maybe_apply_module_replacements", prepare), \
                patch.object(model_builder, "_apply_model_method_replacements", methods):
            yield generated_class
