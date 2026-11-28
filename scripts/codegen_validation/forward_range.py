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
"""Test-only forward provenance, complete output and optimizer-gradient observations."""
from __future__ import annotations

import ast
from collections.abc import Mapping
import inspect
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

from common import code_fingerprint
from training_evidence import TrainingEvidence, tensor_digest
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.trainer.base import BaseTrainer


def output_digest(value: Any) -> Any:
    """Compare every returned tensor, including local DTensor shards, without modifying the graph."""
    if isinstance(value, torch.Tensor):
        return tensor_digest(value)
    if isinstance(value, Mapping):
        return {str(name): output_digest(item) for name, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (tuple, list)):
        return [output_digest(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported forward output: {type(value).__name__}")


def source_forwards(model_type: type) -> dict:
    """Freeze artifact-owned forward declarations before any Trainer runtime binding."""
    module = sys.modules[model_type.__module__]
    path = Path(module.__file__).resolve()
    result = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if not isinstance(node, ast.ClassDef):
            continue
        owner = getattr(module, node.name, None)
        if not inspect.isclass(owner):
            continue
        for declaration in node.body:
            if not isinstance(declaration, ast.FunctionDef) or declaration.name not in ('forward', '_forward_impl'):
                continue
            function = inspect.unwrap(getattr(owner, declaration.name))
            code = getattr(function, '__code__', None)
            if code is None or Path(code.co_filename).resolve() != path:
                raise ValueError(f"Forward is not owned by its artifact: {node.name}.{declaration.name}")
            tp_operations = []
            for item in ast.walk(declaration):
                if (isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                        and isinstance(item.func.value, ast.Attribute) and item.func.value.attr == '_hp_tp'):
                    tp_operations.append(item.func.attr)
            result[(owner, declaration.name)] = {'code': code, 'class': node.name,
                                                'sha256': code_fingerprint(code), 'tp_operations': tp_operations}
    return result


class FullForwardEvidence(TrainingEvidence):
    """Extend the existing fixed-state oracle without replacing any model computation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Extend the fixed-state observer with forward and boundary inventories."""
        super().__init__(*args, **kwargs)
        self.declarations = source_forwards(self.generated) if self.generated is not None else {}
        self.forward_owners = {}
        self.module_names = {}
        self.boundaries = set()
        self.report.update(forward_inventory={}, forward_calls={}, runtime_wrapper_calls={}, tp_calls={},
                           module_outputs={}, gradients=[], boundary_modules=[], source_declarations=[])

    def __enter__(self) -> FullForwardEvidence:
        """Install scoped observers without changing the model's computations."""
        super().__enter__()
        original_build = BaseTrainer._build_model
        original_plan = ShardingPlanner.plan
        original_step = BaseTrainer.step_optimizers_and_schedulers

        def plan(planner, model, mesh, **kwargs):
            result = original_plan(planner, model, mesh, **kwargs)
            self.boundaries.update(name for name, spec in result.modules.items() if spec.is_boundary)
            return result

        def build(trainer):
            original_build(trainer)
            self.install_forward_observers()

        def step(trainer):
            values = {}
            for name, parameter in self.model.named_parameters():
                gradient = getattr(parameter, 'main_grad', None)
                gradient = parameter.grad if gradient is None else gradient
                values[name] = None if gradient is None else tensor_digest(gradient)
            if not any(value is not None for value in values.values()):
                raise ValueError('No optimizer gradients were observed')
            self.report['gradients'].append({'step': len(self.report['gradients']) + 1, 'values': values})
            return original_step(trainer)

        self.stack.enter_context(patch.object(ShardingPlanner, 'plan', plan))
        self.stack.enter_context(patch.object(BaseTrainer, '_build_model', build))
        self.stack.enter_context(patch.object(BaseTrainer, 'step_optimizers_and_schedulers', step))
        return self

    def install_forward_observers(self) -> None:
        """Observe all actual modules and require every instantiated source forward to run."""
        seen = set()
        instantiated = set()
        for fqn, child in self.model.named_modules(remove_duplicate=False):
            if id(child) in seen:
                continue
            seen.add(id(child))
            self.module_names[id(child)] = fqn
            for name in ('forward', '_forward_impl'):
                declaration = next((self.declarations[(owner, name)] for owner in type(child).__mro__
                                    if (owner, name) in self.declarations), None)
                if declaration is None:
                    continue
                instantiated.add((declaration['class'], name))
                owner = next(owner for owner in type(child).__mro__ if name in vars(owner))
                current = inspect.unwrap(getattr(owner, name))
                if getattr(current, '__code__', None) is not declaration['code']:
                    raise ValueError(f"Runtime binding replaced an artifact class method: {fqn}.{name}")
                key = f'{fqn}.{name}'
                bound = getattr(child, name)
                bound_code = getattr(getattr(bound, '__func__', bound), '__code__', None)
                self.report['forward_inventory'][key] = {
                    'class': declaration['class'], 'method': name, 'sha256': declaration['sha256'],
                    'file': str(Path(declaration['code'].co_filename).resolve()),
                    'bound_file': None if bound_code is None else str(Path(bound_code.co_filename).resolve()),
                    'tp_operations': declaration['tp_operations'],
                }
                self.forward_owners[(id(child), name)] = (key, declaration['code'])
            if isinstance(child, (nn.ModuleList, nn.ModuleDict)):
                continue
            self.stack.callback(child.register_forward_hook(self._output_hook(fqn)).remove)
        self.report['source_declarations'] = [
            {'class': row['class'], 'method': name, 'instantiated': (row['class'], name) in instantiated}
            for (_, name), row in self.declarations.items()]
        self.report['boundary_modules'] = sorted(self.boundaries)
        self._flush()

    def _output_hook(self, fqn):
        def observe(_module, _arguments, outputs):
            values = self.report['module_outputs'].setdefault(fqn, [])
            values.append(output_digest(outputs))
        return observe

    def _profile(self, frame, event, argument):
        super()._profile(frame, event, argument)
        if event != 'call':
            return
        name = frame.f_code.co_name
        if name == '_execute' and frame.f_code.co_filename.replace('\\', '/').endswith('models/_generated/phase_c.py'):
            parent = frame.f_back
            while parent is not None:
                found = self.forward_owners.get((id(parent.f_locals.get('self')), parent.f_code.co_name))
                if found is not None and found[1] is parent.f_code:
                    key = found[0].rsplit('.', 1)[0] + ':' + str(frame.f_locals.get('kind'))
                    self.report['tp_calls'][key] = self.report['tp_calls'].get(key, 0) + 1
                    break
                parent = parent.f_back
            return
        if name not in ('forward', '_forward_impl'):
            return
        expected = self.forward_owners.get((id(frame.f_locals.get('self')), name))
        if expected is None:
            return
        key, code = expected
        bucket = 'forward_calls' if frame.f_code is code else 'runtime_wrapper_calls'
        self.report[bucket][key] = self.report[bucket].get(key, 0) + 1
