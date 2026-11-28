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
"""CPU controls for complete forward provenance and real output/gradient difference detection."""
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from common import inventory, assert_forward_coverage, numerical_differences
from forward_range import FullForwardEvidence
from run_forward_range import mutate_copy
from hyper_parallel.models._generated.phase_c import _GeneratedTPOperators
from hyper_parallel.trainer.base import BaseTrainer


class TestForwardRange(unittest.TestCase):
    """A valid delegate passes; an identical native bypass and changed mathematics do not."""

    def setUp(self) -> None:
        """Create a disposable artifact module and register unittest cleanup."""
        self.directory = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        path = self.root / 'modeling_probe.py'
        path.write_text('''import torch
from torch import nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.scale = 1.0
    def forward(self, value):
        return self._hp_tp.all_reduce(value * self.weight) * self.scale
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = Block()
    def forward(self, value):
        return {'logits': self.proj(value)}
''')
        name = 'range_fixture_' + self.root.name
        spec = importlib.util.spec_from_file_location(name, path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.module
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(self.module)

    def observe(self, label: str, generated: bool = True, scale: float = 1.0,
                bypass: bool = False, delegate: bool = False) -> dict:
        """Run a toy artifact under the same provenance and numerical observers."""
        model = self.module.Model()
        model.proj.scale = scale
        lowerer = SimpleNamespace(execution_op=lambda *_args, **_kwargs: SimpleNamespace(execute=lambda value: value))
        model.proj._hp_tp = _GeneratedTPOperators(lowerer)
        trainer = SimpleNamespace(model=model)
        with patch.object(BaseTrainer, '_build_model', lambda _trainer: None), \
                patch.object(BaseTrainer, 'preforward', lambda _trainer, batch: batch), \
                patch.object(BaseTrainer, 'step_optimizers_and_schedulers', lambda _trainer: None):
            observer = FullForwardEvidence(self.root / label, self.root / 'initial',
                                           'replay' if generated else 'record', type(model) if generated else None)
            observer.boundaries = {'proj'}
            with observer:
                if bypass:
                    model.proj.forward = MethodType(lambda child, value: value * child.weight, model.proj)
                elif delegate:
                    original = model.proj.forward
                    model.proj.forward = MethodType(lambda _child, value: original(value), model.proj)
                BaseTrainer._build_model(trainer)
                values = BaseTrainer.preforward(trainer, torch.ones(2))
                model(values)['logits'].sum().backward()
                BaseTrainer.step_optimizers_and_schedulers(trainer)
        return observer.report

    def test_all_source_forwards_tp_operations_outputs_and_gradients_are_observed(self):
        native = self.observe('native', generated=False)
        generated = self.observe('generated')
        assert_forward_coverage(generated, 1)
        self.assertEqual(generated['forward_calls'], {'.forward': 1, 'proj.forward': 1})
        self.assertEqual(generated['tp_calls'], {'proj:all_reduce': 1})
        self.assertFalse(numerical_differences(native, generated))
        self.assertIn('logits', generated['module_outputs'][''][0])
        self.assertIsNotNone(generated['gradients'][0]['values']['proj.weight'])

    def test_native_runtime_delegate_must_still_execute_every_artifact_body(self):
        self.observe('native', generated=False)
        generated = self.observe('delegate', delegate=True)
        assert_forward_coverage(generated, 1)

    def test_numerically_identical_native_bypass_fails_forward_coverage(self):
        native = self.observe('native', generated=False)
        generated = self.observe('bypass', bypass=True)
        self.assertFalse(numerical_differences(native, generated))
        with self.assertRaises(AssertionError):
            assert_forward_coverage(generated, 1)

    def test_real_arithmetic_difference_changes_outputs_and_optimizer_gradients(self):
        native = self.observe('native', generated=False)
        generated = self.observe('scaled', scale=1.125)
        assert_forward_coverage(generated, 1)
        changed = numerical_differences(native, generated)
        self.assertIn({'category': 'module_outputs', 'module': ''}, changed)
        self.assertIn({'category': 'module_outputs', 'module': 'proj'}, changed)
        self.assertIn({'category': 'optimizer_gradients'}, changed)

    def test_disposable_tp_cp_ep_mutations_execute_changed_mathematics(self):
        source = self.root / 'bundle'
        source.mkdir()
        path = source / 'modeling_probe_gen_npu.py'
        path.write_text('''from types import SimpleNamespace
class CPGroupedQueryAttentionModule:
    _hp_tp = SimpleNamespace(reduce_scatter=lambda value, dim: value)
    def _forward_impl(self, value):
        return (value, None)
    def forward(self, value):
        outputs = self._forward_impl(value)
        output = self._hp_tp.reduce_scatter(outputs[0], dim=1)
        return (output, None)
class EPGroupedExperts:
    def forward(self, value):
        return value
''')
        before = inventory(source)
        for kind in ('tp', 'cp', 'ep'):
            with self.subTest(kind=kind):
                destination = self.root / kind
                mutate_copy(source, destination, kind)
                namespace = {}
                code = (destination / path.name).read_text(encoding='utf-8')
                exec(compile(code, str(destination / path.name), 'exec'), namespace)  # pylint: disable=exec-used
                name = 'EPGroupedExperts' if kind == 'ep' else 'CPGroupedQueryAttentionModule'
                value = namespace[name]().forward(8)
                self.assertEqual(value if kind == 'ep' else value[0], 9)
                self.assertEqual(inventory(source), before)


if __name__ == '__main__':
    unittest.main()
