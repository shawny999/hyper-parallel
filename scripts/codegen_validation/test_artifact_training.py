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
"""CPU tests proving that the comparison cannot silently train the native model instead."""

from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import inspect
import json
from pathlib import Path
import random
from types import MethodType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from artifact_training import artifact_model_build, load_bundle, restore_build_metadata, verify_component_source
from common import audit_bundle, code_fingerprint, dump
from compare_bundle_training import verify_runs
from training_evidence import TrainingEvidence, restore_rng, rng_state

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.distributed.module_binding import ParallelModuleSpec


class ToyParallelBlock(nn.Module):
    """A tiny stage owner whose source can be independently emitted on CPU."""

    def __init__(self) -> None:
        """Create one CPU parameter."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Call the selected stage."""
        return self.dispatch(value)

    def dispatch(self, value: torch.Tensor) -> torch.Tensor:
        """Keep a visible toy numerical operation."""
        return value * self.weight

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Declare stages without acquiring resources."""
        return ParallelModuleSpec("toy.cp", "cp", {"dispatch": ("value",)})


class ToyModel(nn.Module):
    """A fixed-layout CPU model for state/alias replay, not an eight-card training substitute."""

    def __init__(self) -> None:
        """Create a trainable parameter and a derived buffer."""
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.proj = nn.Linear(2, 2)
        self.register_buffer("scale", torch.ones(1), persistent=False)
        self._hp_model_adapter_spec = None
        self._hp_family_weight_mapping = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the model-owned computation."""
        return self.proj(value) * self.scale


class TestArtifactTraining(unittest.TestCase):
    """Verify constructor provenance, fixed randomness and negative loss/provenance oracles."""

    def setUp(self) -> None:
        """Use only an isolated test directory."""
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.root = Path(self.contexts.enter_context(tempfile.TemporaryDirectory()))

    def test_training_loader_reuses_existing_generated_phase_binding_and_restores_it(self):
        """Source-owned TP boundaries use the existing binder without replacing native Trainer code."""
        apply_module = importlib.import_module("hyper_parallel.distributed.apply")
        generated_module = importlib.import_module("hyper_parallel.models._generated.phase_c")
        original = apply_module._apply_phase_c
        loaded = (ToyModel, ToyModel, {"architecture": "ToyModel"})
        with patch("artifact_training.load_bundle", return_value=loaded):
            with artifact_model_build(self.root, {}):
                self.assertIs(apply_module._apply_phase_c, generated_module.apply_generated_phase_c)
        self.assertIs(apply_module._apply_phase_c, original)

    def test_fingerprint_normalizes_docstring_indentation_but_keeps_math_constants(self):
        """Moving a method changes docstring indentation, while a changed operand must still fail."""
        codes = []
        for indent, constant in ((8, 2), (4, 2), (4, 3)):
            spaces = ' ' * indent
            text = ('def forward(value):\n    """Math.\n\n' + spaces + 'Args:\n' + spaces
                    + '    value: Input.\n' + spaces + '"""\n    return value + ' + str(constant) + '\n')
            codes.append(compile(text, '<fingerprint-fixture>', 'exec').co_consts[0])
        self.assertEqual(code_fingerprint(codes[0]), code_fingerprint(codes[1]))
        self.assertNotEqual(code_fingerprint(codes[0]), code_fingerprint(codes[2]))

    def emitted_model(self, replacement: str | None = None) -> nn.Module:
        """Load independent classes with the same native algorithm and a real artifact file."""
        expanded = SimpleNamespace(imports=(), source=inspect.getsource(ToyParallelBlock))
        text = ("import torch\nfrom torch import nn\nfrom types import SimpleNamespace\n"
                "from hyper_parallel.distributed.module_binding import ParallelModuleSpec\n"
                + "\n".join(expanded.imports) + "\n\n" + expanded.source + "\n" + inspect.getsource(ToyModel))
        if replacement is not None:
            text = text.replace("return value * self.weight", replacement)
        path = self.root / f"modeling_{len(list(self.root.glob('modeling_*.py')))}.py"
        path.write_text(text, encoding="utf-8")
        name = "stage_artifact_" + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(module)
        model = getattr(module, "ToyModel")()
        model.proj = getattr(module, "ToyParallelBlock")()
        return model

    def test_independent_native_equivalent_component_restores_metadata_only(self):
        """Class identity may differ, but native declarations never replace artifact code or weights."""
        native, generated = ToyModel(), self.emitted_model()
        native.proj = ToyParallelBlock()
        with torch.no_grad():
            generated.proj.weight.fill_(7)
        info = SimpleNamespace(layout=None)
        self.assertIsNot(type(generated.proj), type(native.proj))
        with patch("artifact_training.get_family_build_info",
                   side_effect=lambda item: info if item is native.proj else None):
            restore_build_metadata(generated, native, {})
        self.assertIs(generated.proj._hp_family_build_info, info)
        self.assertTrue(torch.all(generated.proj.weight == 7))
        self.assertEqual(generated.proj.dispatch.__func__.__module__, type(generated).__module__)

    def test_same_name_changed_body_and_native_fallback_are_rejected(self):
        """A matching class name alone never proves generated implementation correctness."""
        native = ToyParallelBlock()
        changed = self.emitted_model("return value + self.weight")
        with self.assertRaisesRegex(ValueError, "body differs"):
            verify_component_source(changed.proj, native, self.root)
        verify_component_source(native, native, self.root)

    def test_readability_audit_rejects_import_only_missing_stage_and_changed_stage(self):
        """Constructors and matching class names cannot substitute for visible native methods."""
        model = self.emitted_model()
        text = Path(inspect.getfile(type(model))).read_text(encoding="utf-8")
        bundle = self.root / "bundle"
        bundle.mkdir()
        (bundle / "__init__.py").write_text("", encoding="utf-8")
        entry = bundle / "modeling_Toy_gen_npu.py"
        entry.with_suffix(".py.diff").write_text("diff", encoding="utf-8")
        required = {"ToyParallelBlock": {name: code_fingerprint(getattr(ToyParallelBlock, name).__code__)
                                          for name in ("forward", "dispatch")}}
        entry.write_text(text, encoding="utf-8")
        audit_bundle(bundle, required)
        wrapped = text.replace("def forward(self, value: torch.Tensor) -> torch.Tensor:",
                               "def _forward_impl(self, value):")
        wrapped = wrapped.replace("    def dispatch(", "    def forward(self, value):\n"
                                  "        return self._forward_impl(self._hp_tp.to_local(value))\n\n"
                                  "    def dispatch(")
        entry.write_text(wrapped, encoding="utf-8")
        audit_bundle(bundle, required)
        for candidate in ("from test_artifact_training import ToyParallelBlock\n", wrapped.replace(
                "def _forward_impl(", "def missing_core("), text.replace(
                "def dispatch(", "def removed_dispatch("), text.replace(
                "return value * self.weight", "return value + self.weight")):
            entry.write_text(candidate, encoding="utf-8")
            with self.assertRaises(AssertionError):
                audit_bundle(bundle, required)

    def test_stage_profile_records_artifact_dispatch_and_detects_native_fallback(self):
        """Actual selected stage calls, not only the model entry forward, must execute artifact code."""
        model = self.emitted_model()
        observer = TrainingEvidence(self.root / "calls", self.root / "initial", "record", type(model))
        observer.model = model
        observer.collect_stage_owners()
        previous = sys.getprofile()
        try:
            sys.setprofile(observer._profile)
            model(torch.ones(2))
            model.proj.dispatch = MethodType(ToyParallelBlock.dispatch, model.proj)
            model(torch.ones(2))
        finally:
            sys.setprofile(previous)
        self.assertEqual(observer.report["artifact_stage_calls"]["proj.dispatch"], 1)
        self.assertEqual(observer.report["native_stage_fallback_calls"]["proj.dispatch"], 1)

    def test_metadata_restore_never_copies_forward_or_parameter_values(self):
        """Restoring runtime declarations must not turn the artifact into the reference implementation."""
        native, generated = ToyModel(), ToyModel()
        with torch.no_grad():
            native.proj.weight.fill_(2)
            generated.proj.weight.fill_(7)
        function = generated.forward.__func__
        restore_build_metadata(generated, native, {})
        self.assertIs(generated.forward.__func__, function)
        self.assertTrue(torch.all(generated.proj.weight == 7))
        self.assertTrue(generated._hp_components_prepared)

    def test_super_delegate_requires_the_emitted_forward_on_stack(self):
        """A native base call is legitimate only behind the actual emitted forward."""
        model = self.emitted_model()
        module = sys.modules[type(model).__module__]
        extra = (f'import {__name__}\nclass Thin({__name__}.ToyParallelBlock):\n'
                 '    def forward(self, value):\n        return super().forward(value)\n'
                 '    def dispatch(self, value):\n        return value * self.weight\n')
        with Path(module.__file__).open('a', encoding='utf-8') as handle:
            handle.write(extra)
        exec(compile(extra, module.__file__, 'exec'), vars(module))  # pylint: disable=exec-used
        model.proj = module.Thin()
        observer = TrainingEvidence(self.root / 'delegate', self.root / 'initial', 'record', type(model))
        observer.model = model
        observer.collect_stage_owners()
        previous = sys.getprofile()
        try:
            sys.setprofile(observer._profile)
            model(torch.ones(2))
            self.assertEqual(observer.report['native_delegate_calls']['proj.forward'], 1)
            self.assertFalse(observer.report['native_stage_fallback_calls'])
            ToyParallelBlock.forward(model.proj, torch.ones(2))
        finally:
            sys.setprofile(previous)
        self.assertEqual(observer.report['native_stage_fallback_calls']['proj.forward'], 1)

    def test_metadata_restore_rejects_changed_constructor_schema(self):
        """Missing/extra modules cannot be repaired from a reference model."""
        native, generated = ToyModel(), ToyModel()
        generated.extra = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "schema differs"):
            restore_build_metadata(generated, native, {})

    def test_rng_state_round_trips_python_numpy_and_torch(self):
        """The comparison fixes all CPU random streams, not only torch.manual_seed."""
        state = rng_state()
        first = (random.random(), float(np.random.random()), torch.rand(4))
        restore_rng(state)
        second = (random.random(), float(np.random.random()), torch.rand(4))
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_fixed_state_replay_preserves_parameter_identity_and_rng(self):
        """Actual tensors are restored before optimizer creation while parameter objects remain owned by the model."""
        first = ToyModel()
        trainer = SimpleNamespace(model=first)
        initial = self.root / "initial"
        with patch.object(BaseTrainer, "_build_model", lambda _trainer: None):
            with TrainingEvidence(self.root / "native", initial, "record") as native:
                BaseTrainer._build_model(trainer)
            second = ToyModel()
            trainer.model = second
            identity = id(second.proj.weight)
            with TrainingEvidence(self.root / "generated", initial, "replay") as generated:
                BaseTrainer._build_model(trainer)
        self.assertEqual(identity, id(second.proj.weight))
        self.assertEqual(native.report["initial_hashes"], generated.report["initial_hashes"])
        torch.testing.assert_close(first.proj.weight, second.proj.weight, rtol=0, atol=0)

    def test_loader_requires_a_class_defined_by_the_artifact(self):
        """A source file that merely reexports the native model cannot masquerade as generated training."""
        source = self.root / "native_source.py"
        source.write_text("class ProbeModel:\n    pass\n", encoding="utf-8")
        name = "artifact_training_native_probe"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(module)
        artifact = self.root / "modeling_Probe_gen_npu.py"
        artifact.write_text(f"#   source : {name}\n#   architecture : ProbeModel\nfrom {name} import ProbeModel\n",
                            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not defined"):
            load_bundle(artifact)

    def test_comparison_rejects_unexecuted_artifact_or_changed_loss(self):
        """Matching labels and hashes are insufficient when no generated forward executes."""
        for label in ("native", "native-repeat", "generated"):
            directory = self.root / label
            dump(directory / "native/rank-0.json", {
                "native_completed": True, "optimizer_steps": 2,
                "steps": [{"step": 1, "loss": 1}, {"step": 2, "loss": 0.9}],
                "child_exits": [0], "distributed_identity_unchanged": True, "torch_version": "fixture"})
            dump(directory / "comparison-rank-0.json", {"initial_schema": {}, "initial_hashes": {},
                "inputs": [{"same": True}], "final_hashes": {}, "artifact_forward_calls": {"forward": 2},
                "artifact_stage_expectations": {"experts.dispatch": {"class": "Experts"}},
                "artifact_stage_calls": {"experts.dispatch": 2}, "native_stage_fallback_calls": {}})
            dump(directory / "model-source-rank-0.json", {"constructed_from_artifact": label == "generated"})
        self.assertEqual(verify_runs(self.root, 1, 2, 0, 0)["status"], "PASS")
        observed = {"initial_schema": {}, "initial_hashes": {}, "inputs": [{"same": True}],
                    "final_hashes": {}, "artifact_forward_calls": {},
                    "artifact_stage_expectations": {"experts.dispatch": {"class": "Experts"}},
                    "artifact_stage_calls": {"experts.dispatch": 2}, "native_stage_fallback_calls": {}}
        dump(self.root / "generated/comparison-rank-0.json", observed)
        with self.assertRaises(AssertionError):
            verify_runs(self.root, 1, 2, 0, 0)
        observed["artifact_forward_calls"] = {"forward": 2}
        observed["artifact_stage_calls"] = {}
        dump(self.root / "generated/comparison-rank-0.json", observed)
        with self.assertRaises(AssertionError):
            verify_runs(self.root, 1, 2, 0, 0)
        observed["artifact_stage_calls"] = {"experts.dispatch": 2}
        observed["native_stage_fallback_calls"] = {"experts.dispatch": 1}
        dump(self.root / "generated/comparison-rank-0.json", observed)
        with self.assertRaises(AssertionError):
            verify_runs(self.root, 1, 2, 0, 0)
        observed["native_stage_fallback_calls"] = {}
        dump(self.root / "generated/comparison-rank-0.json", observed)
        path = self.root / "generated/native/rank-0.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["steps"][1]["loss"] = 0.8
        dump(path, row)
        with self.assertRaises(AssertionError):
            verify_runs(self.root, 1, 2, 0, 0)


if __name__ == "__main__":
    unittest.main()
