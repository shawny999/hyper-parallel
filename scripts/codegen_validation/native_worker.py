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
"""Test-only observers around the unchanged standard training entry point."""

from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import json
from importlib import metadata
import logging
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch

from common import DISTRIBUTED_ENV, dump

import torch_npu
import hyper_parallel
from hyper_parallel.codegen import train_sidecar
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.models import replacement
from hyper_parallel.trainer.base import BaseTrainer

BAD_RULE = [{"match": "model.layers.*.self_attn", "replace": {"kind": "module", "implementation": {
    "_target_": "acceptance_missing_factory.NoSuchFactory"}}}]


class LogCapture(logging.Handler):
    """Capture sidecar log levels without changing error handling."""

    def __init__(self, records: list) -> None:
        """Retain the test-owned event list."""
        super().__init__()
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        """Retain the original level and formatted message."""
        self.records.append({"level": record.levelname, "message": record.getMessage()})


def layout_value(value: Any) -> Any:
    """Encode native placement objects directly, without the Codegen observation helpers."""
    if isinstance(value, dict):
        return {str(getattr(key, "value", key)): layout_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [layout_value(item) for item in value]
    name = type(value).__name__
    if name == "Replicate":
        return "R"
    if name == "StridedShard":
        return f"SS({value.dim},{value.split_factor})"
    if name == "Shard":
        return f"S({value.dim})"
    if name == "Partial":
        return f"P({value.reduce_op})"
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Native layout oracle cannot encode {type(value)}")



def _publication_pause_line(entry: Path) -> int:
    """Locate the standard entry boundary immediately after sidecar startup."""
    tree = ast.parse(entry.read_text(encoding="utf-8"))
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    index = next(index for index, node in enumerate(main_node.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name) and node.value.func.id == "start_codegen_sidecar")
    return main_node.body[index + 1].lineno


def main() -> None:
    """Run native training and record all ranks, optimizer steps and actual replacements."""
    parser = argparse.ArgumentParser()
    for name in ("source", "entry", "yaml", "output", "evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--child-invalid-target", action="store_true")
    parser.add_argument("--publication-only", action="store_true")
    args, overrides = parser.parse_known_args()
    assert Path(hyper_parallel.__file__).resolve().parents[1] == args.source.resolve()
    rank = int(os.environ.get("RANK", "0"))
    report = {"rank": rank, "pid": os.getpid(), "source": str(args.source.resolve()), "steps": [],
              "torch_version": metadata.version("torch"), "python_version": sys.version.split()[0],
              "optimizer_steps": 0, "launches": [], "replacements": [], "plans": [], "sidecar_logs": [],
              "native_completed": False}
    parent_environment = dict(os.environ)
    children = []
    original_popen = subprocess.Popen
    original_step = BaseTrainer.on_step_end
    original_optimizer = BaseTrainer.step_optimizers_and_schedulers
    original_install = replacement._install_module_candidates
    original_plan = ShardingPlanner.plan
    original_fusion = torch_npu.npu_fusion_attention
    attention_owners = {}
    report['fusion_attention_calls'] = {}
    report['fusion_attention_interfaces'] = {}

    def fusion(*values, **keywords):
        owner = sys._getframe(1).f_locals.get('module')
        result = original_fusion(*values, **keywords)
        for fqn in attention_owners.get(id(owner), ()):
            counts = report['fusion_attention_calls']
            counts[fqn] = counts.get(fqn, 0) + 1
        return result

    def popen(command, **keywords):
        if not isinstance(command, list) or command[1:3] != ["-m", "hyper_parallel.codegen"]:
            return original_popen(command, **keywords)
        command = [*command, "--output", str(args.output)]
        if args.child_invalid_target:
            command.append("--plan_overrides=" + json.dumps(BAD_RULE))
        process = original_popen(command, **keywords)  # pylint: disable=consider-using-with
        report["launches"].append({"pid": process.pid, "ppid": os.getpid(), "command": command,
                                   "environment": {name: keywords["env"][name] for name in DISTRIBUTED_ENV
                                                   if name in keywords["env"]}})
        children.append(process)
        return process

    def optimizer(trainer):
        original_optimizer(trainer)
        report["optimizer_steps"] += 1

    def step(trainer, loss=None, loss_dict=None, grad_norm=None):
        original_step(trainer, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        report["steps"].append({"step": len(report["steps"]) + 1, "loss": float(loss)})
        dump(args.evidence / f"rank-{rank}.json", report)

    def install(model, candidates):
        candidates = tuple(candidates)
        result = original_install(model, candidates)
        for fqns, source, candidate in candidates:
            interface = getattr(candidate, 'attention_interface', None)
            if interface is not None:
                attention_owners[id(candidate)] = tuple(fqns)
                for fqn in fqns:
                    report['fusion_attention_interfaces'][fqn] = interface.__module__ + ':' + interface.__name__
            report["replacements"].append({"module_fqns": list(fqns),
                "source_type": f"{type(source).__module__}:{type(source).__name__}",
                "replacement_type": f"{type(candidate).__module__}:{type(candidate).__name__}"})
        return result

    def plan(planner, model, mesh, **keywords):
        result = original_plan(planner, model, mesh, **keywords)
        report["plans"].append({"arguments": {key: value for key, value in keywords.items()
                                             if isinstance(value, (str, int, bool, float))},
                                "mesh_dim_names": list(result.mesh_dim_names),
                                "modules": {fqn: {field: layout_value(getattr(spec, field, None))
                                                  for field in ("in_src", "in_dst", "out_src", "out_dst")}
                                            for fqn, spec in result.modules.items()}})
        return result

    handler = LogCapture(report["sidecar_logs"])
    train_sidecar.logger.addHandler(handler)

    class PublicationComplete(Exception):
        """End only the publication-only test before constructing Trainer runtime."""

    pause_line = -1

    def trace(frame, event, _argument):
        if (event == "line" and frame.f_code.co_name == "main"
                and Path(frame.f_code.co_filename).resolve() == args.entry.resolve()
                and frame.f_lineno == pause_line):
            sys.settrace(None)
            ready = args.evidence / "publication.complete"
            if rank == 0:
                assert len(children) == 1
                assert children[0].wait(timeout=600) == 0
                ready.touch()
            else:
                deadline = time.monotonic() + 600
                while not ready.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Rank 0 never completed sidecar publication")
                    time.sleep(0.05)
            raise PublicationComplete()
        return trace

    if args.publication_only:
        pause_line = _publication_pause_line(args.entry)
        args.evidence.mkdir(parents=True, exist_ok=True)
        sys.settrace(trace)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch_npu, 'npu_fusion_attention', fusion))
            stack.enter_context(patch.object(train_sidecar, "subprocess", SimpleNamespace(Popen=popen)))
            stack.enter_context(patch.object(BaseTrainer, "on_step_end", step))
            stack.enter_context(patch.object(BaseTrainer, "step_optimizers_and_schedulers", optimizer))
            stack.enter_context(patch.object(replacement, "_install_module_candidates", install))
            stack.enter_context(patch.object(ShardingPlanner, "plan", plan))
            sys.argv = [str(args.entry), str(args.yaml), *overrides]
            runpy.run_path(str(args.entry), run_name="__main__")
            report["native_completed"] = True
    except PublicationComplete:
        report["publication_only"] = True
    except Exception as exc:
        report["native_error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        sys.settrace(None)
        report["child_exits"] = [child.wait(timeout=600) for child in children]
        for thread in threading.enumerate():
            if thread.name == "hyper-parallel-codegen-sidecar":
                thread.join(timeout=10)
        report["distributed_identity_unchanged"] = all(
            os.environ.get(name) == parent_environment.get(name)
            for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"))
        dump(args.evidence / f"rank-{rank}.json", report)
        train_sidecar.logger.removeHandler(handler)


if __name__ == "__main__":
    main()
