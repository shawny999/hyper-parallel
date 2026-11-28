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
"""CPU/meta worker calling the public CLI with test-only observation and resource guards."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import json
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch

from common import audit_bundle, code_fingerprint, declared_stages, dump

import hyper_parallel
from hyper_parallel.codegen import cli, manager
from hyper_parallel.codegen.session import CodegenBuildSession

def main() -> int:
    """Generate a real bundle and persist evidence separately from its contents."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--release", type=Path)
    args, overrides = parser.parse_known_args()
    imported = Path(hyper_parallel.__file__).resolve().parents[1]
    if imported != args.source.resolve():
        raise RuntimeError(f"Editable import mismatch: {imported} != {args.source}")
    report = {"source": str(imported), "pid": os.getpid(), "overrides": overrides, "snapshots": []}
    original_finish = CodegenBuildSession.finish
    original_lock = manager.artifact_generation_lock

    def event(kind: str) -> None:
        if args.events:
            args.events.parent.mkdir(parents=True, exist_ok=True)
            with args.events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": kind, "time_ns": time.monotonic_ns(),
                                         "pid": os.getpid()}) + "\n")

    def finish(session, *positional, **keywords):
        snapshot = original_finish(session, *positional, **keywords)
        plan = snapshot.plan or {}
        report["snapshots"].append({
            "status": snapshot.status.value,
            "replacements": list(snapshot.replacements.applications),
            "parallel_dims": plan.get("parallel_dims", {}),
            "param_plan": plan.get("param_plan", {}),
            "boundaries": plan.get("boundary_classes", {}),
            "injections": plan.get("injections", []),
            "source_file": snapshot.source.get("file_path"),
            "source_sha256": hashlib.sha256(Path(snapshot.source["file_path"]).read_bytes()).hexdigest(),
        })
        return snapshot

    @contextmanager
    def lock(*positional, **keywords):
        event("lock_requested")
        with original_lock(*positional, **keywords):
            event("lock_acquired")
            deadline = time.monotonic() + 120
            while args.release and not args.release.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Acceptance publication barrier was not released")
                time.sleep(0.02)
            yield
            event("lock_releasing")

    def forbidden(*_positional, **_keywords):
        raise AssertionError("Offline generation attempted to acquire distributed/device runtime state")

    code = 1
    with ExitStack() as stack:
        stack.enter_context(patch.object(CodegenBuildSession, "finish", finish))
        stack.enter_context(patch.object(manager, "artifact_generation_lock", lock))
        for name in ("init_process_group", "new_group", "get_rank", "get_world_size"):
            stack.enter_context(patch.object(torch.distributed, name, forbidden))
        stack.enter_context(patch.object(torch.cuda, "init", forbidden))
        if hasattr(torch, "npu") and hasattr(torch.npu, "init"):
            stack.enter_context(patch.object(torch.npu, "init", forbidden))
        try:
            event("generation_started")
            code = cli.main(["--verbose", "generate", "--config", str(args.yaml),
                             "--output", str(args.output), *overrides])
            if code == 0:
                required = {}
                for snapshot in report["snapshots"]:
                    for record in snapshot["replacements"]:
                        module_name, name = record["replacement_type"].split(":")
                        component = getattr(importlib.import_module(module_name), name)
                        declaration = getattr(component, "parallel_module_spec", None)
                        if callable(declaration):
                            required[name] = {method: code_fingerprint(getattr(component, method).__code__)
                                              for method in declared_stages(inspect.getsource(declaration))}
                report["required_methods"] = required
                report["files"] = audit_bundle(args.output, required)
            return code
        finally:
            report["exit"] = code
            dump(args.evidence, report)


if __name__ == "__main__":
    raise SystemExit(main())
