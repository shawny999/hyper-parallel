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
"""Local acceptance helpers; importing this module never imports the framework."""

from __future__ import annotations

import ast
import hashlib
import inspect
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time
from types import CodeType
import textwrap


C8 = ["--accelerator.tp_size=2", "--accelerator.cp_size=2", "--accelerator.ep_size=2",
      "--accelerator.pp_size=1", "--accelerator.sequence_parallel=true", "--accelerator.loss_parallel=true",
      "--fsdp_config.dp_shard_size=4", "--fsdp_config.edp_shard_size=4"]
C0 = ["--accelerator.tp_size=1", "--accelerator.cp_size=1", "--accelerator.ep_size=1",
      "--accelerator.pp_size=1", "--accelerator.sequence_parallel=false", "--accelerator.loss_parallel=false",
      "--fsdp_config.dp_shard_size=1", "--fsdp_config.edp_shard_size=1", "--plan_overrides=[]"]
DISTRIBUTED_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                   "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
                   "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_ERROR_FILE")


class Blocked(RuntimeError):
    """A required resource or baseline is unavailable; this is never a passing case."""


def npu_processes(output: str) -> list[dict[str, int]]:
    """Read device/PID rows and reject an unrecognized process-table format."""
    _, header, table = output.partition("Process id")
    if not header:
        raise ValueError("npu-smi output has no process-table header")
    result = []
    for line in table.splitlines()[1:]:
        if re.match(r"^\|\s+\d", line):
            match = re.match(r"^\|\s+(\d+)\s+\d+\s*\|\s*(\d+)\s*\|", line)
            if match is None:
                raise ValueError(f"Unrecognized NPU process row: {line}")
            result.append({"device": int(match.group(1)), "pid": int(match.group(2))})
    if not result and "No running processes found" not in table:
        raise ValueError("npu-smi process table is incomplete")
    return result


def dump(path: Path, value: object) -> None:
    """Persist test evidence outside generated bundles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def inventory(root: Path) -> dict[str, str]:
    """Hash every regular bundle file, rejecting symlinks and unknown extra content."""
    result = {}
    if not root.is_dir():
        return result
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AssertionError(f"Unexpected bundle symlink: {path}")
        if path.is_file() and "__pycache__" not in path.relative_to(root).parts:
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def code_fingerprint(code: CodeType) -> str:
    """Compare executable bodies without file paths, line numbers or annotation flags."""
    values = list(code.co_consts)
    if values and isinstance(values[0], str):
        values[0] = inspect.cleandoc(values[0])
    constants = tuple(_constant_fingerprint(value) for value in values)
    fields = (code.co_code, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
              code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, constants)
    return hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()


def _constant_fingerprint(value):
    if isinstance(value, CodeType):
        return code_fingerprint(value)
    if isinstance(value, tuple):
        return tuple(_constant_fingerprint(item) for item in value)
    if isinstance(value, frozenset):
        return ("frozenset", sorted(repr(_constant_fingerprint(item)) for item in value))
    return (type(value).__name__, repr(value))


def declared_stages(source: str) -> set[str]:
    """Read the native binding declaration without calling a resource factory."""
    names = {"forward"}
    found = False
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "ParallelModuleSpec":
            continue
        values = ([node.args[2]] if len(node.args) > 2 else [])
        values.extend(keyword.value for keyword in node.keywords if keyword.arg == "stages")
        for value in values:
            if not isinstance(value, ast.Dict):
                raise ValueError("Cannot statically verify a dynamic native stage declaration")
            found = True
            names.update(key.value for key in value.keys if isinstance(key, ast.Constant))
    if not found:
        raise ValueError("Native stage declaration is missing a verifiable stages mapping")
    return names


def _visible_method(classes: dict, codes: dict, class_name: str, method_name: str):
    methods = {code.co_name: code for code in codes[class_name].co_consts if isinstance(code, CodeType)}
    if method_name in methods:
        return methods[method_name]
    for base in classes[class_name].bases:
        if isinstance(base, ast.Name) and base.id in classes:
            result = _visible_method(classes, codes, base.id, method_name)
            if result is not None:
                return result
    return None


def audit_bundle(root: Path, required_methods: dict | None = None) -> dict[str, str]:
    """Require complete, readable Python source, a diff and a package entry point."""
    files = inventory(root)
    assert "__init__.py" in files, f"Missing package entry point: {root}"
    models = list(root.glob("modeling_*_gen_npu.py"))
    assert models and all(Path(str(path) + ".diff").is_file() for path in models), str(root)
    classes = 0
    remaining = dict(required_methods or {})
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        classes += sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        _audit_visible_methods(path, text, tree, remaining)
        assert "__hp_prepared_components__" not in text, str(path)
        assert "_hp_module_cache" not in text, str(path)
        assert "_hp_module_constructors" not in text, str(path)
    assert classes, "A complete model must contain class definitions"
    assert not any(name.endswith("_component_replay.json") for name in files)
    assert not remaining, (root, "Missing native stage classes", sorted(remaining))
    return files


def _audit_visible_methods(path: Path, text: str, tree: ast.Module, remaining: dict) -> None:
    """Require the selected native bodies to be locally defined, including inherited methods."""
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    assert len(definitions) == sum(isinstance(node, ast.ClassDef) for node in tree.body), str(path)
    assert not any(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                   for node in ast.walk(tree)), str(path)
    codes = {code.co_name: code for code in compile(text, str(path), "exec").co_consts
             if isinstance(code, CodeType)}
    _audit_imported_methods(path, tree, remaining, definitions)
    for class_name in sorted(remaining.keys() & definitions.keys()):
        host = definitions[class_name]
        native = None
        if len(host.bases) == 1 and isinstance(host.bases[0], ast.Attribute):
            base = ast.unparse(host.bases[0])
            if base.startswith('hyper_parallel.components.modules.'):
                owner, name = base.rsplit('.', 1)
                native = getattr(importlib.import_module(owner), name)
        for method_name, expected in remaining[class_name].items():
            _audit_defined_method(path, definitions, codes, class_name, method_name, expected, native)
        remaining.pop(class_name)



def _audit_imported_methods(path, tree, remaining, definitions):
    """Check imported components against native method fingerprints."""
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith('hyper_parallel.components.modules.'):
            continue
        owner = importlib.import_module(node.module)
        for item in node.names:
            name = item.asname or item.name
            if name not in remaining or name in definitions:
                continue
            native = getattr(owner, item.name)
            for method_name, expected in remaining[name].items():
                assert code_fingerprint(getattr(native, method_name).__code__) == expected, (path, name, method_name)
            remaining.pop(name)


def _audit_defined_method(path, definitions, codes, class_name, method_name, expected, native):
    """Validate a materialized stage or its explicit native delegation."""
    method = _visible_method(definitions, codes, class_name, method_name)
    if native is not None:
        assert code_fingerprint(getattr(native, method_name).__code__) == expected
        if method is None:
            return
        if method_name == 'forward':
            assert '_hp_tp' in method.co_names or '_hp_boundary' in method.co_names
            core = _visible_method(definitions, codes, class_name, '_forward_impl')
            assert core is not None and 'super' in core.co_names
            return
    assert method is not None, (path, class_name, method_name, "missing local method")
    if method_name == "forward" and "_hp_tp" in method.co_names:
        assert "_forward_impl" in method.co_names, (path, class_name, "missing TP core call")
        method = _visible_method(definitions, codes, class_name, "_forward_impl")
        assert method is not None, (path, class_name, "missing local TP core")
    assert code_fingerprint(method) == expected, (path, class_name, method_name, "native body differs")


def environment(optional_path: str = "") -> dict[str, str]:
    """Return a clean single-process parent environment without changing os.environ."""
    result = dict(os.environ)
    for name in DISTRIBUTED_ENV:
        result.pop(name, None)
    result.update(WORLD_SIZE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result["PYTHONPATH"] = os.pathsep.join(part for part in (str(Path(__file__).resolve().parent), optional_path)
                                         if part)
    return result


def stop_group(process: subprocess.Popen) -> None:
    """Stop only the process group created for this acceptance command."""
    if os.name == "posix":
        try:
            getattr(os, "killpg")(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))
        else:
            process.kill()
        process.wait(timeout=10)


def command(argv: list[str], cwd: Path, env: dict[str, str], log: Path, timeout: int) -> int:
    """Run an isolated command and preserve native crashes as nonzero results."""
    if any(Path(argument).name == "train_bundle.py" for argument in argv):
        while True:
            occupied = npu_processes(subprocess.check_output([shutil.which("npu-smi") or "npu-smi", "info"], text=True))
            dump(log.with_suffix(".availability.json"),
                 {"case": str(log), "time": time.time(), "occupied": occupied, "interval": 60})
            if not occupied:
                break
            print(json.dumps({"status": "WAITING_FOR_IDLE", "case": str(log), "occupied": occupied}), flush=True)
            time.sleep(60)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        with subprocess.Popen(argv, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                return process.wait(timeout=timeout)
            finally:
                stop_group(process)


def wait_event(path: Path, kind: str, timeout: int = 60) -> None:
    """Wait for a recorded event, rather than assuming a timing delay proves overlap."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and any(json.loads(line)["event"] == kind
                                 for line in path.read_text(encoding="utf-8").splitlines() if line):
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {kind}: {path}")


def compare_losses(left: list[dict], right: list[dict], steps: int, atol: float, rtol: float) -> None:
    """Compare all native optimizer-step losses; missing steps and NaNs always fail."""
    assert steps > 0 and atol >= 0 and rtol >= 0
    assert len(left) == len(right) == steps, (len(left), len(right), steps)
    for index, (before, after) in enumerate(zip(left, right), start=1):
        assert before["step"] == after["step"] == index
        a, b = float(before["loss"]), float(after["loss"])
        assert math.isfinite(a) and math.isfinite(b), (index, a, b)
        assert abs(a - b) <= atol + rtol * abs(a), (index, a, b, atol, rtol)


def source_receipt(source: Path) -> dict:
    """Verify a test-only export receipt when the isolated server source has no Git metadata."""
    path = source / ".codegen-source-receipt.json"
    if not path.is_file():
        return {}
    receipt = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in receipt["files"].items():
        candidate = (source / name).resolve()
        if source.resolve() not in candidate.parents or not candidate.is_file():
            raise Blocked(f"Exported source file is missing or outside its root: {name}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise Blocked(f"Exported source differs from its recorded commit: {name}")
    extras = [str(path.relative_to(source)) for path in (source / "hyper_parallel").rglob("*.py")
              if str(path.relative_to(source)).replace(os.sep, "/") not in receipt["files"]]
    if extras:
        raise Blocked(f"Untracked Python modules remain in the isolated source: {extras[:5]}")
    return receipt


def assert_forward_coverage(row: dict, steps: int) -> None:
    """A native wrapper may delegate, but it cannot replace any required artifact body."""
    assert row['forward_inventory'], 'No artifact forwards were inventoried'
    for key, entry in row['forward_inventory'].items():
        fqn = key.rsplit('.', 1)[0]
        calls = row['forward_calls'].get(key, 0)
        observed = len(row['module_outputs'].get(fqn, []))
        assert calls >= observed >= steps, (key, calls, observed)
        for operation in set(entry['tp_operations']):
            expected = observed * entry['tp_operations'].count(operation)
            assert row['tp_calls'].get(fqn + ':' + operation, 0) >= expected, (fqn, operation, expected)
    for fqn in row['boundary_modules']:
        assert row['module_outputs'].get(fqn), ('Unexecuted plan boundary', fqn)
    assert len(row['gradients']) == steps


def numerical_differences(left: dict, right: dict) -> list[dict]:
    """Identify exact output/logit/gradient differences separately from provenance checks."""
    result = []
    for category in ('module_outputs',):
        before, after = left[category], right[category]
        for fqn in sorted(set(before) | set(after)):
            if before.get(fqn) != after.get(fqn):
                result.append({'category': category, 'module': fqn})
    if left['gradients'] != right['gradients']:
        result.append({'category': 'optimizer_gradients'})
    return result
