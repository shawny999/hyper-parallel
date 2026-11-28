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
"""Generate once, then compare native/native-repeat/generated training from one YAML."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from common import audit_bundle, command, compare_losses, dump, environment, inventory


PACK = Path(__file__).resolve().parent


def read_run(directory: Path, world: int, steps: int) -> tuple[dict, dict, dict]:
    """Require all rank exits, actual optimizer progress and independent model-source evidence."""
    native, observed, sources = {}, {}, {}
    for rank in range(world):
        native[rank] = json.loads((directory / "native" / f"rank-{rank}.json").read_text(encoding="utf-8"))
        observed[rank] = json.loads((directory / f"comparison-rank-{rank}.json").read_text(encoding="utf-8"))
        sources[rank] = json.loads((directory / f"model-source-rank-{rank}.json").read_text(encoding="utf-8"))
        row = native[rank]
        assert row["native_completed"] and row["optimizer_steps"] == len(row["steps"]) == steps
        assert row["child_exits"] == ([0] if rank == 0 else [])
        assert row["distributed_identity_unchanged"]
    return native, observed, sources


def verify_runs(root: Path, world: int, steps: int, atol: float, rtol: float) -> dict:
    """A matching loss is insufficient without identical starting state/inputs and artifact execution."""
    reference = read_run(root / "native", world, steps)
    results = []
    for label in ("native-repeat", "generated"):
        native, observed, sources = read_run(root / label, world, steps)
        for rank in range(world):
            before, after = reference[1][rank], observed[rank]
            assert before["initial_schema"] == after["initial_schema"], (label, rank, "initial schema")
            assert before["initial_hashes"] == after["initial_hashes"], (label, rank, "initial weights")
            assert before["inputs"] and before["inputs"] == after["inputs"], (label, rank, "inputs or RNG streams")
            assert reference[0][rank]["torch_version"] == native[rank]["torch_version"]
            compare_losses(reference[0][rank]["steps"], native[rank]["steps"], steps, atol, rtol)
            if atol == rtol == 0:
                assert before["final_hashes"] == after["final_hashes"], (label, rank, "final parameter updates")
            if label == "generated":
                assert sources[rank].get("constructed_from_artifact"), (rank, "native fallback")
                assert sum(after["artifact_forward_calls"].values()) >= steps, (rank, "artifact forward never executed")
                expected = after["artifact_stage_expectations"]
                assert not after["native_stage_fallback_calls"], (rank, "stage fell back to native source")
                for stage in expected:
                    assert after["artifact_stage_calls"].get(stage, 0) >= steps, (
                        rank, "artifact stage never executed", stage)
                generation = root / "generation.json"
                required = json.loads(generation.read_text())["required_methods"] if generation.exists() else {}
                assert set(required) <= {entry["class"] for entry in expected.values()}, (
                    rank, "missing parallel stages")
        results.append({"mode": label, "ranks": world, "steps": steps, "status": "PASS"})
    return {"status": "PASS", "comparisons": results, "atol": atol, "rtol": rtol}


def main() -> int:
    """Keep all model, parallelism, data and optimizer settings in a shared YAML plus common overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--entry", default="scripts/train_lm.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--optional-pythonpath", default=os.environ.get("PYTHONPATH", ""))
    args, overrides = parser.parse_known_args()
    if args.steps < 1 or args.nproc_per_node < 1:
        parser.error("--steps and --nproc-per-node must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env = environment(args.optional_pythonpath)
    env.update(PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=env.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
               TOKENIZERS_PARALLELISM="false")
    options = [*overrides, f"--training.train_iters={args.steps}"]
    bundle = args.output / "bundle"
    source, config = args.source.resolve(), args.config.resolve()
    common = ["--source", str(source), "--config", str(config), "--entry", args.entry,
              "--initial-state", str(args.output / "initial-state")]
    commands = []
    report = {"status": "FAIL", "config": str(config), "overrides": options, "commands": commands}
    try:
        generate = [sys.executable, str(PACK / "generation_worker.py"), "--source", str(source),
                    "--yaml", str(config), "--output", str(bundle), "--evidence", str(args.output / "generation.json"),
                    *options]
        code = command(generate, source, env, args.output / "generation.log", args.timeout)
        commands.append({"argv": generate, "exit": code})
        assert code == 0, "Generation failed; inspect generation.log"
        expected_files = audit_bundle(bundle)
        for label, mode, action in (("native", "native", "record"), ("native-repeat", "native", "replay"),
                                    ("generated", "generated", "replay")):
            folder = args.output / label
            folder.mkdir()
            argv = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                    f"--nproc-per-node={args.nproc_per_node}", str(PACK / "train_bundle.py"),
                    *common, "--mode", mode, "--state-action", action, "--evidence", str(folder), *options]
            if mode == "generated":
                argv.extend(["--bundle", str(bundle)])
            code = command(argv, source, env, folder / "torchrun.log", args.timeout)
            commands.append({"argv": argv, "exit": code})
            assert code == 0, f"{label} failed, including possible native abort on exit; inspect {folder}"
            read_run(folder, args.nproc_per_node, args.steps)
            if label == "native-repeat":
                reference = read_run(args.output / "native", args.nproc_per_node, args.steps)
                repeated = read_run(folder, args.nproc_per_node, args.steps)
                for rank in range(args.nproc_per_node):
                    for field in ("initial_schema", "initial_hashes", "inputs", "final_hashes"):
                        assert reference[1][rank][field] == repeated[1][rank][field], (
                            "NATIVE_REPEAT_NOT_DETERMINISTIC", rank, field)
                    compare_losses(reference[0][rank]["steps"], repeated[0][rank]["steps"], args.steps, 0, 0)
                dump(args.output / "native-repeat-gate.json", {"status": "PASS", "before_generated": True,
                     "ranks": args.nproc_per_node, "steps": args.steps})
        report.update(verify_runs(args.output, args.nproc_per_node, args.steps, args.atol, args.rtol))
        assert inventory(bundle) == expected_files, "Training rewrote the generated model under test"
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        dump(args.output / "comparison-report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
