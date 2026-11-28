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
"""Torchrun worker for native or generated-model training using the same Trainer YAML."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import runpy
import sys

from artifact_training import artifact_model_build
from common import dump
from forward_range import FullForwardEvidence as TrainingEvidence

from hyper_parallel.trainer.config.parser import load_training_config


def main() -> None:
    """Select only the model source; all training settings remain in the native YAML parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "generated"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--entry", default="scripts/train_lm.py")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--state-action", choices=("record", "replay"), required=True)
    args, overrides = parser.parse_known_args()
    config = load_training_config(str(args.config), overrides)
    if config.training.seed is None or not config.training.enable_full_determinism:
        raise ValueError("YAML/overrides must set training.seed and training.enable_full_determinism=true")
    if config.compile.enabled or config.accelerator.pp_size > 1:
        raise ValueError("The initial artifact comparison supports eager training and pp_size=1")
    if config.dataloader.num_workers or config.dataloader.use_background_prefetcher:
        raise ValueError("Use dataloader.num_workers=0 and use_background_prefetcher=false for fixed-input comparison")
    if args.mode == "generated" and (args.bundle is None or args.state_action != "replay"):
        raise ValueError("Generated mode requires --bundle and a recorded native initial state to replay")
    report = {"mode": args.mode, "config": str(args.config.resolve()), "overrides": overrides,
              "seed": config.training.seed}
    try:
        with ExitStack() as stack:
            generated = None
            if args.mode == "generated":
                generated = stack.enter_context(artifact_model_build(args.bundle, report))
            stack.enter_context(TrainingEvidence(args.evidence, args.initial_state, args.state_action, generated))
            worker = Path(__file__).with_name("native_worker.py")
            sys.argv = [str(worker), "--source", str(args.source), "--entry", str(args.source / args.entry),
                        "--yaml", str(args.config), "--output", str(args.evidence / "sidecar-bundle"),
                        "--evidence", str(args.evidence / "native"), *overrides]
            runpy.run_path(str(worker), run_name="__main__")
    finally:
        import_rank = os.environ.get("RANK", "0")
        dump(args.evidence / f"model-source-rank-{import_rank}.json", report)


if __name__ == "__main__":
    main()
