# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for the native-training Codegen sidecar."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

from hyper_parallel.codegen import train_sidecar


class TestTrainSidecar(unittest.TestCase):
    """Verify rank selection, environment isolation, and failure handling."""

    @patch.object(train_sidecar.subprocess, "Popen")
    def test_nonzero_rank_does_not_start_sidecar(self, popen: MagicMock) -> None:
        """Only global rank zero may publish a generated bundle."""
        with patch.dict(os.environ, {"RANK": "3"}, clear=True):
            process = train_sidecar.start_codegen_sidecar("train.yaml")

        self.assertIsNone(process)
        popen.assert_not_called()

    @patch.object(train_sidecar.threading, "Thread")
    @patch.object(train_sidecar.subprocess, "Popen")
    def test_rank_zero_starts_isolated_codegen_process(
        self,
        popen: MagicMock,
        thread: MagicMock,
    ) -> None:
        """Rank zero launches the CLI without changing the parent environment."""
        child = MagicMock(pid=1234)
        popen.return_value = child
        config_path = Path("recipes/train.yaml")
        distributed_env = {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "8",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "HP_TEST_VALUE": "kept",
        }

        with patch.dict(os.environ, distributed_env, clear=True):
            process = train_sidecar.start_codegen_sidecar(
                config_path,
                ["--model.config_path=assets/model"],
            )
            parent_after_start = dict(os.environ)

        self.assertIs(process, child)
        self.assertEqual(parent_after_start, distributed_env)
        command = popen.call_args.args[0]
        child_env = popen.call_args.kwargs["env"]
        self.assertEqual(
            command[:4],
            [sys.executable, "-m", "hyper_parallel.codegen", "--verbose"],
        )
        self.assertEqual(command[-3], "--config")
        self.assertEqual(command[-2], str(config_path.resolve()))
        self.assertEqual(command[-1], "--model.config_path=assets/model")
        self.assertEqual(child_env["HP_TEST_VALUE"], "kept")
        for name in train_sidecar._DISTRIBUTED_ENV_NAMES:  # pylint: disable=protected-access
            self.assertNotIn(name, child_env)
        thread.return_value.start.assert_called_once_with()

    @patch.object(train_sidecar.subprocess, "Popen", side_effect=OSError("unavailable"))
    def test_start_failure_does_not_escape(self, _popen: MagicMock) -> None:
        """A child startup error is reported without failing native training."""
        with patch.dict(os.environ, {"RANK": "0"}, clear=True), self.assertLogs(
            train_sidecar.logger,
            level="WARNING",
        ) as captured:
            process = train_sidecar.start_codegen_sidecar("train.yaml")

        self.assertIsNone(process)
        self.assertIn("native training continues", captured.output[0])

    def test_monitor_reports_generation_failure(self) -> None:
        """A nonzero Codegen exit remains a warning instead of a Trainer error."""
        process = MagicMock()
        process.wait.return_value = 7

        with self.assertLogs(train_sidecar.logger, level="WARNING") as captured:
            train_sidecar._monitor_sidecar(process, "train.yaml")  # pylint: disable=protected-access

        self.assertIn("exit code 7", captured.output[0])
        self.assertIn("native training continues", captured.output[0])

    def test_native_training_entry_starts_codegen_with_same_arguments(self) -> None:
        """The standard training entry starts rank-filtered generation first."""
        config = MagicMock()
        config._yaml_path = "/recipes/train.yaml"  # pylint: disable=protected-access
        parse_args = MagicMock(return_value=config)
        trainer_type = MagicMock()
        parser_module = ModuleType("hyper_parallel.trainer.config.parser")
        parser_module.parse_training_args = parse_args
        config_module = ModuleType("hyper_parallel.trainer.config")
        config_module.TrainerConfig = object
        trainer_module = ModuleType("hyper_parallel.trainer.text_trainer")
        trainer_module.TextTrainer = trainer_type
        script_path = Path(__file__).parents[3] / "scripts" / "train_lm.py"
        spec = importlib.util.spec_from_file_location("_codegen_test_train_lm", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        overrides = ["--accelerator.tp_size=2", "--training.train_iters=10"]

        with patch.dict(
            sys.modules,
            {
                "hyper_parallel.trainer.config.parser": parser_module,
                "hyper_parallel.trainer.config": config_module,
                "hyper_parallel.trainer.text_trainer": trainer_module,
            },
        ):
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        with patch.object(module, "start_codegen_sidecar") as start_sidecar, patch.object(
            sys, "argv", ["scripts/train_lm.py", "train.yaml", *overrides]
        ):
            module.main()

        start_sidecar.assert_called_once_with(config._yaml_path, overrides)  # pylint: disable=protected-access
        trainer_type.assert_called_once_with(config)
        trainer_type.return_value.train.assert_called_once_with()

    @patch.object(train_sidecar.os, "execv")
    @patch.object(train_sidecar, "start_codegen_sidecar")
    def test_launcher_forwards_identical_arguments_before_native_exec(
        self,
        start_sidecar: MagicMock,
        execv: MagicMock,
    ) -> None:
        """The codegen-owned launcher preserves Trainer CLI overrides."""
        overrides = ["--model.config_path=assets/model", "--training.train_iters=10"]
        with patch.object(Path, "is_file", return_value=True):
            train_sidecar.launch_training_with_codegen(
                "recipes/train.yaml",
                overrides,
                trainer_entry="scripts/train_lm.py",
            )

        resolved_config = str(Path("recipes/train.yaml").resolve())
        resolved_entry = str(Path("scripts/train_lm.py").resolve())
        start_sidecar.assert_called_once_with(resolved_config, overrides)
        execv.assert_called_once_with(
            sys.executable,
            [sys.executable, resolved_entry, resolved_config, *overrides],
        )


if __name__ == "__main__":
    unittest.main()
