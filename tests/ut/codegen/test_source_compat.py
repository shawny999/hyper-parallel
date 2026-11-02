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
"""Tests for environment-driven remote-source compatibility."""

from transformers.utils import generic

from hyper_parallel.codegen.source.compat import install_output_recorder_compat


def test_output_recorder_compat_installs_only_when_capability_is_missing():
    """Install the descriptor by capability without inspecting a model family."""
    sentinel = object()
    original = getattr(generic, "OutputRecorder", sentinel)
    if original is not sentinel:
        delattr(generic, "OutputRecorder")
    try:
        install_output_recorder_compat()
        installed = generic.OutputRecorder
        recorder = installed(dict, index=2)

        install_output_recorder_compat()

        assert generic.OutputRecorder is installed
        assert recorder.target_class is dict
        assert recorder.index == 2
    finally:
        if original is sentinel:
            delattr(generic, "OutputRecorder")
        else:
            generic.OutputRecorder = original
