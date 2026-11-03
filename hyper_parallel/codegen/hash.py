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
"""Signature and file-hash helpers for codegen artifacts.

Snapshot signatures are the single source of truth for artifact reuse.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_file(path: str) -> str:
    """Compute the sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Compute the sha256 of a UTF-8 text payload."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(data: object) -> str:
    """Stable JSON serialization (sorted keys, no whitespace).

    Used for signature inputs — key order must not affect the digest.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_snapshot_payload(snapshot: Any) -> str:
    """Serialize the stable identity projection of a build snapshot."""
    signature_payload = getattr(snapshot, "signature_payload", None)
    if not callable(signature_payload):
        raise TypeError("snapshot must provide signature_payload()")
    return canonical_json(signature_payload())


def signature_from_snapshot(snapshot: Any, *, prefix: int = 12) -> str:
    """Compute a short signature from a build snapshot's stable facts."""
    return sha256_text(canonical_snapshot_payload(snapshot))[:prefix]


def file_line_count(path: str) -> int:
    """Count lines in a text file."""
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


__all__ = [
    "canonical_json",
    "canonical_snapshot_payload",
    "file_line_count",
    "sha256_file",
    "sha256_text",
    "signature_from_snapshot",
]
