# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Recover inline rules from existing YAML-derived codegen metadata.

The user YAML is not extended.  Codegen reads the already frozen
``module_overrides`` and ``injections`` metadata, whose fields originate from
``plan_overrides.replace`` actions. Frozen execution records retain private
module and injection fields for the code generator.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from hyper_parallel.codegen.inline.ir import InlineRule


def collect_inline_rules(meta: Any) -> tuple[InlineRule, ...]:
    """Return normalized inline rules recovered from ``meta``."""

    rules: list[InlineRule] = []
    for record in getattr(meta, "module_overrides", None) or []:
        target = _replacement_target(record)
        if not target:
            continue
        rules.append(
            InlineRule(
                match=_matches(record),
                module_type=record.get("module_type"),
                replace_target=target,
            )
        )
    for record in getattr(meta, "injections", None) or []:
        local = _target_path(record.get("local_compute_fn"))
        inner = _target_path(record.get("inner_wrapper"))
        if local is None and inner is None:
            continue
        rules.append(
            InlineRule(
                match=_matches(record),
                inner_wrapper_target=inner,
                local_compute_target=local,
            )
        )
    return tuple(rules)


def _replacement_target(record: dict[str, Any]) -> Any:
    """Keep a model-owned factory contract; otherwise use the observed type.

    The native observation proves which concrete type was installed, while a
    factory under the model adapter convention also carries source-emission
    facts such as the interface passed into a generic component.  Targets
    outside that convention remain represented by the observed replacement
    type and require no model-specific handling here.
    """
    declared = record.get("replacement_target")
    parts = str(declared or "").split(".")
    if (
        len(parts) >= 4
        and parts[:2] == ["hyper_parallel", "models"]
        and parts[3] == "adapter"
    ):
        return declared
    return record.get("factory")


def _matches(record: dict[str, Any]) -> tuple[str, ...]:
    value = record.get("fqns") or record.get("match") or record.get("fqn")
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def _target_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("_target_") or value.get("target")
    return getattr(value, "__qualname__", None)
