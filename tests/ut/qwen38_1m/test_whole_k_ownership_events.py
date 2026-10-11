# SPDX-License-Identifier: Apache-2.0
"""Removing vector work must preserve the qualified operand ownership joins."""

import json
from pathlib import Path

import regex as re


def test_lean_recipe_preserves_existing_hardware_buffer_ownership_sequence():
    source = Path("tools/qwen4exp/native_int8_projection_probe.cpp").read_text()
    run = source[source.index("__aicore__ inline void Run(") : source.index("\n private:")]
    events = re.findall(r"(SetFlag|WaitFlag)<HardEvent::([^>]+)>\(([^)]+)\)", run)
    # This baseline is from 40039ae16's hardware-qualified original recipe.
    # Buffer reuse joins protect L1 operands, L0C readback and UB output stores;
    # changing them needs its own ownership proof and hardware qualification.
    baseline = json.loads(Path("tests/ut/qwen38_1m/reference/whole_k_events.json").read_text())
    assert [list(event) for event in events] == baseline
