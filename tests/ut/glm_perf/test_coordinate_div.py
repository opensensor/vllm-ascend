# SPDX-License-Identifier: Apache-2.0
"""Coordinate rewrites preserve ordinary arithmetic and isolate dependencies."""

import pytest
import torch

from tools.glm_perf.coordinate_div import NativeCoordinateDivision, rewrite_floor_division

SOURCE = """def coordinates(value, divisor):
    return torch.div(value, divisor, rounding_mode="floor"), value // divisor
"""


def test_rewrite_binds_dependencies_without_changing_original_globals():
    namespace = {"torch": torch}
    exec(SOURCE, namespace)
    original = namespace["coordinates"]
    before = dict(original.__globals__)
    calls = []

    def division(value, divisor):
        calls.append((value, divisor))
        return torch.div(value, divisor, rounding_mode="floor")

    changed = rewrite_floor_division(original, SOURCE, division)
    value = torch.tensor([-641, -640, -1, 0, 1, 639, 640, 641], dtype=torch.int64)
    assert all(torch.equal(a, b) for a, b in zip(original(value, 640), changed(value, 640)))
    assert len(calls) == 2
    assert original.__globals__ == before
    assert changed.__glm_resident_original__ is original


def test_native_coordinates_preserve_cpu_dtype_and_negative_floor():
    native = object.__new__(NativeCoordinateDivision)
    value = torch.tensor([-641, -1, 0, 641], dtype=torch.int64)
    assert torch.equal(native(value, 640), torch.tensor([-2, -1, 0, 1]))
    assert native(-641, 640) == -2


def test_rewrite_rejects_changed_function_before_mutating_globals():
    namespace = {"torch": torch}
    source = "def coordinates(value, divisor):\n    return value / divisor\n"
    exec(source, namespace)
    with pytest.raises(ValueError, match="no floor divisions"):
        rewrite_floor_division(namespace["coordinates"], source, lambda a, b: a // b)


def test_rewrite_rejects_nested_definition_scope():
    source = (
        "def coordinates(value, divisor):\n"
        "    def inner(value):\n"
        "        return value // divisor\n"
        "    return inner(value)\n"
    )
    namespace = {}
    exec(source, namespace)
    with pytest.raises(ValueError, match="nested definitions"):
        rewrite_floor_division(namespace["coordinates"], source, lambda a, b: a // b)
