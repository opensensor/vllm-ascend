# SPDX-License-Identifier: Apache-2.0
"""Protect active graph buffers, weights and unknown resources during cleanup."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.glm_fused_moe import NativeFusedMoE
from tools.glm_perf.transient_projection import release_projection_capture_buffers


def projection():
    operation = NativeFusedMoE.__new__(NativeFusedMoE)
    operation.scratch = {"large": torch.empty(1024)}
    operation.configs = {"shape": torch.arange(8)}
    operation.weights = torch.ones(4)
    operation.kernel = object()
    return operation


def test_scratch_released_once_without_touching_active_descriptors_or_weights():
    active, inactive = projection(), projection()
    active_config = active.configs["shape"]
    weight, kernel = inactive.weights, inactive.kernel
    released = weakref.ref(inactive.scratch["large"])
    other = SimpleNamespace(scratch={"keep": object()}, configs={"keep": object()})
    resources = {"one": {"a4": active, "a8": inactive}, "alias": [inactive, other]}
    resources["cycle"] = resources
    counts = release_projection_capture_buffers(resources, (active,))
    assert counts == dict(operators=2, scratch_entries=2, config_entries=1)
    assert released() is None and not inactive.scratch and not inactive.configs
    assert inactive.weights is weight and inactive.kernel is kernel
    assert not active.scratch and active.configs["shape"] is active_config
    assert other.scratch and other.configs


def test_unknown_projection_buffer_contract_fails_before_partial_clear():
    operation = projection()
    operation.configs = []
    with pytest.raises(ValueError, match="contract changed"):
        release_projection_capture_buffers({"candidate": operation}, ())
    assert operation.scratch
