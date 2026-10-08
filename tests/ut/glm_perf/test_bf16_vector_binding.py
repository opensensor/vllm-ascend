# SPDX-License-Identifier: Apache-2.0
"""Shared permanent converter bindings retain callers and restore on transition."""

from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.instance_bindings import InstanceBindings
from tools.glm_perf.resident_candidates.bf16_vector import bind_converters


class ScalarConverter:
    def __init__(self):
        self.configs = {(torch.device("cpu"), 7, 4): object()}
        self.calls = []

    def _convert(self, source, dtype, mode):
        self.calls.append(mode)
        return source.to(dtype)


class VectorConverter:
    def __init__(self):
        self.configs, self.calls = {}, []

    def prepare_counts(self, counts):
        assert all(type(count) is int and count > 0 for count in counts)
        self.configs.update(((count, mode), None) for count in counts for mode in (0, 1, 4, 5))

    def convert(self, source, dtype, mode):
        self.calls.append(mode)
        return source.to(dtype)


def runner_for(converter):
    modules = [
        SimpleNamespace(_native_bf16_cast=converter, indexer_op=object(), rope_dim=64, head_dim=128, n_head=32)
        for _ in range(2)
    ]
    return SimpleNamespace(
        model=SimpleNamespace(modules=lambda: modules[:1]),
        drafter=SimpleNamespace(model=SimpleNamespace(modules=lambda: modules[1:])),
    )


def test_shared_converter_used_by_existing_closures_is_bound_once_and_restored():
    scalar, vector, bindings = ScalarConverter(), VectorConverter(), InstanceBindings()
    caller = lambda source, dtype, mode: scalar._convert(source, dtype, mode)
    runner = runner_for(scalar)
    originals = scalar.configs
    assert bind_converters(bindings, runner, vector) == 1
    assert len(bindings.originals) == 1 and scalar.configs is originals
    caller(torch.ones(7), torch.float32, 4)
    caller(torch.ones(7), torch.float16, 5)
    caller(torch.ones(7).half(), torch.bfloat16, 2)
    # Shapes outside the prepared host inventory keep the permanent path.
    caller(torch.ones(9), torch.float32, 4)
    assert vector.calls == [4, 5] and scalar.calls == [2, 4]
    assert (640 * 32 * 128, 1) in vector.configs
    bindings.restore()
    assert "_convert" not in scalar.__dict__
    caller(torch.ones(7), torch.float32, 4)
    assert scalar.calls == [2, 4, 4]


def test_failed_binding_restores_all_prior_instance_methods():
    scalar, vector, bindings = ScalarConverter(), VectorConverter(), InstanceBindings()
    bindings.bind(scalar, "_convert", lambda *args: None)
    with pytest.raises(ValueError, match="already bound"):
        bind_converters(bindings, runner_for(scalar), vector)
    assert not bindings.originals and "_convert" not in scalar.__dict__


def test_no_converter_rejected_before_preparation():
    runner = SimpleNamespace(model=SimpleNamespace(modules=lambda: []))
    with pytest.raises(ValueError, match="no permanent"):
        bind_converters(InstanceBindings(), runner, None)


def test_zero_rope_width_and_empty_legacy_descriptors_do_not_require_a_launch():
    scalar, vector, bindings = ScalarConverter(), VectorConverter(), InstanceBindings()
    scalar.configs[torch.device("cpu"), 0, 1] = object()
    runner = runner_for(scalar)
    for root in (runner.model, runner.drafter.model):
        for module in root.modules():
            module.rope_dim = 0
    assert bind_converters(bindings, runner, vector) == 1
    assert all(count > 0 for count, _ in vector.configs)
    bindings.restore()
