# SPDX-License-Identifier: Apache-2.0
"""Small-shape fusion retains the existing prefill and precision gates."""

from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.resident_candidates.mhc_decode_post import select_native_post


def inputs(rows=2, width=4096):
    return (
        torch.empty(rows, width, dtype=torch.float16),
        torch.empty(rows, 4, width),
        torch.empty(rows, 4, 1),
        torch.empty(rows, 4, 4),
    )


def config(**overrides):
    return SimpleNamespace(**dict(dict(use_310p_native_mhc_post=True, use_310p_fp16_mhc_state=True), **overrides))


def original(*args):
    return False


@pytest.mark.parametrize("rows,expected", [(0, False), (1, True), (2, True), (8, True), (9, False), (640, False)])
def test_only_qualified_decode_rows_added(rows, expected):
    assert select_native_post(original, config(), *inputs(rows)) == expected


def test_existing_prefill_selection_retained():
    assert select_native_post(lambda *args: True, config(), *inputs(640))


@pytest.mark.parametrize("flag", ["use_310p_native_mhc_post", "use_310p_fp16_mhc_state"])
def test_disabled_native_or_fp16_state_falls_back(flag):
    assert not select_native_post(original, config(**{flag: False}), *inputs())


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("problem", ["dtype", "shape", "layout", "device"])
def test_unqualified_tensor_contract_falls_back(index, problem):
    args = list(inputs())
    value = args[index]
    if problem == "dtype":
        args[index] = value.double()
    elif problem == "shape":
        args[index] = value.unsqueeze(0)
    elif problem == "device":
        args[index] = value.to("meta")
    else:
        backing = torch.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype)
        args[index] = backing[..., ::2]
    assert not select_native_post(original, config(), *args)


def test_other_model_width_is_not_enabled():
    assert not select_native_post(original, config(), *inputs(width=2048))
