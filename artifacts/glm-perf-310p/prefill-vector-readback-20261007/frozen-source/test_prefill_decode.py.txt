# SPDX-License-Identifier: Apache-2.0
"""Keep decode on its original kernel resource without changing routed tensors."""

from types import SimpleNamespace

import pytest

from tools.glm_perf.resident_candidates.prefill_decode import PrefillDecodeNative


class Native:
    device = "npu:0"
    input_dtype = "half"
    activation_bits = 4
    prepared_weight_layout = True
    fp16_route_workspace = True
    route_workspace_dtype = "half"

    def __init__(self):
        self.scratch = {"owned": object()}
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self

    def grouped(self, *args):
        return self(*args)

    def geometry(self, *args):
        return self(*args)

    def pack_weight_codes(self, *args):
        return self(*args)


@pytest.mark.parametrize("tokens,prefill_selected", [(2, False), (8, False), (16, False), (17, True), (640, True)])
def test_static_boundary_preserves_resources_and_argument_identity(tokens, prefill_selected):
    prefill, decode = Native(), Native()
    selected = prefill if prefill_selected else decode
    other = decode if prefill_selected else prefill
    dispatch = PrefillDecodeNative(prefill, decode)
    x, routes = SimpleNamespace(shape=(tokens, 4096)), object()
    assert dispatch.geometry(x, routes) is selected
    assert dispatch(x, routes, expert_offset=72) is selected
    assert selected.calls[-1] == ((x, routes), {"expert_offset": 72})
    geometry = SimpleNamespace(tokens=tokens)
    assert dispatch.grouped(x, routes, geometry) is selected
    assert selected.calls[-1][0] == (x, routes, geometry)
    assert other.calls == []
    assert dispatch.dispatches == {"prefill": 2 if prefill_selected else 0, "decode": 0 if prefill_selected else 2}
    assert dispatch.scratch is prefill.scratch
    assert dispatch.pack_weight_codes(routes, 3) is prefill
    assert decode.scratch and prefill.scratch


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("device", "npu:1"),
        ("input_dtype", "float"),
        ("activation_bits", 8),
        ("prepared_weight_layout", False),
        ("fp16_route_workspace", False),
    ],
)
def test_incompatible_native_resources_rejected_before_dispatch(attribute, value):
    prefill, decode = Native(), Native()
    setattr(decode, attribute, value)
    with pytest.raises(ValueError, match=attribute):
        PrefillDecodeNative(prefill, decode)
    assert prefill.calls == decode.calls == []
