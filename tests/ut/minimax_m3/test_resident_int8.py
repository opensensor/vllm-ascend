# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.minimax_m3.resident_int8 import convert_banks, reorder_codes
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts


@pytest.mark.parametrize("experts,chunk", [(1, 8), (8, 3), (17, 8)])
def test_resident_layout_round_trip_is_lossless_across_chunk_boundaries(experts, chunk):
    codes = torch.randint(-128, 128, (experts, 32, 256), dtype=torch.int8)
    original = codes.clone()
    converted = reorder_codes(codes, chunk=chunk)
    assert torch.equal(codes, original)
    assert torch.equal(reorder_codes(converted, inverse=True, chunk=chunk), original)


@pytest.mark.parametrize("fail", [False, True])
def test_partial_and_complete_conversions_roll_back_exactly(monkeypatch, fail):
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(mem_get_info=lambda: (2**30, 2**30), memory_reserved=lambda: 0, memory_allocated=lambda: 0),
        raising=False,
    )
    bank = PackedM3Experts(256, 256, 2, 1, 0, 2, prefix="layer")
    original = {}
    for projection in ("w1", "w3", "w2"):
        codes = torch.randint(-128, 128, (2, 256, 128), dtype=torch.int8)
        setattr(bank, projection + "_codes", codes)
        original[projection] = codes.clone()

    class Kernel:
        requires_int8_layout = True

        def __init__(self, bad=False):
            self.banks, self.scales = {}, {}
            self.bad = bad

        def __call__(self, x, codes, scales, ends):
            return torch.full((len(x), codes.shape[1]), float("nan") if self.bad else 1.0)

    baseline, candidate = Kernel(), Kernel(fail)
    if fail:
        with pytest.raises(RuntimeError, match="comparison failed"):
            convert_banks(bank, baseline, candidate, True)
        assert bank._m3_int8_layouts == {"w1"}
    else:
        assert convert_banks(bank, baseline, candidate, True)["passed"]
        assert bank._m3_int8_layouts == {"w1", "w3", "w2"}
    assert convert_banks(bank, baseline, candidate, False)["passed"]
    assert not bank._m3_int8_layouts
    for projection, expected in original.items():
        assert torch.equal(getattr(bank, projection + "_codes"), expected)
