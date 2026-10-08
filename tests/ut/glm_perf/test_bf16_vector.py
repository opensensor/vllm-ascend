# SPDX-License-Identifier: Apache-2.0
"""Storage precision, descriptor ownership and gate admission on CPU."""

import pytest
import torch

from tools.glm_perf.bf16_vector import MODES, NativeBf16Vector, reference
from tools.glm_perf.bf16_vector_probe import input_bits, run


@pytest.mark.parametrize("mode", MODES)
def test_all_bf16_words_and_rounding_boundaries_match_torch_for_finite_values(mode):
    bits = input_bits(5 * 65536, mode, 0)
    source = bits.view(torch.bfloat16 if mode == 1 else torch.float32)
    result = reference(bits, mode)
    dtype = {0: torch.bfloat16, 1: torch.float32, 4: torch.float32, 5: torch.float16}[mode]
    expected = source.float() if mode == 1 else source.bfloat16().to(dtype)
    finite = ~torch.isnan(source.float())
    assert torch.equal(result[finite], expected.view(result.dtype)[finite])
    if mode == 1:
        assert torch.equal(result, bits.to(torch.int32) << 16), "BF16 NaN payloads must be retained on decode"


@pytest.mark.parametrize("mode,expected", [(0, [0x7FC0, -64]), (4, [0x7FC00000, -4194304]), (5, [0x7E00, -512])])
def test_rounding_preserves_signed_canonical_nan(mode, expected):
    bits = torch.tensor([0x7F800001, -8388607], dtype=torch.int32)
    assert reference(bits, mode).tolist() == expected


@pytest.fixture
def native():
    instance = NativeBf16Vector.__new__(NativeBf16Vector)
    instance.device, instance.configs, instance.calls, instance.kernel = torch.device("cpu"), {}, {}, object()

    def launch(kernel, args, cores):
        value, output, config = args
        count, mode = config.tolist()
        assert kernel is instance.kernel and count == value.numel()
        assert output.numel() * output.element_size() % 32 == 0
        assert cores == min(8, (count + 1023) // 1024)
        bits = value.view(torch.int16 if mode == 1 else torch.int32)
        storage = torch.int32 if mode in (1, 4) else torch.int16
        output.zero_()
        output.view(storage)[:count].copy_(reference(bits, mode))

    instance.launch = launch
    return instance


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("count", [0, 1, 7, 15, 17, 1025])
def test_capture_launch_uses_only_prepared_descriptors_and_owned_padding(native, mode, count, monkeypatch):
    native.prepare_counts([count] if count else [])
    source_dtype = torch.bfloat16 if mode == 1 else torch.float32
    bits = input_bits(count, mode, 1)
    value = bits.view(source_dtype)
    dtype = {0: torch.bfloat16, 1: torch.float32, 4: torch.float32, 5: torch.float16}[mode]
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("descriptor upload during serving"))
    result = native.convert(value, dtype, mode)
    assert torch.equal(result.view(torch.int32 if dtype == torch.float32 else torch.int16), reference(bits, mode))
    assert torch.equal(value.view(bits.dtype), bits)
    assert native.calls == ({mode: 1} if count else {})


def test_unprepared_count_and_wrong_precision_rejected_before_launch(native):
    with pytest.raises(ValueError, match="outside capture"):
        native.convert(torch.ones(1), torch.float32, 4)
    with pytest.raises(ValueError, match="qualified"):
        native.convert(torch.ones(1).half(), torch.float32, 4)
    with pytest.raises(ValueError, match="qualified"):
        native.convert(torch.ones(1), torch.float16, 4)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_counts_rejected(native, count):
    with pytest.raises(ValueError, match="positive integers"):
        native.prepare_counts([count])


def test_hardware_gate_requires_explicit_selection(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        run(tmp_path, tmp_path / "report.json")
