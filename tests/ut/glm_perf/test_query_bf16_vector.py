# SPDX-License-Identifier: Apache-2.0
"""Exhaustive conversion math and launch ownership, using CPU tensors only."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.build_query_bf16_vector import build
from tools.glm_perf.query_bf16_vector import NativeQueryVector, fp16_bits_to_bf16
from tools.glm_perf.query_bf16_vector_probe import run, verify


def test_all_non_nan_fp16_patterns_match_torch_including_subnormals_and_ties():
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.float16)
    actual = fp16_bits_to_bf16(bits)
    not_nan = ~torch.isnan(values)
    assert not_nan.sum() == 63490
    assert torch.equal(actual[not_nan], values.bfloat16().view(torch.int16)[not_nan])


def test_nan_policy_preserves_the_existing_scalar_converters_sign():
    bits = torch.tensor([0x7C01, 0x7FFF, -1023, -1], dtype=torch.int16)
    assert fp16_bits_to_bf16(bits).tolist() == [0x7FC0, 0x7FC0, -64, -64]


@pytest.mark.parametrize("value", [torch.empty(1, dtype=torch.int32), torch.empty(1, dtype=torch.int16, device="meta")])
def test_oracle_rejects_non_cpu_storage(value):
    with pytest.raises(ValueError, match="CPU INT16"):
        fp16_bits_to_bf16(value)


@pytest.fixture
def native():
    helper = NativeQueryVector.__new__(NativeQueryVector)
    helper.device, helper.configs, helper.kernel = torch.device("cpu"), {}, object()
    helper.receipts = []

    def launch(kernel, args, blocks):
        source, backing, config = args
        count = int(config[0])
        assert kernel is helper.kernel and source.is_contiguous() and backing.numel() % 16 == 0
        assert count == source.numel() and blocks == min(8, (count + 1023) // 1024)
        backing.zero_()
        backing[:count].copy_(source.bfloat16())
        helper.receipts.append(count)

    helper.launch = launch
    return helper


@pytest.mark.parametrize("count", [0, 1, 15, 16, 17, 1023, 1024, 1025, 8192, 32768])
def test_launch_owns_dma_padding_and_consumes_prepared_descriptors(native, count, monkeypatch):
    if count:
        native.prepare_counts([count])
    backing = torch.arange(count + 32, dtype=torch.float32).half()
    value = backing[16 : 16 + count]
    before = backing.clone()
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: pytest.fail("serving-time descriptor transfer"))
    result = native(value)
    assert result.shape == value.shape and torch.equal(result.view(torch.int16), value.bfloat16().view(torch.int16))
    assert torch.equal(backing, before)
    padded = (count + 15) // 16 * 16
    assert not torch.any(result.as_strided((padded,), (1,))[count:])
    assert native.receipts == ([count] if count else [])


def test_unprepared_or_unqualified_inputs_fail_before_launch(native):
    with pytest.raises(ValueError, match="prepared outside"):
        native(torch.empty(2, dtype=torch.float16))
    with pytest.raises(ValueError, match="contiguous FP16"):
        native(torch.empty(2, dtype=torch.float32))
    with pytest.raises(ValueError, match="contiguous FP16"):
        native(torch.empty((2, 3), dtype=torch.float16).t())


@pytest.mark.parametrize("count", [-1, 0, True, 1.5])
def test_invalid_descriptor_counts_are_rejected(native, count):
    with pytest.raises(ValueError, match="positive integer"):
        native.prepare_counts([count])


def test_device_gate_is_disabled_without_an_explicit_flag(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        run(tmp_path / "missing", tmp_path / "output.json")
    assert not (tmp_path / "output.json").exists()


def test_compile_only_bundle_pins_sources_and_the_original_bridge(tmp_path, monkeypatch):
    compiler, bridge, cann = (tmp_path / name for name in ("compiler", "bridge.so", "cann"))
    compiler.write_bytes(b"compiler fixture")
    bridge.write_bytes(b"original bridge")
    cann.mkdir()
    calls = []

    def compile_binary(args, *, check):
        assert check and args[3] == "--npu-arch=dav-2002"
        calls.append(args)
        Path(args[2]).write_bytes(b"compile-only fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tools.glm_perf.build_query_bf16_vector.subprocess.run", compile_binary)
    output = build(tmp_path / "build", compiler, bridge, "bridge_fixture", 950, cann)
    report = verify(output)
    assert len(calls) == 1 and report["compile_only"] is True and report["hardware_gates"] == "not_run"
    assert report["bridge"]["path"] == str(bridge)
    assert not list(output.glob("*.so")), "copying the bridge can register its namespace twice"
    with pytest.raises(FileExistsError):
        build(output, compiler, bridge, "bridge_fixture", 950, cann)
    source = output / "query_bf16_vector.py"
    source.write_text("changed helper")
    with pytest.raises(ValueError, match="asset changed"):
        verify(output)
    assert json.loads((output / "provenance.json").read_text())["hardware_gates"] == "not_run"
