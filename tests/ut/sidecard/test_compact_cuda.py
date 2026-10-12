# SPDX-License-Identifier: Apache-2.0
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sidecard.architecture_experts import cuda_measure
from tools.sidecard.compact_cuda import CompactCudaOracle, encode_compact_weights
from tools.sidecard.device_backend import CUDA_COMPACT_PLANES, CUDA_NIBBLE_PLANES, encode_cuda_weights
from tools.sidecard.one_card import validate_compact_gate
from tools.sidecard.resident import required_cuda_bytes


@pytest.mark.parametrize("bits", (2, 3))
def test_compact_planes_decode_every_signed_code_at_the_kernel_address(bits):
    # Includes multiple experts, blocks, code pairs and channels with all edges.
    shape = (2, 64, 96)
    sign = 1 << (bits - 1)
    values = (np.arange(np.prod(shape)) % (2 * sign) - sign).astype(np.int8).reshape(shape)
    values = np.random.default_rng(bits).permutation(values.flatten()).reshape(shape)
    encoded = encode_compact_weights(values, bits)
    assert len(encoded) == values.size * bits // 8
    words = np.frombuffer(encoded, dtype="<u4" if bits == 2 else "<u2")
    restored = np.empty_like(values)
    for expert in range(shape[0]):
        for channel in range(shape[1]):
            for column in range(shape[2]):
                group, inner = divmod(column, 32)
                pair, lane = divmod(inner, 16)
                position = (expert * (shape[2] // 32) + group) * 2 + pair
                packed = (
                    int(words[position * shape[1] + channel])
                    if bits == 2
                    else sum(
                        int(words[(position * bits + word) * shape[1] + channel]) << (16 * word) for word in range(bits)
                    )
                )
                restored[expert, channel, column] = ((packed >> (bits * lane)) & (2 * sign - 1)) - sign
    np.testing.assert_array_equal(restored, values)


def test_w4_keeps_the_qualified_encoding():
    values = np.random.default_rng(4).integers(-8, 8, size=(2, 64, 96), dtype=np.int8)
    assert encode_compact_weights(values, 4) == encode_cuda_weights(values, CUDA_NIBBLE_PLANES)


@pytest.mark.parametrize("bits", (2, 3, 4))
@pytest.mark.parametrize("edge", (-1, 1))
def test_codes_outside_declared_precision_are_rejected(bits, edge):
    sign = 1 << (bits - 1)
    value = -sign - 1 if edge < 0 else sign
    with pytest.raises(ValueError, match="declared precision"):
        encode_compact_weights(np.full((1, 32, 32), value, dtype=np.int8), bits)


@pytest.mark.parametrize("bits", (True, 1, 5, 2.0, "2"))
def test_invalid_precision_is_rejected(bits):
    with pytest.raises(ValueError, match="block32"):
        encode_compact_weights(np.zeros((1, 32, 32), dtype=np.int8), bits)


@pytest.mark.parametrize("shape", ((0, 32, 32), (1, 31, 32), (1, 32, 31), (32, 32)))
def test_invalid_geometry_is_rejected(shape):
    with pytest.raises(ValueError, match="block32"):
        encode_compact_weights(np.zeros(shape, dtype=np.int8), 2)


def test_unsigned_values_are_rejected():
    with pytest.raises(ValueError, match="block32"):
        encode_compact_weights(np.zeros((1, 32, 32), dtype=np.uint8), 2)


def test_missing_compact_abi_is_rejected_before_banks_or_device_initialization(monkeypatch):
    class UnreadBanks:
        def items(self):
            pytest.fail("bank materialized before ABI admission")

    monkeypatch.setattr("tools.sidecard.device_backend.digest_file", lambda path: "pinned")
    monkeypatch.setattr(ctypes, "CDLL", lambda path: object())
    # Avoid importing the hardware helper: the ABI must be checked first.
    with pytest.raises(AttributeError, match="sidecard_oracle_forward_compact"):
        CompactCudaOracle(UnreadBanks(), {"library": Path("unused.so"), "library_sha256": "pinned"}, 1)


@pytest.mark.parametrize("bits", (True, 2, 16))
def test_measure_rejects_invalid_activation_precision_before_device_use(bits):
    with pytest.raises(ValueError, match="A4 or A8"):
        cuda_measure(None, 3, None, None, activation_bits=bits)


def test_compact_memory_admission_accounts_mixed_banks_and_peak_staging():
    plan = SimpleNamespace(
        by_layer={3: (0,), 8: (0, 1), 33: (0, 1, 2)},
        manifest={"layer_bits": {"3": 4, "8": 3, "33": 2}},
    )
    # W4/W3/W2 expert code sizes: 12/9/6 MiB. Staging needs the largest
    # gate/up bank (12 MiB), not a sum of the layer-specific temporaries.
    expected = (12 + 2 * 9 + 3 * 6 + 12 + 256) * 1024**2 + 6 * 49152
    assert required_cuda_bytes(plan, CUDA_COMPACT_PLANES) == expected


@pytest.fixture
def compact_receipt(tmp_path):
    receipt = {
        "passed": True,
        "candidate_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "candidate_weight_layout": CUDA_COMPACT_PLANES,
        "cuda_experts_per_layer": 144,
        "candidate_partial_reduce": True,
        "candidate_experimental_projection": False,
        "cases": [
            {"layer": layer, "tokens": tokens, "routes": tokens * routes, "bitwise_equal": True}
            for layer in (3, 8, 33)
            for tokens in (128, 512, 640)
            for routes in (1, 2, 8)
        ],
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(receipt))
    return path, receipt


def test_compact_admission_requires_all_27_boundary_cases(compact_receipt):
    path, receipt = compact_receipt
    assert validate_compact_gate(path, "a" * 64, "b" * 64)["cases"] == 27
    receipt["cases"].pop()
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete pinned"):
        validate_compact_gate(path, "a" * 64, "b" * 64)


@pytest.mark.parametrize(
    "field,value",
    (
        ("candidate_sha256", "c" * 64),
        ("checkpoint_sha256", "c" * 64),
        ("candidate_weight_layout", CUDA_NIBBLE_PLANES),
        ("cuda_experts_per_layer", 72),
        ("candidate_partial_reduce", False),
        ("candidate_experimental_projection", True),
        ("passed", False),
    ),
)
def test_compact_admission_rejects_relabelled_or_failed_receipts(compact_receipt, field, value):
    path, receipt = compact_receipt
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete pinned"):
        validate_compact_gate(path, "a" * 64, "b" * 64)


def test_compact_admission_rejects_duplicate_cases_even_with_full_set(compact_receipt):
    path, receipt = compact_receipt
    receipt["cases"].append(receipt["cases"][0])
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete pinned"):
        validate_compact_gate(path, "a" * 64, "b" * 64)
