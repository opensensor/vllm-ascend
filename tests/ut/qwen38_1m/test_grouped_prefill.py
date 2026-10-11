# SPDX-License-Identifier: Apache-2.0
"""Boundary and storage gates for the explicitly selected prefill resource."""

import ctypes
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re
import torch

from tools.qwen4exp.native_prefill import MAX_ROUTES, NativeGroupedPrefill
from tools.qwen4exp.resident_candidates import grouped_prefill
from vllm_ascend.models.qwen4_exp.w4_moe import PackedExpertBank
from vllm_ascend.models.qwen4_exp.w4a8_int4 import NATIVE_INT4_BACKEND


def operands(rows=129, experts=2):
    outputs, inputs = 2560, 640
    bank = SimpleNamespace(
        weight=torch.empty(experts, outputs, inputs // 2, dtype=torch.int8),
        weight_scale=torch.empty(experts, outputs, inputs // 128, dtype=torch.float16),
        weight_offset=torch.empty(experts, outputs, inputs // 128, dtype=torch.float16),
        weight_sum=torch.empty(experts, outputs, inputs // 128, dtype=torch.float16),
    )
    prepared = (
        torch.empty(rows, inputs // 2, dtype=torch.int8),
        torch.empty(rows, inputs // 2, dtype=torch.int8),
        torch.empty(rows, inputs // 128, 8, dtype=torch.float32),
        torch.empty(rows, inputs // 128, 8, dtype=torch.float32),
    )
    return bank, prepared, torch.tensor([17, rows], dtype=torch.int64)


def test_reuses_original_banks_and_device_boundaries_without_host_counts(monkeypatch):
    monkeypatch.setattr("tools.qwen4exp.native_prefill._on_npu", lambda device: True)
    monkeypatch.setattr("tools.qwen4exp.native_prefill._capturing", lambda device: False)
    bank, prepared, ends = operands()
    launches = []
    resource = NativeGroupedPrefill(None, lambda kernel, arguments, blocks: launches.append((arguments, blocks)))
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("device count read on host"))
    result = resource(bank, prepared, ends)
    arguments, blocks = launches[0]
    assert blocks == 8
    assert result.shape == (129, 2560) and result.dtype == torch.float16
    assert all(a is b for a, b in zip(arguments[:4], prepared))
    assert arguments[4] is bank.weight and arguments[8] is ends
    assert arguments[-1].tolist() == [129, 2, 2560, 640, 8, 0, 1]
    resource(bank, prepared, ends)
    assert launches[1][0][-1] is arguments[-1]


@pytest.mark.parametrize("rows", [0, 1, 30, 128, MAX_ROUTES + 1])
def test_decode_and_oversized_rows_are_rejected(rows):
    bank, prepared, ends = operands(rows)
    resource = NativeGroupedPrefill(None, lambda *args: pytest.fail("unsupported kernel launched"))
    with pytest.raises(ValueError, match="geometry/dtype"):
        resource(bank, prepared, ends)


@pytest.mark.parametrize("kind", ["routed", "scalar_metadata", "wide_offsets", "wrong_experts"])
def test_rejects_incompatible_native_layouts(kind):
    bank, prepared, ends = operands()
    if kind == "routed":
        ends = ends.int()
    elif kind == "scalar_metadata":
        prepared = (*prepared[:2], prepared[2][..., 0], prepared[3][..., 0])
    elif kind == "wide_offsets":
        bank.weight_offset = bank.weight_offset.float()
    else:
        ends = ends[:1]
    resource = NativeGroupedPrefill(None, lambda *args: pytest.fail("unsupported kernel launched"))
    with pytest.raises(ValueError, match="geometry/dtype"):
        resource(bank, prepared, ends)


@pytest.mark.parametrize(
    "rows,dtype,backend,candidate",
    [
        (3, torch.int32, NATIVE_INT4_BACKEND, False),
        (128, torch.int64, NATIVE_INT4_BACKEND, False),
        (512, torch.int64, NATIVE_INT4_BACKEND, False),
        (513, torch.int64, NATIVE_INT4_BACKEND, True),
        (513, torch.int32, NATIVE_INT4_BACKEND, False),
        (513, torch.int64, "eager_dequant", False),
    ],
)
def test_resident_binding_preserves_decode_and_small_batches(monkeypatch, rows, dtype, backend, candidate):
    baseline, selected = object(), object()
    monkeypatch.setattr(PackedExpertBank, "native_linear", lambda *args: baseline)
    resources = {grouped_prefill.RESOURCE_NAME: {"projection": lambda *args: selected}}
    binding = grouped_prefill.replacements(resources)
    function = binding["vllm_ascend.models.qwen4_exp.w4_moe:PackedExpertBank.native_linear"]
    bank, prepared, ends = operands(rows)
    bank.backend = backend
    result = function(bank, prepared, ends.to(dtype))
    assert result is (selected if candidate else baseline)


def test_repeated_candidate_preparation_unwraps_the_original(monkeypatch):
    original = lambda *args: None
    monkeypatch.setattr(PackedExpertBank, "native_linear", original)
    resources = {grouped_prefill.RESOURCE_NAME: {"projection": lambda *args: None}}
    target = "vllm_ascend.models.qwen4_exp.w4_moe:PackedExpertBank.native_linear"
    first = grouped_prefill.replacements(resources)[target]
    monkeypatch.setattr(PackedExpertBank, "native_linear", first)
    second = grouped_prefill.replacements(resources)[target]
    assert second._qwen_grouped_prefill_base is original


@pytest.fixture(scope="module")
def compiled_production_policy(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    kernel = root / "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel"
    text = (kernel / "qwen_w4_a8_int4_matmul_v310.cpp").read_text()
    # Compile the actual dispatch body with schedules replaced by recording
    # stubs. This exercises production pointer/dtype branching on CPU; it is
    # deliberately separate from the hardware arithmetic and event gates.
    body = text[text.index("template <uint32_t N>") : text.index("}  // namespace")]
    constants = "\n".join(re.findall(r"constexpr [^;]+;", text[: text.index("class NativeInt4")]))
    source = (
        r"""
#include <cstdint>
#define __aicore__
#define __gm__
using GM_ADDR = void*;
int selectedM, selectedN, boundaryReads;
template<typename T> struct GlobalTensor {
 T* p; void SetGlobalBuffer(T* q) { p=q; }
 T GetValue(int64_t i) { ++boundaryReads; return p[i]; }
};
namespace native_int4 {
template<unsigned M, unsigned N=64, bool A=false, bool B=false, bool C=false>
struct Schedule {
 template<typename... Args> void Init(Args...) {}
 void Process() { selectedM=M; selectedN=N; }
};
}
"""
        + f'#include "{kernel / "qwen_w4_a8_int4_matmul_v310_tiling_data.h"}"\n'
        + constants
        + body
        + r"""
extern "C" int policy(int64_t rows, int64_t experts, int64_t local) {
 QwenW4A8Int4KernelTilingData td{rows, experts, 1280, 2560, 8, 0, 1};
 int64_t ends[128]{}; ends[experts-1]=local;
 boundaryReads=0;
 RunSchedule<160>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,ends,nullptr,&td);
 return selectedM*10000+selectedN*10+boundaryReads;
}
"""
    )
    directory = tmp_path_factory.mktemp("qwen-grouped-policy")
    (directory / "policy.cpp").write_text(source)
    library = directory / "policy.so"
    subprocess.run(
        ["c++", "-shared", "-fPIC", "-std=c++17", str(directory / "policy.cpp"), "-o", str(library)], check=True
    )
    loaded = ctypes.CDLL(str(library))
    loaded.policy.argtypes = [ctypes.c_int64] * 3
    loaded.policy.restype = ctypes.c_int
    return loaded.policy


@pytest.mark.parametrize(
    "rows,experts,local,m,n,reads",
    [
        (30, 85, 0, 16, 160, 0),
        (128, 85, 0, 16, 160, 0),
        (25600, 85, 0, 32, 160, 1),
        (25600, 85, 6257, 32, 160, 1),
        (25600, 85, 21760, 32, 160, 1),
        (25600, 85, 21761, 128, 64, 1),
    ],
)
def test_production_schedule_uses_local_work_and_preserves_decode(
    compiled_production_policy, rows, experts, local, m, n, reads
):
    assert compiled_production_policy(rows, experts, local) == m * 10000 + n * 10 + reads
