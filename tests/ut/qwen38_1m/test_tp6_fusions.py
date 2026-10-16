# SPDX-License-Identifier: Apache-2.0
import ctypes
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tools.qwen4exp import native_prefill, tp6_fusions, tp6_operand_reuse
from tools.qwen4exp.build_tp6_fusions import stage
from tools.qwen4exp.local_route_epilogue import reference_reduce, reference_weighted
from tools.qwen4exp.tp6_fusion_sources import digest, private_schedule, sources

ROOT = Path(__file__).resolve().parents[3]
STUBS = Path(__file__).parent / "local_epilogue_cpu_stubs"


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    binary = tmp_path_factory.mktemp("tp6-epilogue") / "kernel.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-shared",
            "-fPIC",
            f"-I{STUBS}",
            f"-I{ROOT / 'tools/qwen4exp'}",
            str(STUBS / "tp6_harness.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(binary))
    function = library.run_tp6_epilogue
    function.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, *([ctypes.c_void_p] * 7)]
    function.restype = ctypes.c_int

    def run(mode, routed, inverse, weights, ends, *, fails=False):
        rows = routed.shape[0]
        output = torch.full((rows if mode == "ordered" else rows // 10, 2560), float("nan"))
        reads, local = ctypes.c_uint64(), ctypes.c_uint64()
        result = function(
            tp6_fusions.MODES.index(mode),
            rows,
            ends.numel(),
            routed.data_ptr(),
            inverse.data_ptr(),
            weights.data_ptr(),
            ends.data_ptr(),
            output.data_ptr(),
            ctypes.byref(reads),
            ctypes.byref(local),
        )
        assert result == int(fails)
        if not fails:
            assert local.value <= 32 * 1024
        return output, reads.value

    return run


@pytest.mark.parametrize("experts", [85, 86])
@pytest.mark.parametrize("rows,active", [(10, 0), (130, 1), (650, 327), (650, 650), (20480, 1)])
@pytest.mark.parametrize("mode", tp6_fusions.MODES)
def test_native_tp6_tail_bounds_permutation_arithmetic(native, experts, rows, active, mode):
    generator = torch.Generator().manual_seed(rows + experts)
    routed = torch.randn(rows, 2560, dtype=torch.float16, generator=generator)
    routed[active:] = float("nan")
    inverse = torch.randperm(rows, generator=generator)
    weights = torch.rand(rows, 1, generator=generator)
    ends = torch.tensor([(i + 1) * active // experts for i in range(experts)])
    expected = reference_weighted(routed, inverse, weights, active)
    if mode != "ordered":
        expected = reference_reduce(expected, mode=mode)
    actual, reads = native(mode, routed, inverse, weights, ends)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert reads == active * 2560 * 2


@pytest.mark.parametrize("mode", tp6_fusions.MODES)
def test_native_invalid_indices_and_peer_nonfinite(native, mode):
    routed = torch.full((10, 2560), float("nan"), dtype=torch.float16)
    inverse = torch.arange(10)
    weights = torch.tensor([-0.0, -1.0, float("nan"), float("inf"), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])[:, None]
    ends = torch.zeros(85, dtype=torch.int64)
    expected = reference_weighted(routed, inverse, weights, 0)
    if mode != "ordered":
        expected = reference_reduce(expected, mode=mode)
    actual, reads = native(mode, routed, inverse, weights, ends)
    assert reads == 0
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    finite = ~torch.isnan(expected)
    assert torch.equal(actual[finite].view(torch.int32), expected[finite].view(torch.int32))
    inverse[0] = 10
    native(mode, routed, inverse, weights, ends, fails=True)
    inverse[0] = 0
    ends[-1] = 11
    native(mode, routed, inverse, weights, ends, fails=True)


def test_staging_is_pinned_and_does_not_claim_device_validation(tmp_path):
    destination = tmp_path / "stage"
    receipt = stage(ROOT, destination)
    assert receipt["hardware_validated"] is receipt["device_opened"] is receipt["server_modified"] is False
    assert not receipt["binaries"]
    with pytest.raises(FileExistsError):
        stage(ROOT, destination)
    with pytest.raises(ValueError, match="both compiled"):
        tp6_fusions.verify_build(destination)
    files = sources(ROOT)
    original = files["native_int4_schedule.h"]
    with pytest.raises(ValueError, match="unqualified"):
        private_schedule(original + b"\n")
    assert b"Store(out, tile, row, count);" not in files["tp6_local_schedule.h"]
    assert b"Paired gate/up consumes output_" in files["tp6_local_schedule.h"]
    for name in ("native_tp6_gate_pack.bin", "native_tp6_finalize.bin"):
        (destination / name).write_bytes(b"test-only-not-a-kernel")
        receipt["binaries"][name] = digest((destination / name).read_bytes())
    (destination / "provenance.json").write_text(json.dumps(receipt))
    assert tp6_fusions.verify_build(destination) == receipt
    (destination / "native_tp6_finalize.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        tp6_fusions.verify_build(destination)


def test_native_nz_to_pack_group_mapping():
    # Model every index consumed by LocalPack::Emit, including partial M tiles.
    # This checks layout only; it is not a numerical simulation of Cube/vector.
    for count in (1, 7, 8, 9, 16):
        dense = torch.arange(16 * 128).reshape(16, 128)
        nz = dense.reshape(16, 8, 16).permute(1, 0, 2).contiguous().flatten()
        actual = torch.stack(
            [
                torch.cat([nz[nb * 256 + row * 16 : nb * 256 + (row + 1) * 16] for nb in range(8)])
                for row in range(count)
            ]
        )
        assert torch.equal(actual, dense[:count])


@pytest.fixture(scope="module")
def zero_native(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = tmp_path_factory.mktemp("tp6-zero")
    # Exercise the real generated EmitZero body with bounds/event checking.
    # Omit the unrelated activation math, which this fixture does not model.
    source = sources(ROOT)["tp6_local_pack.h"].decode()
    start = source.index("  __aicore__ inline void Emit(")
    stop = source.index("  __aicore__ inline void EmitZero(")
    (root / "tp6_zero_local_pack.h").write_text(source[:start] + source[stop:])
    binary = root / "zero.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            f"-I{root}",
            f"-I{STUBS.parent / 'streaming_cpu_stubs'}",
            str(STUBS / "tp6_zero_harness.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    function = ctypes.CDLL(str(binary)).run_tp6_zero
    function.argtypes = [ctypes.c_int64, ctypes.c_int64, *([ctypes.c_void_p] * 4)]
    function.restype = ctypes.c_int
    return function


@pytest.mark.parametrize("rows,active", [(1, 0), (13, 1), (65, 32), (2560, 2560), (20480, 19001)])
def test_native_zero_pack_initializes_only_peer_tail(zero_native, rows, active):
    low, high = (torch.full((rows, 320), 17, dtype=torch.int8) for _ in range(2))
    scale, total = (torch.full((rows, 5, 8), 17.0) for _ in range(2))
    assert zero_native(rows, active, *(t.data_ptr() for t in (low, high, scale, total))) == 0
    for tensor, packed_zero in ((low, -120), (high, 0), (scale, 1), (total, 0)):
        assert bool((tensor[:active] == 17).all())
        assert bool((tensor[active:] == packed_zero).all())


def test_eager_launch_failure_retains_and_poisons(monkeypatch):
    monkeypatch.setattr(native_prefill, "_on_npu", lambda device: False)
    sentinel = torch.ones(3)

    def fail(kernel, args, blocks):
        raise RuntimeError("submission uncertain")

    resource = tp6_fusions.EagerResource(None, fail)
    with pytest.raises(RuntimeError, match="uncertain"):
        resource.submit([sentinel])
    assert resource.retained[0] is sentinel
    with torch.no_grad(), pytest.raises(RuntimeError, match="cannot be reused"):
        resource.preflight(sentinel.device)


@pytest.fixture
def hc(monkeypatch):
    monkeypatch.setattr(tp6_operand_reuse, "_on_npu", lambda device: True)
    generator = torch.Generator().manual_seed(42)

    def weight(shape):
        return (torch.randn(shape, generator=generator) * 0.01).half()

    return SimpleNamespace(
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        use_combine=True,
        hc_count=4,
        hidden_size=2560,
        input_mix_weight_down=weight((320, 10240)),
        input_mix_weight_up=weight((10240, 320)),
        block_inject_weight=weight((4, 10240)),
        _normalize=lambda value: value.float() * 0.33,
    )


@pytest.mark.parametrize("rows", tp6_operand_reuse.DECODE_ROWS)
@torch.no_grad()
def test_operand_is_reused_only_within_forward_and_preserves_mix_combine(hc, rows):
    packed = []

    def pack(value):
        result = value.half()
        packed.append(result)
        return result

    def fallback(module, value):
        raise AssertionError("unexpected fallback")

    replacement = tp6_operand_reuse.make_mix(fallback, pack=pack)
    pointers = [w.data_ptr() for w in (hc.input_mix_weight_down, hc.input_mix_weight_up, hc.block_inject_weight)]
    for repeat in range(2):
        value = torch.randn(rows, 10240).half() + repeat
        norm = hc._normalize(value)
        gate = F.silu(F.linear(norm.half(), hc.input_mix_weight_down) / 4)
        gate = torch.sigmoid(F.linear(gate, hc.input_mix_weight_up))
        expected = (gate.reshape(rows, 4, 2560) * norm.reshape(rows, 4, 2560)).mean(1).half()
        actual, (hyper, operand) = replacement(hc, value)
        assert hyper is value and operand is packed[-1] and len(packed) == repeat + 1
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
        assert torch.equal(F.linear(norm.half(), hc.block_inject_weight), F.linear(operand, hc.block_inject_weight))
    assert packed[0].data_ptr() != packed[1].data_ptr()
    assert pointers == [
        w.data_ptr() for w in (hc.input_mix_weight_down, hc.input_mix_weight_up, hc.block_inject_weight)
    ]


def test_operand_unsupported_and_training_use_fallback(hc):
    sentinel = object()
    replacement = tp6_operand_reuse.make_mix(lambda module, value: sentinel)
    assert replacement(hc, torch.zeros(3, 10240).half()) is sentinel
    with torch.no_grad():
        assert replacement(hc, torch.zeros(5, 10240).half()) is sentinel
        hc.use_combine = False
        assert replacement(hc, torch.zeros(3, 10240).half()) is sentinel


def meta_bank(experts=85):
    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    return SimpleNamespace(
        weight=tensor((experts, 1280, 1280), torch.int8),
        weight_scale=tensor((experts, 1280, 20), torch.float16),
        weight_offset=tensor((experts, 1280, 20), torch.float16),
        weight_sum=tensor((experts, 1280, 20), torch.float16),
    )


@pytest.mark.parametrize(
    "experts,rows,routed,broadcast",
    [(85, 30, True, 10), (86, 120, True, 1), (85, 180, False, 1), (86, 20480, False, 1)],
)
@torch.no_grad()
def test_gate_pack_admission_descriptor_outputs_without_allocating_bank(monkeypatch, experts, rows, routed, broadcast):
    monkeypatch.setattr(tp6_fusions, "_capturing", lambda device: False)
    monkeypatch.setattr(tp6_fusions, "_tensors_on_one_device", lambda tensors, device: None)
    calls = []
    op = tp6_fusions.GatePack(None, lambda kernel, args, blocks: calls.append((args, blocks)))
    source = rows // broadcast
    prepared = tuple(
        torch.empty(shape, dtype=dtype, device="meta")
        for shape, dtype in (
            ((source, 1280), torch.int8),
            ((source, 1280), torch.int8),
            ((source, 20, 8), torch.float32),
            ((source, 20, 8), torch.float32),
        )
    )
    ends = torch.empty(rows if routed else experts, dtype=torch.int32 if routed else torch.int64, device="meta")
    bank = meta_bank(experts)
    output = op(bank, prepared, ends)
    assert [tuple(t.shape) for t in output] == [(rows, 320), (rows, 320), (rows, 5, 8), (rows, 5, 8)]
    assert calls[0][1] == 8 and len(calls[0][0]) == 14
    assert next(iter(op._configs))[0] == (rows, experts, 1280, 2560, 8, int(routed), broadcast)
    assert calls[0][0][4] is bank.weight
    op(bank, prepared, ends)
    assert len(op._configs) == 1
    bank.weight_sum = bank.weight_sum.float()
    with pytest.raises(ValueError, match="geometry"):
        op(bank, prepared, ends)
    assert len(calls) == 2


@pytest.mark.parametrize("mode", tp6_fusions.MODES)
@torch.no_grad()
def test_finalize_wrapper_geometry_and_no_graph_admission(monkeypatch, mode):
    monkeypatch.setattr(tp6_fusions, "_capturing", lambda device: False)
    monkeypatch.setattr(tp6_fusions, "_tensors_on_one_device", lambda tensors, device: None)
    calls = []
    op = tp6_fusions.Finalize(None, lambda kernel, args, blocks: calls.append(args), mode)
    routed = torch.empty((130, 2560), dtype=torch.float16, device="meta")
    inverse = torch.empty(130, dtype=torch.int64, device="meta")
    weights = torch.empty((130, 1), dtype=torch.float32, device="meta")
    ends = torch.empty(86, dtype=torch.int64, device="meta")
    assert op(routed, inverse, weights, ends).shape == (13, 2560)
    with pytest.raises(ValueError, match="geometry"):
        op(routed, inverse, weights, ends[:84])
    monkeypatch.setattr(tp6_fusions, "_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="eager inference"):
        op(routed, inverse, weights, ends)
    assert len(calls) == 1


def test_preflight_imports_no_framework(tmp_path):
    script = (
        "from tools.qwen4exp.benchmark_tp6_fusions_310 import plan; import sys; "
        "assert plan()['hardware_validated'] is False; assert 'torch' not in sys.modules"
    )
    subprocess.run([__import__("sys").executable, "-c", script], cwd=ROOT, check=True)
