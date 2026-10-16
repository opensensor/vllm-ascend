# SPDX-License-Identifier: Apache-2.0
import ctypes
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import local_route_epilogue as candidate
from tools.qwen4exp.build_local_route_epilogue import ZERO_TAIL, local_output_header, stage

ROOT = Path(__file__).resolve().parents[3]
STUBS = Path(__file__).parent / "local_epilogue_cpu_stubs"


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    binary = tmp_path_factory.mktemp("local-epilogue") / "kernel.so"
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
            str(STUBS / "harness.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(binary))
    function = library.run_local_epilogue
    function.argtypes = [ctypes.c_int, ctypes.c_int64, *([ctypes.c_void_p] * 7)]
    function.restype = ctypes.c_int

    def launch(mode, routed, inverse, weights, ends, *, fail=False):
        rows = routed.shape[0]
        shape = (rows if mode == "ordered" else rows // candidate.TOP_K, candidate.WIDTH)
        output = torch.full(shape, float("nan"))
        reads, local_bytes = ctypes.c_uint64(), ctypes.c_uint64()
        result = function(
            candidate.MODES.index(mode),
            rows,
            routed.data_ptr(),
            inverse.data_ptr(),
            weights.data_ptr(),
            ends.data_ptr(),
            output.data_ptr(),
            ctypes.byref(reads),
            ctypes.byref(local_bytes),
        )
        if fail:
            assert result == 1
        else:
            assert result == 0
            assert local_bytes.value <= (12 if mode == "pairwise" else 32) * 1024
        return output, reads.value

    return launch


def operands(rows, active):
    generator = torch.Generator().manual_seed(rows + active)
    routed = torch.randn(rows, candidate.WIDTH, dtype=torch.float16, generator=generator)
    routed[active:] = float("nan")
    inverse = torch.randperm(rows, generator=generator)
    weights = torch.rand(rows, 1, generator=generator)
    ends = torch.tensor([(i + 1) * active // candidate.EXPERTS for i in range(candidate.EXPERTS)])
    return routed, inverse, weights, ends


@pytest.mark.parametrize("rows", [130, 650, 2590, 5150])
@pytest.mark.parametrize("ownership", ["none", "one", "quarter", "almost", "all"])
@pytest.mark.parametrize("mode", candidate.MODES)
def test_actual_native_source_never_reads_tail_and_matches_explicit_oracle(native, rows, ownership, mode):
    active = {"none": 0, "one": 1, "quarter": rows // 4, "almost": rows - 1, "all": rows}[ownership]
    routed, inverse, weights, ends = operands(rows, active)
    expected = candidate.reference_weighted(routed, inverse, weights, active)
    if mode != "ordered":
        expected = candidate.reference_reduce(expected, mode=mode)
    actual, reads = native(mode, routed, inverse, weights, ends)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    # Every active row appears once in the permutation and is read once.
    assert reads == active * candidate.WIDTH * 2
    assert torch.isnan(routed[active:]).all()


@pytest.mark.parametrize("mode", candidate.MODES)
def test_maximum_capacity(native, mode):
    rows, active = candidate.MAX_ROUTES, 1
    routed, inverse, weights, ends = operands(rows, active)
    actual, reads = native(mode, routed, inverse, weights, ends)
    expected = candidate.reference_weighted(routed, inverse, weights, active)
    if mode != "ordered":
        expected = candidate.reference_reduce(expected, mode=mode)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert reads == active * candidate.WIDTH * 2


@pytest.mark.parametrize("mode", candidate.MODES)
def test_zero_sign_subnormal_and_nonfinite_weights(native, mode):
    routed, inverse, weights, ends = operands(130, 65)
    inverse = torch.arange(130)
    routed[0, :6] = torch.tensor([-0.0, 0.0, 2**-24, -(2**-24), 65504.0, -65504.0]).half()
    weights[:6, 0] = torch.tensor([-0.0, -1.0, 2**-24, 0.0, float("inf"), float("nan")])
    weights[65:71, 0] = weights[:6, 0]
    expected = candidate.reference_weighted(routed, inverse, weights, 65)
    if mode != "ordered":
        expected = candidate.reference_reduce(expected, mode=mode)
    actual, _ = native(mode, routed, inverse, weights, ends)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    valid = ~torch.isnan(expected)
    assert torch.equal(actual[valid].view(torch.int32), expected[valid].view(torch.int32))


def test_cancellation_proves_reduction_order_cannot_be_assumed(native):
    routed, inverse, weights, ends = operands(130, 130)
    routed.zero_()
    inverse = torch.arange(130)
    routed[:4] = torch.tensor([65504.0, 1.0, -65504.0, 1.0]).half()[:, None]
    weights.fill_(1.0)
    weights[1] = weights[3] = 2**-10
    sequential, _ = native("sequential", routed, inverse, weights, ends)
    pairwise, _ = native("pairwise", routed, inverse, weights, ends)
    assert sequential[0, 0] == 2**-10 and pairwise[0, 0] == 0
    assert not torch.equal(sequential, pairwise)


@pytest.mark.parametrize("bad", ["negative_end", "large_end", "negative_inverse", "large_inverse"])
def test_native_bounds_fail_before_out_of_range_access(native, bad):
    routed, inverse, weights, ends = operands(130, 32)
    if bad == "negative_end":
        ends[-1] = -1
    elif bad == "large_end":
        ends[-1] = 131
    elif bad == "negative_inverse":
        inverse[0] = -1
    else:
        inverse[0] = 130
    native("ordered", routed, inverse, weights, ends, fail=True)


@pytest.fixture
def pair(monkeypatch, native, request):
    monkeypatch.setattr(candidate.route_finalize, "_supports_device", lambda _: True)
    monkeypatch.setattr(candidate.route_finalize, "_is_capturing", lambda: False)
    monkeypatch.setattr(candidate.qfn_prefill_schedule, "eligible", lambda *a: True)
    projected, inverse, weights, ends = operands(130, 32)
    bank = SimpleNamespace(weight=torch.empty((128, 2560, 320), device="meta"))
    submissions = []

    class Projection:
        def __init__(self):
            self._configs = {}

        def __call__(self, *args):
            submissions.append("projection")
            return projected

    def launch(kernel, tensors, blocks):
        routed, order, route_weights, boundaries, output, descriptor = tensors
        assert blocks == 8 and descriptor.tolist() == [130]
        submissions.append("epilogue")
        actual, _ = native(kernel, routed, order, route_weights, boundaries)
        output.copy_(actual)

    mode = getattr(request, "param", "ordered")
    epilogue = candidate.LocalRouteEpilogue(mode, launch, mode=mode)
    owner = candidate.LocalDownEpilogue(Projection(), epilogue, native_backend="native")
    return owner, (bank, (torch.empty(130, 320, dtype=torch.int8),), ends, inverse, weights), submissions, projected


@pytest.mark.parametrize("pair", candidate.MODES, indirect=True)
@torch.no_grad()
def test_pair_preserves_backend_sum_and_never_exposes_bad_tail(pair, monkeypatch):
    owner, args, submissions, projected = pair
    weighted = candidate.reference_weighted(projected, args[3], args[4], 32)
    expected = (
        weighted.reshape(-1, 10, 2560).sum(1)
        if owner.epilogue.mode == "ordered"
        else candidate.reference_reduce(weighted, mode=owner.epilogue.mode)
    )
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *a: pytest.fail("normal path must not read route counts"))
    actual = owner(*args)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert submissions == ["projection", "epilogue"] and not owner.retained
    assert owner.calls == 1


@pytest.mark.parametrize(
    "reason",
    ["projection_full", "epilogue_full", "capture", "shape", "inverse", "weights", "ends", "eligibility", "grad"],
)
def test_pair_admission_rejects_before_any_submission(pair, monkeypatch, reason):
    owner, args, submissions, _ = pair
    bank, prepared, ends, inverse, weights = args
    if reason == "projection_full":
        monkeypatch.setattr(candidate.native_prefill, "MAX_CONFIGS", 0)
    elif reason == "epilogue_full":
        monkeypatch.setattr(candidate, "MAX_CONFIGS", 0)
    elif reason == "capture":
        monkeypatch.setattr(candidate.route_finalize, "_is_capturing", lambda: True)
    elif reason == "shape":
        bank.weight = torch.empty((128, 1280, 1280), device="meta")
    elif reason == "inverse":
        inverse = inverse.int()
    elif reason == "weights":
        weights = weights.half()
    elif reason == "ends":
        ends = ends[:64]
    elif reason == "eligibility":
        monkeypatch.setattr(candidate.qfn_prefill_schedule, "eligible", lambda *a: False)
    with torch.set_grad_enabled(reason == "grad"), pytest.raises(ValueError, match="not admitted"):
        owner(bank, prepared, ends, inverse, weights)
    assert not submissions and not owner.retained


@torch.no_grad()
def test_cached_descriptors_remain_usable_at_capacity(pair, monkeypatch):
    owner, args, _, _ = pair
    device = args[1][0].device
    owner.projection._configs[((130, 128, 2560, 640, 8, 0, 1), device)] = object()
    owner.epilogue._configs[(130, device)] = torch.tensor([130])
    monkeypatch.setattr(candidate.native_prefill, "MAX_CONFIGS", 1)
    monkeypatch.setattr(candidate, "MAX_CONFIGS", 1)
    assert owner.can_run(*args)
    owner(*args)


@torch.no_grad()
def test_no_fallback_after_producer_and_failure_retains_resources(pair):
    owner, args, submissions, projected = pair
    failure = RuntimeError("consumer submission failed")

    def fail(*args):
        raise failure

    owner.epilogue.launch = fail
    with pytest.raises(RuntimeError, match="consumer submission"):
        owner(*args)
    assert owner.failed and owner.failure is failure and owner.retained[-1] is projected
    with pytest.raises(RuntimeError, match="preserve resources"):
        owner(*args)
    assert submissions == ["projection"]


def test_frozen_header_transformation_is_only_zero_tail_removal(tmp_path):
    receipt = stage(ROOT, tmp_path / "staged")
    original = (tmp_path / "staged/native_int4_schedule.h").read_bytes()
    generated = (tmp_path / "staged/native_int4_local_output.h").read_bytes()
    assert generated == local_output_header(original)
    assert ZERO_TAIL not in generated.decode()
    assert (
        generated.replace(b"        // Inactive rows are private to the paired local epilogue.\n", ZERO_TAIL.encode())
        == original
    )
    assert not receipt["hardware_validated"] and not receipt["server_modified"] and not receipt["binaries"]
    with pytest.raises(ValueError, match="qualified frozen"):
        local_output_header(original + b"// changed")
    with pytest.raises(FileExistsError):
        stage(ROOT, tmp_path / "staged")


def test_traffic_budget_is_logical_bytes():
    traffic = candidate.traffic_bytes(20480, 5120)
    mib = 1024**2
    assert traffic == dict(
        omitted_projection_zero_writes=75 * mib,
        omitted_epilogue_tail_reads=75 * mib,
        eliminated_fp32_intermediate=200 * mib,
        eliminated_intermediate_write_and_read=400 * mib,
        final_fp32_output=20 * mib,
    )


@torch.no_grad()
def test_producer_failure_cannot_fall_back_or_drop_inputs(pair):
    owner, args, submissions, _ = pair

    class FailedProjection:
        _configs = {}

        def __call__(self, *args):
            raise RuntimeError("producer submission failed")

    owner.projection = FailedProjection()
    with pytest.raises(RuntimeError, match="producer submission"):
        owner(*args)
    assert owner.failed and owner.retained[0] is args[0] and not submissions


def test_bad_reduction_mode_is_not_silently_selected():
    with pytest.raises(ValueError, match="select ordered"):
        candidate.LocalRouteEpilogue(None, None, mode="automatic")


def test_compile_staged_refuses_source_changes(tmp_path):
    from tools.qwen4exp.build_local_route_epilogue import compile_staged

    staged = tmp_path / "staged"
    stage(ROOT, staged)
    (staged / "native_local_down_projection.cpp").write_text("changed")
    compiler = tmp_path / "compiler"
    compiler.touch()
    with pytest.raises(ValueError, match="staged source changed"):
        compile_staged(staged, compiler, tmp_path)


def test_private_gate_stops_on_wrong_reduction_without_enabling_serving():
    from tools.qwen4exp.benchmark_local_route_epilogue import validate_pair

    class BadPair:
        epilogue = SimpleNamespace(mode="pairwise")

        def __call__(self, bank, packed, ends, inverse, weights):
            return torch.ones(len(inverse) // 10, 2560)

    bank = SimpleNamespace(weight=torch.empty(1))
    receipt = validate_pair(
        BadPair(), bank, lambda x: x, lambda bank, packed, ends: torch.zeros(len(packed), 2560).half(), rows=(130,)
    )
    assert not receipt["passed"] and not receipt["serving_enabled"] and len(receipt["records"]) == 1


@pytest.fixture
def packed_adapter(pair, monkeypatch):
    owner, (bank, prepared, ends, inverse, weights), submissions, routed = pair
    baseline_rows = routed.clone()
    baseline_rows[32:] = 0
    bank.native_linear = lambda values, boundaries: baseline_rows
    bank.grouped_linear = lambda values, boundaries: baseline_rows

    class RetainedPack:
        failed = False
        mismatches = 0
        kernel, local_kernel = object(), object()
        launch = activation = pack = None

        def __call__(self, projected, projection, boundaries, compute_dtype, params_dtype):
            return baseline_rows

    class PrivatePack:
        def __init__(self, *args, local_projection, **kwargs):
            self.guard = local_projection
            self.verify = True

        def __call__(self, projected, projection, boundaries, compute_dtype, params_dtype):
            self.guard(projection, prepared, boundaries)
            return projection.native_linear(prepared, boundaries)

    monkeypatch.setattr(candidate, "W4ActivationPack", PrivatePack)
    adapter = candidate.PackedLocalDownEpilogue(owner, RetainedPack(), lambda value, order, w: value[order].float() * w)
    args = (torch.zeros(130, 1280).half(), bank, ends, inverse, weights, torch.float32, torch.float16)
    return adapter, args, submissions


@torch.no_grad()
def test_packed_adapter_defaults_off_then_shadows_against_retained_sum(packed_adapter):
    adapter, args, submissions = packed_adapter
    expected = adapter(*args)
    assert not submissions and not adapter.enabled and not adapter.packer.verify
    adapter.enabled = True
    actual = adapter(*args)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert adapter.checked == 1 and adapter.mismatches == 0
    assert submissions == ["projection", "epilogue"] and not adapter.retained
    adapter.verify = False
    assert torch.equal(adapter(*args).view(torch.int32), expected.view(torch.int32))
    assert adapter.checked == 1


@torch.no_grad()
def test_full_pipeline_shadow_rejects_wrong_reduction(packed_adapter):
    adapter, args, _ = packed_adapter
    expected = adapter(*args)
    original = adapter.pair.epilogue.launch

    def wrong(kernel, tensors, blocks):
        original(kernel, tensors, blocks)
        tensors[-2].add_(1)

    adapter.pair.epilogue.launch = wrong
    adapter.enabled = True
    actual = adapter(*args)
    assert torch.equal(actual, expected)
    assert adapter.mismatches == 1 and not adapter.enabled


@torch.no_grad()
def test_packed_admission_failure_keeps_complete_retained_consumer(packed_adapter, monkeypatch):
    adapter, args, submissions = packed_adapter
    expected = adapter(*args)
    monkeypatch.setattr(candidate, "MAX_CONFIGS", 0)
    adapter.enabled, adapter.verify = True, False
    assert torch.equal(adapter(*args), expected)
    assert not submissions


@torch.no_grad()
def test_packed_post_admission_failure_never_calls_baseline(packed_adapter, monkeypatch):
    adapter, args, submissions = packed_adapter
    adapter.enabled = True
    monkeypatch.setattr(adapter, "baseline", lambda *a: pytest.fail("post-submission fallback is unsafe"))

    def fail(*args):
        raise RuntimeError("epilogue failed")

    adapter.pair.epilogue.launch = fail
    with pytest.raises(RuntimeError, match="epilogue failed"):
        adapter(*args)
    assert adapter.failed and adapter.retained and adapter.pair.retained
    assert submissions == ["projection"]


@pytest.fixture(scope="module")
def projection_entry(tmp_path_factory):
    directory = tmp_path_factory.mktemp("local-projection-entry")
    stage(ROOT, directory / "frozen")
    library = directory / "projection.so"
    # The exact descriptor comes from the qualified git object. Only Schedule
    # is replaced by a recorder: this checks entry ABI/admission, not INT4 math.
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            f"-I{STUBS}",
            f"-I{ROOT / 'tools/qwen4exp'}",
            f"-I{directory / 'frozen'}",
            str(STUBS / "projection_harness.cpp"),
            "-o",
            str(library),
        ],
        check=True,
    )
    handle = ctypes.CDLL(str(library))
    function = handle.inspect_projection_entry
    function.argtypes = [ctypes.c_void_p] * 3
    function.restype = ctypes.c_int
    return function


@pytest.mark.parametrize(
    "active,expected",
    [(0, [0, 0, 0]), (1, [1, 32, 160]), (8192, [1, 32, 160]), (8193, [1, 128, 64]), (20480, [1, 128, 64])],
)
def test_projection_entry_preserves_local_count_schedule_and_skips_empty_rank(projection_entry, active, expected):
    descriptor = torch.tensor([20480, 128, 2560, 640, 8, 0, 1])
    ends = torch.full((128,), active, dtype=torch.int64)
    selection = torch.full((3,), -1, dtype=torch.int64)
    assert projection_entry(descriptor.data_ptr(), ends.data_ptr(), selection.data_ptr()) == 0
    assert selection.tolist() == expected


@pytest.mark.parametrize("field,value", [(0, 128), (0, 20481), (1, 127), (2, 1280), (3, 2560), (4, 1), (5, 1), (6, 2)])
def test_projection_entry_rejects_other_abis_before_schedule(projection_entry, field, value):
    descriptor = torch.tensor([20480, 128, 2560, 640, 8, 0, 1])
    descriptor[field] = value
    ends = torch.zeros(128, dtype=torch.int64)
    selection = torch.full((3,), -1, dtype=torch.int64)
    assert projection_entry(descriptor.data_ptr(), ends.data_ptr(), selection.data_ptr()) == 0
    assert selection.tolist() == [0, 0, 0]
