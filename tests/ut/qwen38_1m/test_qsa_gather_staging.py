# SPDX-License-Identifier: Apache-2.0
import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import qsa_gather_staging as qsa


def inputs(rows=3, heads=1, width=4096):
    cache = torch.zeros(40, heads * 16, 64, 16, dtype=torch.float16)
    selection = SimpleNamespace(
        group_indices=torch.zeros(rows, 512, dtype=torch.int64),
        group_counts=torch.zeros(rows, dtype=torch.int64),
        tail_starts=torch.zeros(rows, dtype=torch.int64),
        tail_counts=torch.zeros(rows, dtype=torch.int64),
    )
    table = torch.zeros(1, width, dtype=torch.int64)
    return cache, selection, table


def native(monkeypatch):
    monkeypatch.setattr(qsa, "_on_npu", lambda _: True)
    monkeypatch.setattr(qsa, "_is_capturing", lambda: False)
    monkeypatch.setattr(
        qsa,
        "_npu_module",
        lambda: SimpleNamespace(
            get_npu_format=lambda _: 29,
            empty_with_format=lambda *, size, dtype, device, acl_format: torch.empty(size, dtype=dtype, device=device),
        ),
    )
    monkeypatch.setattr(qsa, "_descriptor_ready", lambda: object())
    monkeypatch.setattr(qsa, "_wait_descriptor", lambda _: None)
    calls = []
    return qsa.GatherStaging({4: "kernel4", 8: "kernel8"}, lambda *args: calls.append(args)), calls


@pytest.mark.parametrize("rows,heads,width,transpose", [(1, 1, 2048, False), (3, 2, 4096, True), (64, 2, 4096, False)])
def test_descriptor_exact_pinned_fields(rows, heads, width, transpose):
    fields, blocks = qsa.descriptor((rows, heads, width, transpose, torch.device("cpu")))
    assert fields == [
        rows,
        heads,
        16,
        heads * 16,
        64,
        512,
        129,
        width,
        rows * heads,
        (rows * heads + blocks - 1) // blocks,
        int(transpose),
    ]
    assert blocks == min(rows * heads, 8)


@pytest.mark.parametrize("transpose", [False, True])
def test_native_abi_reuses_output_and_descriptors(monkeypatch, transpose):
    resource, calls = native(monkeypatch)
    cache, selection, table = inputs()
    output = resource(cache, selection, table, head_dim=256, transpose_output=transpose, variant=4)
    assert (
        resource(cache, selection, table, head_dim=256, transpose_output=transpose, output=output, variant=8) is output
    )
    assert len(resource._configs) == 1
    for kernel, args, blocks in calls:
        assert kernel in ("kernel4", "kernel8") and blocks == 3
        assert len(args) == 8 and args[0] is cache and args[6] is output
        assert all(t.dtype == torch.int32 for t in args[1:6])
        assert args[7].tolist() == [3, 1, 16, 16, 64, 512, 129, 4096, 3, 1, int(transpose)]


@pytest.mark.parametrize("cached", [False, True])
def test_capture_rejected_before_geometry_and_cached_descriptors(monkeypatch, cached):
    resource, calls = native(monkeypatch)
    args = inputs()
    if cached:
        resource(*args, head_dim=256, transpose_output=False)
    before = len(calls)
    monkeypatch.setattr(qsa, "_is_capturing", lambda: True)
    monkeypatch.setattr(qsa, "geometry", lambda *a: pytest.fail("capture consulted geometry"))
    with pytest.raises(RuntimeError, match="eager-only"):
        resource(*args, head_dim=256, transpose_output=False)
    assert len(calls) == before


@pytest.mark.parametrize(
    "bad", ["rows", "heads", "width", "groups", "dtype", "strides", "output", "format", "head_dim"]
)
def test_admission_rejects_unsupported_layout(monkeypatch, bad):
    native(monkeypatch)
    cache, selection, table = inputs()
    dim, output = 256, None
    if bad == "rows":
        cache, selection, table = inputs(65)
    if bad == "heads":
        cache, selection, table = inputs(heads=3)
    if bad == "width":
        cache, selection, table = inputs(width=4097)
    if bad == "groups":
        selection.group_indices = selection.group_indices[:, :16]
    if bad == "dtype":
        selection.group_counts = selection.group_counts.float()
    if bad == "strides":
        table = torch.zeros(1, 8192, dtype=torch.int32)[:, ::2]
    if bad == "output":
        output = torch.empty(3, 1, 2064, 256, dtype=torch.float32)
    if bad == "format":
        monkeypatch.setattr(qsa, "_npu_module", lambda: SimpleNamespace(get_npu_format=lambda _: 0))
    if bad == "head_dim":
        dim = 128
    assert qsa.geometry(cache, selection, table, dim, False, output) is None


def test_failure_retains_argument_frames(monkeypatch):
    resource, _ = native(monkeypatch)
    args = inputs()
    ref = weakref.ref(args[0])

    def fail(*args):
        raise RuntimeError("native error")

    resource.launch = fail
    with pytest.raises(RuntimeError, match="native error"):
        resource(*args, head_dim=256, transpose_output=False)
    assert resource.failed and resource.failure_traceback is not None
    del args
    gc.collect()
    assert ref() is not None
    with pytest.raises(RuntimeError, match="retain owner"):
        resource(None, None, None, head_dim=256, transpose_output=False)


def test_raw_storage_bytes_preserve_signed_zero_nan_and_refuse_offset():
    tensor = torch.tensor([0x8000, 0x7E01, 0x0001], dtype=torch.int32).short().view(torch.float16)
    assert torch.equal(qsa.raw_bytes(tensor), tensor.view(torch.uint8))
    with pytest.raises(ValueError, match="standalone"):
        qsa.raw_bytes(tensor[1:])


def bundle(tmp_path):
    header = Path("csrc/attention/qsa_gather_value_nz_v310/op_kernel/qsa_gather_value_nz_v310.h").read_text()
    sources = {}
    for name in qsa.REQUIRED_SOURCES:
        value = header if name == "qsa_gather_value_nz_v310.h" else name
        if name == "qsa_gather_value_nz_v310_tiling_data.h":
            value = Path("csrc/attention/qsa_gather_value_nz_v310/op_kernel", name).read_text()
        if name in ("qsa_gather_staging_4.h", "qsa_gather_staging_8.h"):
            variant = name.split("_")[-1][0]
            value = header.replace(
                "constexpr int64_t TOKEN_BLOCKS_PER_TILE = 4;", f"constexpr int64_t TOKEN_BLOCKS_PER_TILE = {variant};"
            )
        (tmp_path / name).write_text(value)
        sources[name] = qsa._digest(tmp_path / name)
    binaries = {}
    for variant in (4, 8):
        path = tmp_path / f"native{variant}.so"
        path.write_bytes(b"native")
        binaries[f"native_qsa_gather_staging_{variant}"] = dict(
            path=str(path), sha256=qsa._digest(path), entry=f"qwen_qsa_gather_staging_{variant}_v1"
        )
    bridge = tmp_path / "bridge.so"
    bridge.write_bytes(b"bridge")
    manifest = dict(
        namespace=qsa.NAMESPACE,
        native_task_type="KERNEL_TYPE_AICORE",
        bridge_source="native_owned_bridge.cpp",
        flush_host_queue=True,
        eager_only=True,
        sources=sources,
        bridge=dict(path=str(bridge), sha256=qsa._digest(bridge)),
        binaries=binaries,
    )
    (tmp_path / "provenance.json").write_text(json.dumps(manifest))
    return manifest


def test_manifest_valid(tmp_path):
    expected = bundle(tmp_path)
    assert qsa.verify_build(tmp_path) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "namespace",
        "bridge_source",
        "flush_host_queue",
        "eager_only",
        "missing",
        "entry",
        "digest",
        "changed_math",
        "task_type",
        "tiling",
    ],
)
def test_manifest_rejects_incomplete_or_changed_build(tmp_path, bad):
    manifest = bundle(tmp_path)
    if bad in ("namespace", "bridge_source"):
        manifest[bad] = "wrong"
    if bad in ("flush_host_queue", "eager_only"):
        manifest[bad] = False
    if bad == "tiling":
        manifest["sources"]["qsa_gather_value_nz_v310_tiling_data.h"] = "wrong"
    if bad == "task_type":
        manifest["native_task_type"] = "KERNEL_TYPE_AIV_ONLY"
    if bad == "missing":
        del manifest["sources"]["shared_launch_arguments.h"]
    if bad == "entry":
        manifest["binaries"]["native_qsa_gather_staging_8"]["entry"] = "wrong"
    if bad == "digest":
        (tmp_path / "bridge.so").write_bytes(b"changed")
    if bad == "changed_math":
        path = tmp_path / "qsa_gather_staging_8.h"
        path.write_text(path.read_text() + "\n// changed\n")
        manifest["sources"][path.name] = qsa._digest(path)
    (tmp_path / "provenance.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        qsa.verify_build(tmp_path)


def test_private_inputs_cover_boundary_counts_tails_and_safe_small_cache(monkeypatch):
    monkeypatch.setattr(
        qsa,
        "_selection_type",
        lambda: (
            lambda *args: SimpleNamespace(
                **dict(zip(("group_indices", "group_counts", "tail_starts", "tail_counts"), args, strict=True))
            )
        ),
    )
    monkeypatch.setattr(qsa, "_npu_module", lambda: SimpleNamespace(npu_format_cast=lambda tensor, _: tensor))
    counts, tails = set(), set()
    for step in range(9):
        cache, selection, table = qsa.private_inputs(3, 2, 4096, step, "cpu")
        assert cache.shape == (40, 32, 64, 16) and table.shape == (1, 4096)
        assert table.min() >= 0 and table.max() < 40
        assert selection.group_indices.max() < 512
        counts.update(selection.group_counts.tolist())
        tails.update(selection.tail_counts.tolist())
    assert counts == set(qsa.COUNTS) and tails == {0, 1, 2, 3}


def test_descriptor_dependency_waits_only_on_other_stream(monkeypatch):
    calls = []
    event = object()
    producer = SimpleNamespace(record_event=lambda: event)
    consumer = SimpleNamespace(wait_event=lambda value: calls.append(value))
    fake = SimpleNamespace(current_stream=lambda: producer)
    monkeypatch.setattr(torch, "npu", fake, raising=False)
    ready = qsa._descriptor_ready()
    assert ready == (producer, event)
    qsa._wait_descriptor(ready)
    assert not calls
    fake.current_stream = lambda: consumer
    qsa._wait_descriptor(ready)
    assert calls == [event]


def test_cached_descriptor_does_not_repack_fields_or_recreate_events(monkeypatch):
    resource, calls = native(monkeypatch)
    args = inputs()
    output = resource(*args, head_dim=256, transpose_output=True)
    monkeypatch.setattr(qsa, "descriptor", lambda _: pytest.fail("repacked cached descriptor"))
    monkeypatch.setattr(qsa, "_descriptor_ready", lambda: pytest.fail("recreated descriptor event"))
    assert resource(*args, head_dim=256, transpose_output=True, output=output) is output
    assert len(calls) == 2


def test_admitted_internal_path_uses_exact_validated_key_without_double_admission(monkeypatch):
    resource, calls = native(monkeypatch)
    cache, selection, table = inputs()
    key = qsa.geometry(cache, selection, table, 256, True)
    monkeypatch.setattr(qsa, "geometry", lambda *a: pytest.fail("duplicate geometry admission"))
    monkeypatch.setattr(qsa, "_is_capturing", lambda: pytest.fail("duplicate capture admission"))
    output = resource.launch_admitted(key, cache, selection, table)
    assert output.shape == (3, 1, 256, 2064)
    assert calls[0][1][7].tolist()[-1] == 1


def test_admitted_path_still_fails_closed_after_native_error(monkeypatch):
    resource, _ = native(monkeypatch)
    args = inputs()
    key = qsa.geometry(*args, 256, True)
    resource.failed = True
    with pytest.raises(RuntimeError, match="retain owner"):
        resource.launch_admitted(key, *args)


def legacy_state(monkeypatch):
    resource, _ = native(monkeypatch)
    resource(*inputs(), head_dim=256, transpose_output=True)
    old = SimpleNamespace(
        kernels=resource.kernels,
        launch=resource.launch,
        _configs=resource._configs,
        _config_events=resource._config_events,
        failed=False,
    )
    return dict(loaded=True, failed=False, resources=dict(eager_only=True, gather=old)), old


def test_explicit_wrapper_upgrade_preserves_every_native_owner_and_event(monkeypatch):
    state, old = legacy_state(monkeypatch)
    upgraded = qsa.upgrade_owned_wrapper(state)
    assert upgraded is state["resources"]["gather"] and upgraded is not old
    assert upgraded.kernels is old.kernels and upgraded.launch is old.launch
    assert state["previous_gather_wrappers"] == [old]
    for key in old._configs:
        assert upgraded._configs[key] is old._configs[key]
        assert upgraded._config_events[key] is old._config_events[key]
        assert upgraded._config_blocks[key] == qsa.descriptor(key)[1]
    assert qsa.upgrade_owned_wrapper(state) is upgraded
    assert len(state["previous_gather_wrappers"]) == 1


@pytest.mark.parametrize("bad", ["capture", "failed", "unloaded", "resource_failed", "events", "unknown_version"])
def test_wrapper_upgrade_rejects_uncertain_owners_without_mutation(monkeypatch, bad):
    state, old = legacy_state(monkeypatch)
    if bad == "capture":
        monkeypatch.setattr(qsa, "_is_capturing", lambda: True)
    if bad == "failed":
        state["failed"] = True
    if bad == "unloaded":
        state["loaded"] = False
    if bad == "resource_failed":
        old.failed = True
    if bad == "events":
        old._config_events = {}
    if bad == "unknown_version":
        old.WRAPPER_VERSION = 3
    with pytest.raises(RuntimeError):
        qsa.upgrade_owned_wrapper(state)
    assert state["resources"]["gather"] is old and "previous_gather_wrappers" not in state


def test_diagnostic_timing_distinguishes_admitted_ceiling_and_real_model(monkeypatch):
    from contextlib import nullcontext

    from tools.qwen4exp.resident_candidates import qfn_qsa_gather_staging as binding

    resource, calls = native(monkeypatch)
    baseline_calls = []

    def baseline(cache, selection, table, *, head_dim, transpose_output, output=None):
        baseline_calls.append(transpose_output)
        if output is not None:
            return output
        return torch.empty((1, 1, 256, 2064) if transpose_output else (1, 1, 2064, 256), dtype=torch.float16)

    monkeypatch.setattr(qsa.importlib, "import_module", lambda _: SimpleNamespace(_qsa_gather_nz_310=baseline))
    monkeypatch.setattr(qsa, "private_inputs", lambda *a: inputs(1))
    factory = binding.build_forward
    source = (Path(__file__).parent / "fixtures/qfn_pinned_qsa_gather.txt").read_text()
    monkeypatch.setattr(binding, "build_forward", lambda original, owned: factory(original, owned, source=source))

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=Event, synchronize=lambda: None), raising=False)
    labels = []
    with torch.inference_mode():
        result = qsa.timing(
            resource,
            "cpu",
            rows=[1],
            repeats=1,
            diagnostic_routes=True,
            record_function=lambda label: labels.append(label) or nullcontext(),
        )
    assert {r["variant"] for r in result["records"]} == {"installed", 4, 8, "admitted4", "admitted8", "model8"}
    assert len(result["records"]) == 24 and len(labels) == 48
    # Installed handles all4 cases twice, and the actual K-only binding adds
    # only the4 V calls. Preadmitted V calls still exercise the native route.
    assert len(baseline_calls) == 12 and baseline_calls.count(False) == 8
    assert len(calls) == 36
