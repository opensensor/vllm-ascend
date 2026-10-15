# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates import qfn_prefill_schedule as candidate


def inputs(*, rows=650, outputs=2560, width=640):
    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    bank = SimpleNamespace(
        backend="native-int4",
        weight=tensor((128, outputs, width // 2), torch.int8),
        weight_scale=tensor((128, outputs, width // 128), torch.float16),
        weight_offset=tensor((128, outputs, width // 128), torch.float16),
        weight_sum=tensor((128, outputs, width // 128), torch.float16),
    )
    prepared = (
        tensor((rows, width // 2), torch.int8),
        tensor((rows, width // 2), torch.int8),
        tensor((rows, width // 128, 8), torch.float32),
        tensor((rows, width // 128, 8), torch.float32),
    )
    return bank, prepared, tensor((128,), torch.int64)


def source():
    return (Path(__file__).parent / "fixtures/qfn_pinned_native_linear.txt").read_text()


def make_binding(monkeypatch, *, eager_only=False):
    monkeypatch.setattr(candidate, "_on_npu", lambda _: True)
    calls = []
    namespace = {"NATIVE_INT4_BACKEND": "native-int4", "calls": calls}
    exec(
        "def original(bank, prepared, ends):\n    calls.append(('original', bank, prepared, ends))\n    return 17\n",
        namespace,
    )

    def resource(bank, prepared, ends):
        calls.append(("native", bank, prepared, ends))
        return 31

    resource._configs = {}
    return candidate.build_forward(namespace["original"], resource, source=source(), eager_only=eager_only), calls


@pytest.mark.parametrize("projection", [(1280, 2560), (2560, 640)])
@pytest.mark.parametrize("rows", [650, 6400, 20480])
def test_qualified_prefill_uses_captured_resource_once(monkeypatch, projection, rows):
    forward, calls = make_binding(monkeypatch)
    bank, prepared, ends = inputs(rows=rows, outputs=projection[0], width=projection[1])
    assert forward(bank, prepared, ends) == 31
    assert calls == [("native", bank, prepared, ends)]


@pytest.mark.parametrize(
    "case",
    [
        "decode",
        "small_grouped",
        "capacity",
        "expert_count",
        "geometry",
        "backend",
        "operand_dtype",
        "weight_dtype",
        "metadata_shape",
        "weight_metadata",
        "device",
        "contiguous",
        "ends_shape",
    ],
)
def test_unsupported_inputs_preserve_installed_operator(monkeypatch, case):
    forward, calls = make_binding(monkeypatch)
    bank, prepared, ends = inputs()
    if case == "decode":
        ends = torch.empty((30,), dtype=torch.int32, device="meta")
    elif case == "small_grouped":
        bank, prepared, ends = inputs(rows=128)
    elif case == "capacity":
        bank, prepared, ends = inputs(rows=20490)
    elif case == "expert_count":
        bank.weight = bank.weight[:127]
    elif case == "geometry":
        bank, prepared, ends = inputs(outputs=1280, width=640)
    elif case == "backend":
        bank.backend = "other"
    elif case == "operand_dtype":
        prepared = (prepared[0].to(torch.int16), *prepared[1:])
    elif case == "weight_dtype":
        bank.weight = bank.weight.to(torch.float16)
    elif case == "metadata_shape":
        prepared = (*prepared[:2], prepared[2][:, :, :1], prepared[3])
    elif case == "weight_metadata":
        bank.weight_sum = bank.weight_sum.to(torch.float32)
    elif case == "device":
        monkeypatch.setattr(candidate, "_on_npu", lambda _: False)
    elif case == "contiguous":
        bank.weight = torch.empty((128, 320, 2560), device="meta", dtype=torch.int8).transpose(1, 2)
    else:
        ends = ends[:127]
    assert forward(bank, prepared, ends) == 17
    assert calls == [("original", bank, prepared, ends)]


def test_decode_exits_before_bank_inspection(monkeypatch):
    forward, calls = make_binding(monkeypatch)
    bank = object()
    ends = torch.empty((30,), dtype=torch.int32, device="meta")
    assert forward(bank, (), ends) == 17
    assert calls[0][0] == "original"


def qualified_worker(resource=None, *, version=1):
    receipts = []
    for projection in candidate.REQUIRED_PROJECTIONS:
        records = []
        for rows in candidate.REQUIRED_ROWS:
            for local, skew in ((0, False), (rows // 4, False), (rows, False), (rows // 4, True)):
                records.append(
                    {
                        "passed": True,
                        "rows": rows,
                        "local_rows": local,
                        "skew": skew,
                        "cases": [
                            {"schedule": schedule, "exact": True, "rows": rows}
                            for schedule in candidate.REQUIRED_SCHEDULES
                        ],
                    }
                )
        receipts.append({"passed": True, "projection": projection, "records": records})
    state = {
        "loaded": True,
        "failed": False,
        "resources": {"local_count": resource or (lambda *args: 1), "small": lambda *args: 2},
        "schedule_receipts": receipts,
    }
    if version != 1:
        state["version"] = version
    if version == 3:
        state["resources"]["eager_only"] = True
    attribute = (candidate.STATE_ATTRIBUTE, candidate.STATE_ATTRIBUTE_V2, candidate.STATE_ATTRIBUTE_V3)[version - 1]
    return SimpleNamespace(model_runner=SimpleNamespace(execute_model_state=None), **{attribute: state})


@pytest.mark.parametrize("mode", ["local_count", "small"])
def test_factory_borrows_owned_resource_without_replacement(monkeypatch, mode):
    worker = qualified_worker()
    state = getattr(worker, candidate.STATE_ATTRIBUTE)
    resource = state["resources"][mode]
    assert candidate.qualified_resource(worker, mode) is resource
    built = []
    bank_type = SimpleNamespace(native_linear=object())
    monkeypatch.setattr(candidate, "_bank_type", lambda: bank_type)
    monkeypatch.setattr(candidate, "build_forward", lambda original, resource: built.append((original, resource)) or 42)
    assert candidate.make_replacements(worker, mode=mode) == {candidate.TARGET: 42}
    assert built == [(bank_type.native_linear, resource)]
    assert state["resources"][mode] is resource


@pytest.mark.parametrize("mode", ["local_count", "small"])
def test_explicit_v2_factory_borrows_only_v2_resource(monkeypatch, mode):
    worker = qualified_worker()
    original_state = getattr(worker, candidate.STATE_ATTRIBUTE)
    second_state = getattr(qualified_worker(), candidate.STATE_ATTRIBUTE)
    second_state["version"] = 2
    setattr(worker, candidate.STATE_ATTRIBUTE_V2, second_state)
    resource = second_state["resources"][mode]
    assert resource is not original_state["resources"][mode]
    monkeypatch.setattr(candidate, "_bank_type", lambda: SimpleNamespace(native_linear=object()))
    monkeypatch.setattr(candidate, "build_forward", lambda original, resource: resource)
    assert candidate.make_replacements(worker, mode=mode, version=2) == {candidate.TARGET: resource}
    assert candidate.qualified_resource(worker, mode) is original_state["resources"][mode]


@pytest.mark.parametrize("case", ["missing", "failed", "unloaded", "ungated", "wrong_version", "no_version"])
def test_v2_never_borrows_qualified_v1_when_v2_is_unavailable(case):
    worker = qualified_worker()
    state = getattr(qualified_worker(), candidate.STATE_ATTRIBUTE)
    state["version"] = 2
    if case != "missing":
        setattr(worker, candidate.STATE_ATTRIBUTE_V2, state)
    if case == "failed":
        state["failed"] = True
    elif case == "unloaded":
        state["loaded"] = False
    elif case == "ungated":
        state["schedule_receipts"] = []
    elif case == "wrong_version":
        state["version"] = 1
    elif case == "no_version":
        state.pop("version")
    with pytest.raises(RuntimeError):
        candidate.qualified_resource(worker, "local_count", version=2)
    assert callable(candidate.qualified_resource(worker, "local_count"))


@pytest.mark.parametrize("version", [True, False, "2", 2.0, 0, 4, None])
def test_reject_invalid_binding_version(version):
    with pytest.raises(ValueError, match="version"):
        candidate.qualified_resource(qualified_worker(), "local_count", version=version)


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize(
    "case",
    [
        "pending",
        "failed",
        "unloaded",
        "missing_resource",
        "missing_projection",
        "missing_rows",
        "missing_pattern",
        "inexact",
        "wrong_schedule",
        "unsynchronized",
    ],
)
def test_preparation_requires_complete_healthy_qualification(case, version):
    worker = qualified_worker(version=version)
    attribute = (candidate.STATE_ATTRIBUTE, candidate.STATE_ATTRIBUTE_V2, candidate.STATE_ATTRIBUTE_V3)[version - 1]
    state = getattr(worker, attribute)
    if case == "pending":
        worker.model_runner.execute_model_state = object()
    elif case == "failed":
        state["failed"] = True
    elif case == "unloaded":
        state["loaded"] = False
    elif case == "missing_resource":
        state["resources"].pop("local_count")
    elif case == "missing_projection":
        state["schedule_receipts"].pop()
    elif case == "missing_rows":
        state["schedule_receipts"][0]["records"] = state["schedule_receipts"][0]["records"][:4]
    elif case == "missing_pattern":
        state["schedule_receipts"][0]["records"].pop()
    elif case == "inexact":
        state["schedule_receipts"][0]["records"][0]["cases"][0]["exact"] = False
    elif case == "wrong_schedule":
        state["schedule_receipts"][0]["records"][0]["cases"][0]["schedule"] = "other"
    else:
        state["schedule_receipts"][0]["passed"] = False
    with pytest.raises(RuntimeError):
        candidate.qualified_resource(worker, "local_count", version=version)


def test_no_implicit_factory_or_source_drift(monkeypatch):
    with pytest.raises(RuntimeError, match="explicitly"):
        candidate.replacements()
    with pytest.raises(ValueError, match="select"):
        candidate.qualified_resource(qualified_worker(), "baseline")
    with pytest.raises(RuntimeError, match="exact qualified"):
        candidate.build_forward(lambda *args: None, None, source=source().replace("weight_sum", "new_weight_sum"))


def test_descriptor_capacity_declines_only_missing_shape(monkeypatch):
    monkeypatch.setattr(candidate, "_on_npu", lambda _: True)
    monkeypatch.setattr(candidate.native_prefill, "MAX_CONFIGS", 1)
    calls = []
    namespace = {"NATIVE_INT4_BACKEND": "native-int4", "calls": calls}
    exec("def original(*args):\n    calls.append('original')\n    return 17\n", namespace)

    def resource(bank, prepared, ends):
        calls.append("native")
        return 31

    bank, prepared, ends = inputs()
    key = ((650, 128, 2560, 640, 8, 0, 1), prepared[0].device)
    descriptor = object()
    resource._configs = {key: descriptor}
    forward = candidate.build_forward(namespace["original"], resource, source=source())
    monkeypatch.setattr(candidate.native_prefill, "_capturing", lambda _: True)
    assert forward(bank, prepared, ends) == 31
    new_bank, new_prepared, new_ends = inputs(rows=660)
    assert forward(new_bank, new_prepared, new_ends) == 17
    assert resource._configs == {key: descriptor} and calls == ["native", "original"]


def test_capturing_unknown_descriptor_falls_back_without_launch(monkeypatch):
    forward, calls = make_binding(monkeypatch)
    monkeypatch.setattr(candidate.native_prefill, "_capturing", lambda _: True)
    assert forward(*inputs()) == 17
    assert calls[0][0] == "original"


@pytest.mark.parametrize("mode", ["local_count", "small"])
def test_v3_factory_borrows_only_v3_and_enables_eager_guard(monkeypatch, mode):
    worker = qualified_worker()
    v1 = getattr(worker, candidate.STATE_ATTRIBUTE)
    v2 = getattr(qualified_worker(version=2), candidate.STATE_ATTRIBUTE_V2)
    v3 = getattr(qualified_worker(version=3), candidate.STATE_ATTRIBUTE_V3)
    setattr(worker, candidate.STATE_ATTRIBUTE_V2, v2)
    setattr(worker, candidate.STATE_ATTRIBUTE_V3, v3)
    snapshots = [dict(state) for state in (v1, v2, v3)]
    resource = v3["resources"][mode]
    built = []
    bank_type = SimpleNamespace(native_linear=object())
    monkeypatch.setattr(candidate, "_bank_type", lambda: bank_type)

    def build(original, selected, *, eager_only):
        built.append((original, selected, eager_only))
        return selected

    monkeypatch.setattr(candidate, "build_forward", build)
    assert candidate.make_replacements(worker, mode=mode, version=3) == {candidate.TARGET: resource}
    assert built == [(bank_type.native_linear, resource, True)]
    assert candidate.qualified_resource(worker, mode) is v1["resources"][mode]
    assert candidate.qualified_resource(worker, mode, version=2) is v2["resources"][mode]
    assert all(state == snapshot for state, snapshot in zip((v1, v2, v3), snapshots, strict=True))


@pytest.mark.parametrize(
    "case", ["missing", "wrong_version", "no_version", "missing_marker", "false_marker", "nonbool_marker"]
)
def test_v3_cannot_borrow_older_qualified_resources_or_omit_eager_marker(case):
    worker = qualified_worker()
    v2 = getattr(qualified_worker(version=2), candidate.STATE_ATTRIBUTE_V2)
    setattr(worker, candidate.STATE_ATTRIBUTE_V2, v2)
    v3 = getattr(qualified_worker(version=3), candidate.STATE_ATTRIBUTE_V3)
    if case != "missing":
        setattr(worker, candidate.STATE_ATTRIBUTE_V3, v3)
    if case == "wrong_version":
        v3["version"] = 2
    elif case == "no_version":
        v3.pop("version")
    elif case == "missing_marker":
        v3["resources"].pop("eager_only")
    elif case == "false_marker":
        v3["resources"]["eager_only"] = False
    elif case == "nonbool_marker":
        v3["resources"]["eager_only"] = 1
    with pytest.raises(RuntimeError):
        candidate.qualified_resource(worker, "local_count", version=3)
    assert callable(candidate.qualified_resource(worker, "local_count"))
    assert candidate.qualified_resource(worker, "local_count", version=2) is v2["resources"]["local_count"]


@pytest.mark.parametrize("cached", [False, True])
def test_v3_capture_guard_precedes_descriptor_lookup_even_for_cached_shapes(monkeypatch, cached):
    monkeypatch.setattr(candidate, "_on_npu", lambda _: True)
    calls = []
    namespace = {"NATIVE_INT4_BACKEND": "native-int4", "calls": calls}
    exec("def original(*args):\n    calls.append('original')\n    return 17\n", namespace)

    def resource(*args):
        pytest.fail("queue-draining resource must never launch during capture")

    bank, prepared, ends = inputs()
    key = ((650, 128, 2560, 640, 8, 0, 1), prepared[0].device)
    configs = {key: object()} if cached else {}
    resource._configs = configs
    snapshot = dict(configs)

    def capturing(device):
        calls.append("capture")
        return True

    monkeypatch.setattr(candidate.native_prefill, "_capturing", capturing)
    monkeypatch.setattr(candidate, "descriptors_available", lambda *args, **kwargs: pytest.fail("descriptor lookup"))
    forward = candidate.build_forward(namespace["original"], resource, source=source(), eager_only=True)
    assert forward(bank, prepared, ends) == 17
    assert calls == ["capture", "original"]
    assert resource._configs is configs and configs == snapshot


def test_v3_eager_qualified_prefill_launches_once(monkeypatch):
    forward, calls = make_binding(monkeypatch, eager_only=True)
    monkeypatch.setattr(candidate.native_prefill, "_capturing", lambda device: False)
    bank, prepared, ends = inputs()
    assert forward(bank, prepared, ends) == 31
    assert calls == [("native", bank, prepared, ends)]


def test_v3_decode_preserves_cheap_original_path_before_capture_or_geometry(monkeypatch):
    forward, calls = make_binding(monkeypatch, eager_only=True)
    monkeypatch.setattr(candidate.native_prefill, "_capturing", lambda device: pytest.fail("decode capture check"))
    bank = object()
    ends = torch.empty((30,), dtype=torch.int32, device="meta")
    assert forward(bank, (), ends) == 17
    assert calls == [("original", bank, (), ends)]
