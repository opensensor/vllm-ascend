# SPDX-License-Identifier: Apache-2.0

import gc
import hashlib
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import hc_combine as hc


def inputs(rows=2, *, device="meta"):
    return (
        torch.empty(rows, 4, 2560, dtype=torch.float16, device=device),
        torch.empty(rows, 2560, dtype=torch.float32, device=device),
        torch.empty(rows, 4, dtype=torch.float32, device=device),
    )


def resource(monkeypatch):
    monkeypatch.setattr(hc, "_on_npu", lambda device: True)
    monkeypatch.setattr(hc, "_is_capturing", lambda: False)
    calls = []
    native = hc.HCCombine(object(), lambda *args: calls.append(args))
    return native, calls


def test_strict_wrapper_retains_descriptor_and_passes_exact_native_abi(monkeypatch):
    native, calls = resource(monkeypatch)
    values = inputs()
    first = native(*values)
    descriptor = native._configs[(2, values[0].device)]
    second = native(*values)
    assert first.shape == second.shape == (2, 4, 2560) and first.dtype == torch.float16
    assert len(native._configs) == 1 and len(calls) == 2
    for kernel, operands, blocks in calls:
        assert kernel is native.kernel and blocks == 8
        assert all(actual is expected for actual, expected in zip(operands[:3], values, strict=True))
        assert operands[4] is descriptor and descriptor.shape == (1,) and descriptor.dtype == torch.int64


@pytest.mark.parametrize(
    "case",
    [
        "zero",
        "capacity",
        "residual_shape",
        "block_shape",
        "gate_shape",
        "half_block",
        "half_gate",
        "float_residual",
        "device",
        "contiguous",
        "non_tensor",
    ],
)
def test_wrapper_rejects_unsupported_operands_before_allocation(monkeypatch, case):
    native, calls = resource(monkeypatch)
    residual, block, gate = inputs()
    if case == "zero":
        residual, block, gate = inputs(0)
    elif case == "capacity":
        residual, block, gate = inputs(hc.MAX_ROWS + 1)
    elif case == "residual_shape":
        residual = residual.flatten(1)
    elif case == "block_shape":
        block = block[:, :1]
    elif case == "gate_shape":
        gate = gate[:, :1]
    elif case == "half_block":
        block = block.half()
    elif case == "half_gate":
        gate = gate.half()
    elif case == "float_residual":
        residual = residual.float()
    elif case == "device":
        block = torch.empty(2, 2560)
    elif case == "contiguous":
        residual = torch.empty(2, 2560, 4, device="meta", dtype=torch.float16).transpose(1, 2)
    else:
        gate = object()
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("unexpected output allocation"))
    with pytest.raises(ValueError, match="FP16 residual"):
        native(residual, block, gate)
    assert native._configs == {} and not calls


@pytest.mark.parametrize("cached", [False, True])
def test_resource_capture_guard_precedes_cached_descriptor_and_allocation(monkeypatch, cached):
    native, calls = resource(monkeypatch)
    values = inputs()
    if cached:
        native(*values)
        calls.clear()
    descriptors = dict(native._configs)
    monkeypatch.setattr(hc, "_is_capturing", lambda: True)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("capture allocation"))
    with pytest.raises(RuntimeError, match="eager-only"):
        native(*values)
    assert native._configs == descriptors and not calls


def test_descriptor_capacity_never_evicts_cached_shape(monkeypatch):
    native, calls = resource(monkeypatch)
    monkeypatch.setattr(hc, "MAX_CONFIGS", 1)
    native(*inputs())
    descriptor = next(iter(native._configs.values()))
    with pytest.raises(RuntimeError, match="capacity"):
        native(*inputs(3))
    native(*inputs())
    assert len(calls) == 2 and next(iter(native._configs.values())) is descriptor


def test_native_failure_keeps_private_inputs_output_and_earliest_traceback(monkeypatch):
    native, _ = resource(monkeypatch)
    values = inputs(device="cpu")
    refs = [weakref.ref(value) for value in values]

    def fail(kernel, operands, blocks):
        refs.append(weakref.ref(operands[3]))
        raise RuntimeError("uncertain native submission")

    native.launch = fail
    with pytest.raises(RuntimeError, match="uncertain native"):
        native(*values)
    error, traceback = native.failure, native.failure_traceback
    del values
    gc.collect()
    assert native.failed and all(reference() is not None for reference in refs)
    with pytest.raises(RuntimeError, match="retain"):
        native(*inputs())
    assert native.failure is error and native.failure_traceback is traceback


def test_validation_covers_nonzero_rounding_and_exceptional_values_even_one_repeat():
    seen = []

    def candidate(*values):
        seen.append(tuple(value.clone() for value in values))
        return hc.baseline(*values)

    receipt = hc.validate(candidate, "cpu", rows=(1, 3), repeats=1)
    assert receipt["passed"] and len(receipt["cases"]) == 6
    assert all(case["exact"] and case["inputs_preserved"] for case in receipt["cases"])
    assert torch.count_nonzero(seen[0][0]) > 0
    assert torch.isnan(seen[2][0]).any() and torch.isinf(seen[2][1]).any()
    residual, block, injection = seen[1]
    separate = hc.baseline(residual, block, injection)[0, 3, 8]
    fused = (residual.double() + block.double().unsqueeze(1) * injection.double().unsqueeze(-1)).half()[0, 3, 8]
    assert separate == 0 and fused == 2**-24


def test_all_zero_or_input_mutating_candidate_cannot_qualify():
    assert not hc.validate(lambda residual, *args: torch.zeros_like(residual), "cpu", rows=(1,), repeats=1)["passed"]

    def mutate(residual, block, injection):
        output = hc.baseline(residual, block, injection)
        block.zero_()
        return output

    result = hc.validate(mutate, "cpu", rows=(1,), repeats=1)
    assert not result["passed"] and not all(case["inputs_preserved"] for case in result["cases"])


@pytest.mark.parametrize("rows,repeats", [((), 1), ((0,), 1), ((2561,), 1), ((True,), 1), ((1,), 0), ((1,), True)])
def test_reject_vacuous_or_unbounded_validation(rows, repeats):
    with pytest.raises(ValueError):
        hc.validate(None, "cpu", rows=rows, repeats=repeats)


def frozen_build(tmp_path):
    payload = b"frozen HC artifact"
    checksum = hashlib.sha256(payload).hexdigest()
    for name in (*hc.REQUIRED_SOURCES, "bridge.so", "kernel.bin"):
        (tmp_path / name).write_bytes(payload)
    provenance = dict(
        namespace=hc.NAMESPACE,
        bridge_source="native_owned_bridge.cpp",
        flush_host_queue=True,
        eager_only=True,
        sources={name: checksum for name in hc.REQUIRED_SOURCES},
        bridge=dict(path=str(tmp_path / "bridge.so"), sha256=checksum),
        binaries={"native_hc_combine": dict(path=str(tmp_path / "kernel.bin"), sha256=checksum)},
    )
    return provenance


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "namespace",
        "bridge_source",
        "flush",
        "eager",
        "missing_source",
        "source_digest",
        "binary_digest",
        "path",
        "extra_binary",
    ],
)
def test_build_provenance_is_frozen_and_explicitly_eager(tmp_path, mutation):
    provenance = frozen_build(tmp_path)
    if mutation == "namespace":
        provenance["namespace"] = "qwen_route_finalize_v3"
    elif mutation == "bridge_source":
        provenance["bridge_source"] = "reconstruction_bridge.cpp"
    elif mutation == "flush":
        provenance["flush_host_queue"] = False
    elif mutation == "eager":
        provenance["eager_only"] = 1
    elif mutation == "missing_source":
        provenance["sources"].pop("shared_launch_arguments.h")
    elif mutation == "source_digest":
        (tmp_path / "native_hc_combine.cpp").write_text("changed")
    elif mutation == "binary_digest":
        (tmp_path / "kernel.bin").write_text("changed")
    elif mutation == "path":
        provenance["bridge"]["path"] = "relative.so"
    elif mutation == "extra_binary":
        provenance["binaries"]["other"] = provenance["binaries"]["native_hc_combine"]
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    if mutation is None:
        assert hc.verify_build(tmp_path) == provenance
    else:
        with pytest.raises(ValueError):
            hc.verify_build(tmp_path)


def test_load_publishes_successful_kernel_before_wrapper_construction_failure(monkeypatch):
    kernel, owner = object(), {}
    provenance = dict(bridge=dict(path="/bridge.so"), binaries={"native_hc_combine": dict(path="/kernel.bin")})
    monkeypatch.setattr(hc, "verify_build", lambda build: provenance)
    monkeypatch.setattr(
        hc,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(load_library=lambda path: None, qwen_hc_combine_v1=SimpleNamespace(launch=object())),
            classes=SimpleNamespace(qwen_hc_combine_v1=SimpleNamespace(Kernel=lambda path, entry: kernel)),
        ),
    )
    monkeypatch.setattr(hc, "HCCombine", lambda *args: (_ for _ in ()).throw(RuntimeError("wrapper failed")))
    with pytest.raises(RuntimeError, match="wrapper failed"):
        hc.load_resources("/build", owner=owner)
    assert owner == dict(eager_only=True, kernel=kernel)
