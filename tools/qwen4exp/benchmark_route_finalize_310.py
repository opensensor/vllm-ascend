# SPDX-License-Identifier: Apache-2.0
"""Explicit resident gates for route finalization and frozen W4 schedules.

No CLI launches an NPU process. The controller invokes these functions only
inside a paused resident worker, then resumes after reviewing the receipts.
"""

import json
import time
from pathlib import Path

import torch

from tools.qwen4exp.build_route_finalize import QUALIFIED_HEADERS, digest
from tools.qwen4exp.native_prefill import NativeGroupedPrefill
from tools.qwen4exp.route_finalize import RouteFinalizer, baseline, validate

SUPPORTED_NAMESPACES = ("qwen_route_finalize_v1", "qwen_route_finalize_v2", "qwen_route_finalize_v3")
OWNED_BRIDGE_SOURCE = "native_owned_bridge.cpp"
OWNED_ARGUMENTS_SOURCE = "shared_launch_arguments.h"


class QualifiedSchedule(NativeGroupedPrefill):
    def __call__(self, bank, prepared, group_ends):
        if bank.weight.ndim != 3 or bank.weight.shape[0] != 128:
            raise ValueError("qualified schedule requires exactly 128 rank-local experts")
        return super().__call__(bank, prepared, group_ends)


def verify_build(build, *, expected_version=None):
    build = Path(build)
    provenance = json.loads((build / "provenance.json").read_text())
    if provenance["namespace"] not in SUPPORTED_NAMESPACES:
        raise ValueError("route finalization candidate requires exact v1, v2 or v3 namespace")
    if expected_version is not None:
        if type(expected_version) is not int or expected_version not in (1, 2, 3):
            raise ValueError("expected route finalization version must be 1, 2 or 3")
        if provenance["namespace"] != f"qwen_route_finalize_v{expected_version}":
            raise ValueError("route finalization build namespace differs from requested version")
    flush = provenance.get("flush_host_queue", False)
    if type(flush) is not bool or flush != (provenance["namespace"] == "qwen_route_finalize_v3"):
        raise ValueError("flush_host_queue must be true only for the explicit v3 namespace")
    if provenance["namespace"] in SUPPORTED_NAMESPACES[1:] and (
        provenance.get("bridge_source") != OWNED_BRIDGE_SOURCE
        or not {OWNED_BRIDGE_SOURCE, OWNED_ARGUMENTS_SOURCE}.issubset(provenance["sources"])
    ):
        raise ValueError("v2/v3 require the explicitly selected frozen owned bridge and launch argument header")
    if set(provenance["binaries"]) != {"native_route_finalize", "native_prefill_schedule_probe"}:
        raise ValueError("unexpected native binaries")
    for name, expected in QUALIFIED_HEADERS.items():
        if provenance["sources"].get(name) != expected or digest(build / name) != expected:
            raise ValueError("qualified schedule header digest differs")
    for entry in [provenance["bridge"], *provenance["binaries"].values()]:
        path = Path(entry["path"])
        if not path.is_absolute() or path.resolve().parent != build.resolve():
            raise ValueError("native library and binary must reside in the immutable build directory")
        if digest(path) != entry["sha256"]:
            raise ValueError("native binary digest differs from provenance")
    for name, expected in provenance["sources"].items():
        if Path(name).name != name or digest(build / name) != expected:
            raise ValueError("native frozen source digest differs from provenance")
    return provenance


def load_resources(build, *, load_library=False, owner=None, expected_version=None):
    provenance = verify_build(build, expected_version=expected_version)
    if load_library:
        torch.ops.load_library(provenance["bridge"]["path"])
    namespace = provenance["namespace"]
    constructor = getattr(torch.classes, namespace).Kernel
    launch = getattr(torch.ops, namespace).launch
    binaries = provenance["binaries"]
    result = {} if owner is None else owner
    if result:
        raise ValueError("native resource owner must start empty")
    if namespace == "qwen_route_finalize_v3":
        result["eager_only"] = True
    result["finalize"] = RouteFinalizer(
        constructor(binaries["native_route_finalize"]["path"], "qwen_route_finalize_v1"), launch
    )
    for name in ("baseline", "small", "local_count"):
        result[name] = QualifiedSchedule(
            constructor(binaries["native_prefill_schedule_probe"]["path"], f"qwen_prefill_{name}_v1"), launch
        )
    return result


def validate_finalization(resources, *, rows=(10, 650, 6400, 20480)):
    return validate(resources["finalize"], torch.device("npu", torch.npu.current_device()), rows=rows)


def validate_schedule(resources, bank, prepared, group_ends, *, repeats=0):
    """Compare exact qualified operators on caller-owned immutable inputs.

    Caller supplies changing private activations and valid INT64 group ends;
    actual resident bank weights may be borrowed read-only. This gate makes no
    claim about values outside these inputs or model quality/performance.
    """
    if type(repeats) is not int or not 0 <= repeats <= 5:
        raise ValueError("schedule timing repeats must be within 0..5")
    expected = bank.native_linear(prepared, group_ends)
    receipt = []
    for name in ("baseline", "small", "local_count"):
        actual = resources[name](bank, prepared, group_ends)
        exact = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        receipt.append({"schedule": name, "exact": exact, "rows": prepared[0].shape[0]})
        del actual
    passed = all(case["exact"] for case in receipt)
    timings = []
    if passed and repeats:
        del expected
        functions = [("installed", lambda: bank.native_linear(prepared, group_ends))]
        for name in ("baseline", "small", "local_count"):
            functions.append((name, lambda name=name: resources[name](bank, prepared, group_ends)))
        for name, function in functions:
            torch.npu.synchronize()
            elapsed = []
            for _ in range(repeats):
                start = time.perf_counter()
                output = function()
                torch.npu.synchronize()
                elapsed.append((time.perf_counter() - start) * 1000)
                del output
            timings.append({"schedule": name, "wall_ms": elapsed})
    return {"passed": passed, "cases": receipt, "component_timings": timings}


def time_finalization(resources, routed, order, inverse, weights, *, repeats=5):
    """Isolated synchronized operator timings, never an end-to-end speed claim."""
    if not 1 <= repeats <= 20:
        raise ValueError("benchmark repeats must be within 1..20")
    functions = (
        ("qualified", lambda: baseline(routed, order, inverse, weights)),
        ("fused", lambda: resources["finalize"](routed, inverse, weights)),
    )
    rows = []
    for name, function in functions:
        warm_output = function()
        torch.npu.synchronize()
        del warm_output
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = function()
            torch.npu.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
            del result
        rows.append({"mode": name, "wall_ms": elapsed})
    return rows
