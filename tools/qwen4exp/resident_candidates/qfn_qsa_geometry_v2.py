# SPDX-License-Identifier: Apache-2.0
"""Canonical geometry replacement; install only after private graph gate."""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path

import torch

from tools.qwen4exp import qsa_geometry_v2 as qsa_geometry

TARGETS = ("vllm_ascend.models.qwen4_exp.ops.qsa_indexer:_qsa_position_geometry",)
QUALIFIED_SOURCE_SHA256 = "2795de0d0314cecb2477b3260b2861856aff69c4373d4716fb51cec5f7b2fd2c"
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
MAX_CAPTURE_CONTRACTS = 32


def record_capture_contract(resource, positions, compress_ratio, capacity):
    """Cold capture metadata only; never count replays or read tensor contents."""
    if not isinstance(positions, torch.Tensor):
        return
    reason = "ready"
    if positions.dtype != torch.int64:
        reason = "dtype"
    elif positions.ndim != 1 or positions.shape[0] not in qsa_geometry.ROWS:
        reason = "shape"
    elif not positions.is_contiguous():
        reason = "stride"
    elif not qsa_geometry._on_npu(positions.device):
        reason = "device"
    elif type(compress_ratio) is not int or compress_ratio != qsa_geometry.RATIO:
        reason = "ratio"
    elif type(capacity) is not int or not INT64_MIN <= capacity <= INT64_MAX:
        reason = "capacity"
    elif (positions.shape[0], positions.device) not in resource._configs:
        reason = "descriptor"
    entry = dict(
        shape=list(positions.shape),
        dtype=str(positions.dtype),
        stride=list(positions.stride()),
        ratio=compress_ratio if type(compress_ratio) is int else type(compress_ratio).__name__,
        capacity=capacity if type(capacity) is int else type(capacity).__name__,
        reason=reason,
    )
    if entry not in resource.capture_contracts and len(resource.capture_contracts) < MAX_CAPTURE_CONTRACTS:
        resource.capture_contracts.append(entry)


def capture_inventory(worker):
    """Cold inventory seam for the controller; no NPU calls or pointer output."""
    state = getattr(worker, STATE_ATTRIBUTE, {})
    resource = state.get("resources", {}).get("geometry")
    return dict(
        loaded=state.get("loaded", False),
        failed=state.get("failed", False),
        contracts=[] if resource is None else [dict(entry) for entry in resource.capture_contracts],
    )


def build_geometry(original, resource):
    original = getattr(original, "_qfn_qsa_geometry_v2_base", getattr(original, "_qfn_qsa_geometry_base", original))
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_SOURCE_SHA256:
        raise RuntimeError("geometry requires the exact pinned position helper")

    def geometry(positions, compress_ratio, capacity):
        if qsa_geometry._is_capturing():
            record_capture_contract(resource, positions, compress_ratio, capacity)
        if (
            not qsa_geometry.eligible(positions, compress_ratio)
            or type(capacity) is not int
            or not INT64_MIN <= capacity <= INT64_MAX
        ):
            return original(positions, compress_ratio, capacity)
        if resource.failed:
            raise RuntimeError("geometry resource failed; preserve its capture owners")
        if (positions.shape[0], positions.device) not in resource._configs:
            # Never allocate descriptors during capture or evict existing ones.
            return original(positions, compress_ratio, capacity)
        return resource(positions, compress_ratio, capacity)

    geometry._qfn_qsa_geometry_v2_base = original
    return geometry


def require_private_gate(gate):
    receipt = gate.get("receipt", {})
    if (
        gate.get("failed") is not False
        or gate.get("retired") is not True
        or receipt.get("passed") is not True
        or receipt.get("rows") != list(qsa_geometry.ROWS)
        or receipt.get("ratio") != qsa_geometry.RATIO
    ):
        raise RuntimeError("geometry requires a passed, cleanly retired private graph gate")
    cases = receipt.get("cases", ())
    for rows in qsa_geometry.ROWS:
        for offset in range(0, len(qsa_geometry.TEST_POSITIONS), rows):
            capture = [
                case
                for case in cases
                if case.get("rows") == rows and case.get("offset") == offset and "capacity" not in case
            ]
            if len(capture) != 1 or not capture[0].get("captured_exact") or not capture[0].get("inputs_preserved"):
                raise RuntimeError("geometry captured changing-input qualification is incomplete")
            for capacity in qsa_geometry.TEST_CAPACITIES:
                eager = [
                    case
                    for case in cases
                    if case.get("rows") == rows and case.get("offset") == offset and case.get("capacity") == capacity
                ]
                if len(eager) != 1 or not eager[0].get("eager_exact"):
                    raise RuntimeError("geometry eager capacity qualification is incomplete")


def make_replacements(resource, gate):
    require_private_gate(gate)
    if resource.failed or resource.device is None or len(resource._configs) != qsa_geometry.MAX_CONFIGS:
        raise RuntimeError("geometry requires a healthy, fully prewarmed resource")
    from tools.qwen4exp.resident_candidates import qsa_graph_selection
    from vllm_ascend.models.qwen4_exp.ops import qsa_indexer

    if qsa_graph_selection._qsa_position_geometry.__module__ != qsa_graph_selection.__name__:
        raise RuntimeError("reload qsa_graph_selection while paused before geometry binding")
    return {TARGETS[0]: build_geometry(qsa_indexer._qsa_position_geometry, resource)}


RPC_TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"
STATE_ATTRIBUTE = "_qfn_qsa_geometry_v2_probe"
MEMORY_RESERVE_BYTES = 64 << 20


def parse_request(payload):
    request = json.loads(payload)
    if not isinstance(request, dict) or set(request) - {"action", "build", "repeats", "replays"}:
        raise ValueError("geometry probe accepts action, build, repeats and replays")
    if request.get("action") not in ("preflight", "load", "validate"):
        raise ValueError("unknown geometry action")
    if not isinstance(request.get("build"), str) or not Path(request["build"]).is_absolute():
        raise ValueError("geometry requires an absolute immutable build path")
    repeats, replays = request.get("repeats", 3), request.get("replays", 20)
    if type(repeats) is not int or not 1 <= repeats <= 5 or type(replays) is not int or not 1 <= replays <= 100:
        raise ValueError("geometry repetitions outside bounded range")
    return dict(action=request["action"], build=str(Path(request["build"]).resolve()), repeats=repeats, replays=replays)


def run_action(worker, request):
    if getattr(worker.model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("pause(wait) and drain requests before private geometry gates")
    if any(
        getattr(worker, name, None) is not None
        for name in ("profiler", "_qfn_cpu_profile", "_qfn_native_profile_session")
    ):
        raise RuntimeError("stop profilers before private geometry gates")
    if qsa_geometry._is_capturing():
        raise RuntimeError("private geometry gate must begin outside capture")
    state = getattr(worker, STATE_ATTRIBUTE, None)
    if state is None:
        state = dict(build=request["build"], resources={}, gate={}, preflight=False, loaded=False, failed=False)
        setattr(worker, STATE_ATTRIBUTE, state)
    if state["build"] != request["build"] or state["failed"]:
        raise RuntimeError("geometry build already owned or failed; do not replace, retry or unload")
    free, total = torch.npu.mem_get_info()
    memory = dict(
        passed=free >= MEMORY_RESERVE_BYTES, free_bytes=free, total_bytes=total, required_bytes=MEMORY_RESERVE_BYTES
    )
    if request["action"] == "preflight":
        qsa_geometry.verify_build(request["build"])
        state["preflight"] = memory["passed"]
        return memory
    if not state["preflight"]:
        raise RuntimeError("all-rank geometry preflight is required")
    if not memory["passed"]:
        return memory
    device = torch.device("npu", torch.npu.current_device())
    if request["action"] == "load" and state["loaded"]:
        return dict(passed=True, already_loaded=True)
    if request["action"] == "validate" and not state["loaded"]:
        raise RuntimeError("load geometry resources before graph validation")
    if request["action"] == "validate" and state["gate"]:
        require_private_gate(state["gate"])
        return state["gate"]["receipt"]
    state["failed"] = True
    try:
        if request["action"] == "load":
            qsa_geometry.load_resources(request["build"], owner=state["resources"])
            state["resources"]["geometry"].prewarm(device)
            torch.npu.synchronize(device)
            state["loaded"], state["failed"] = True, False
            return dict(passed=True, loaded=True, descriptors=2)
        receipt = qsa_geometry.validate_graphs(
            state["resources"]["geometry"],
            device,
            owner=state["gate"],
            repeats=request["repeats"],
            replays=request["replays"],
        )
        if receipt.get("passed") is True:
            require_private_gate(state["gate"])
            state["failed"] = False
        return receipt
    except BaseException as error:
        state.setdefault("failure", error)
        state.setdefault("failure_traceback", error.__traceback__)
        raise


def make_model_replacements(worker):
    state = getattr(worker, STATE_ATTRIBUTE, None)
    if not isinstance(state, dict) or state.get("failed") or not state.get("loaded"):
        raise RuntimeError("geometry model binding requires healthy private worker qualification")
    if getattr(worker.model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("drain model requests before geometry binding")
    return make_replacements(state["resources"]["geometry"], state["gate"])


def make_rpc(original):
    original = getattr(original, "_qfn_qsa_geometry_v2_probe_base", original)

    def rpc(self, output=None):
        if not isinstance(output, str) or not output.lstrip().startswith("{"):
            return original(self, output)
        try:
            request = parse_request(output)
            return dict(rank=getattr(self, "rank", None), action=request["action"], **run_action(self, request))
        except BaseException as error:
            return dict(rank=getattr(self, "rank", None), passed=False, error=f"{type(error).__name__}: {error}")

    rpc._qfn_qsa_geometry_v2_probe_base = original
    return rpc


def replacements():
    from vllm_ascend._310p.worker_310p import NPUWorker310

    return {RPC_TARGET: make_rpc(NPUWorker310.qfn_cpu_profile_stop)}
