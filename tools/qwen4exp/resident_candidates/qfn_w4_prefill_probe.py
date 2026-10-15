# SPDX-License-Identifier: Apache-2.0
"""Paused, instance-owned native gates; never binds a serving model method.

The controller must drain and preflight every rank before loading any rank.
Native resources are append-only and retained even after a partial failure.
"""

import json
from pathlib import Path

TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"
STATE_ATTRIBUTE = "_qfn_w4_prefill_probe"
STATE_ATTRIBUTE_V2 = "_qfn_w4_prefill_probe_v2"
STATE_ATTRIBUTE_V3 = "_qfn_w4_prefill_probe_v3"
MEMORY_RESERVE_BYTES = 256 * 1024 * 1024
BYTES_PER_ELEMENT_BUDGET = 16
WIDTH = 2560
MAX_ROWS = 20480


def parse_request(payload):
    request = json.loads(payload)
    if not isinstance(request, dict) or set(request) - {"action", "build", "rows", "repeats", "projection", "version"}:
        raise ValueError("probe accepts action, build, rows, repeats, projection and version only")
    version = request.get("version", 1)
    state_attribute(version)
    if request.get("action") not in {"preflight", "load", "validate", "timing", "schedule"}:
        raise ValueError("unknown private prefill probe action")
    build = request.get("build")
    if not isinstance(build, str) or not Path(build).is_absolute():
        raise ValueError("probe requires an absolute immutable build directory")
    rows = request.get("rows", [650])
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 4
        or any(type(row) is not int or not 0 < row <= MAX_ROWS or row % 10 for row in rows)
        or len(set(rows)) != len(rows)
    ):
        raise ValueError("probe rows require 1..4 unique multiples of 10 up to 20480")
    repeats = request.get("repeats", 3)
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("probe repeats require 1..5")
    projection = request.get("projection", "down_proj")
    if projection not in {"gate_up_proj", "down_proj"}:
        raise ValueError("probe projection must be gate_up_proj or down_proj")
    return {
        "action": request["action"],
        "build": str(Path(build).resolve()),
        "rows": rows,
        "repeats": repeats,
        "projection": projection,
        "version": version,
    }


def state_attribute(version):
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("private prefill probe version must be 1, 2 or 3")
    return (STATE_ATTRIBUTE, STATE_ATTRIBUTE_V2, STATE_ATTRIBUTE_V3)[version - 1]


def memory_preflight(torch_module, rows):
    free, total = torch_module.npu.mem_get_info()
    required = max(rows) * WIDTH * BYTES_PER_ELEMENT_BUDGET + MEMORY_RESERVE_BYTES
    return {"passed": free >= required, "free_bytes": free, "total_bytes": total, "required_bytes": required}


def _resident_bank(worker, projection):
    # Borrow existing model weights read-only; never load or copy a second bank.
    model = worker.model_runner.get_model()
    for name, module in model.named_modules():
        if type(module).__name__ == "W4SparseMoE" and getattr(module, "native_int4", False):
            return name, module.projections[projection]
    raise RuntimeError("resident model has no native W4 sparse expert bank")


def run_action(worker, request, torch_module):
    """Keep tensor-bearing failure frames alive until coordinated recovery.

    Native resources alone do not retain every private input/output. In
    particular, a launch or stream failure can occur before a native call has
    returned its output. Retaining the exception traceback preserves those
    frames without adding ownership hooks to the qualified submission path.
    """
    try:
        return _run_action(worker, request, torch_module)
    except BaseException as error:
        state = getattr(worker, state_attribute(request.get("version", 1)), None)
        if state is not None and state["failed"]:
            state.setdefault("failure", error)
            state.setdefault("failure_traceback", error.__traceback__)
        raise


def _run_action(worker, request, torch_module):
    from tools.qwen4exp.benchmark_route_finalize_310 import load_resources, verify_build

    if getattr(worker.model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("pause and drain resident requests before private native gates")
    if any(
        getattr(worker, name, None) is not None
        for name in ("profiler", "_qfn_cpu_profile", "_qfn_native_profile_session")
    ):
        raise RuntimeError("private native gate requires both profilers stopped")
    if torch_module.npu.is_current_stream_capturing():
        raise RuntimeError("private native gate cannot run inside capture")
    version = request.get("version", 1)
    attribute = state_attribute(version)
    state = getattr(worker, attribute, None)
    if state is None:
        state = {
            "build": request["build"],
            "version": version,
            "resources": {},
            "failed": False,
            "loaded": False,
            "preflight": None,
            "validated_rows": set(),
            "schedule_receipts": [],
        }
        setattr(worker, attribute, state)
    if state["build"] != request["build"] or state.get("version", 1) != version:
        raise RuntimeError("private native resource version is already owned; do not replace or unload it")
    if request["action"] == "preflight":
        verify_build(request["build"], expected_version=version)
        receipt = memory_preflight(torch_module, request["rows"])
        state["preflight"] = max(request["rows"]) if receipt["passed"] else None
        return receipt
    if state["failed"]:
        raise RuntimeError("private native gate failed; retain resources for inspection")
    if state["preflight"] is None or max(request["rows"]) > state["preflight"]:
        raise RuntimeError("all-rank memory preflight is required for this row capacity")
    memory = memory_preflight(torch_module, request["rows"])
    if not memory["passed"]:
        return memory
    if request["action"] == "load":
        if state["loaded"]:
            return {"passed": True, "already_loaded": True}
        # Publish ownership before any constructor or native load. Successful
        # constructors are retained even if a later constructor fails.
        state["failed"] = True
        load_resources(request["build"], load_library=True, owner=state["resources"], expected_version=version)
        state["loaded"] = True
        state["failed"] = False
        return {"passed": True, "loaded": sorted(state["resources"]), "memory": memory}
    if not state["loaded"]:
        raise RuntimeError("load private native resources before validation")
    if request["action"] == "timing" and not set(request["rows"]).issubset(state["validated_rows"]):
        raise RuntimeError("validate every finalization shape before timing it")
    state["failed"] = True
    with torch_module.inference_mode():
        receipt = run_validation(worker, state["resources"], request, torch_module)
    torch_module.npu.synchronize()
    if receipt.get("passed") is True:
        state["failed"] = False
        if request["action"] == "validate":
            state["validated_rows"].update(request["rows"])
        elif request["action"] == "schedule":
            state.setdefault("schedule_receipts", []).append(receipt)
    return receipt


def run_validation(worker, resources, request, torch_module):
    from tools.qwen4exp.benchmark_route_finalize_310 import time_finalization, validate_schedule
    from tools.qwen4exp.route_finalize import validate

    device = torch_module.device("npu", torch_module.npu.current_device())
    if request["action"] == "validate":
        return validate(resources["finalize"], device, rows=request["rows"], repeats=request["repeats"])
    generator = torch_module.Generator().manual_seed(517)
    records = []
    if request["action"] == "timing":
        for rows in request["rows"]:
            routed = torch_module.randn(rows, WIDTH, dtype=torch_module.float16, generator=generator).to(device)
            # Setup is outside the measured kernel region. Invert the host
            # permutation before transfer: integer NPU argsort uses AiCPU.
            order_cpu = torch_module.randperm(rows, generator=generator)
            inverse_cpu = torch_module.argsort(order_cpu)
            order, inverse = order_cpu.to(device), inverse_cpu.to(device)
            weights = torch_module.rand(rows, 1, generator=generator).to(device)
            records.append(
                {
                    "rows": rows,
                    "timings": time_finalization(
                        resources, routed, order, inverse, weights, repeats=request["repeats"]
                    ),
                }
            )
            del routed, order, inverse, weights, order_cpu, inverse_cpu
        return {"passed": True, "scope": "component_only", "records": records}
    from vllm_ascend.models.qwen4_exp.w4_moe import pack_activation_device

    name, bank = _resident_bank(worker, request["projection"])
    for rows in request["rows"]:
        if rows <= 128:
            raise ValueError("schedule gate is grouped prefill only, rows must exceed 128")
        inputs = torch_module.randn(rows, bank.weight.shape[2] * 2, dtype=torch_module.float16, generator=generator).to(
            device
        )
        prepared = pack_activation_device(inputs)
        del inputs
        # All peer routes, balanced EP4 work, all-local work and skewed work.
        for local, skew in ((0, False), (rows // 4, False), (rows, False), (rows // 4, True)):
            counts = torch_module.full((128,), local // 128, dtype=torch_module.int64)
            counts[: local % 128] += 1
            if skew:
                counts.zero_()
                counts[71] = local
            ends = counts.cumsum(0).to(device)
            receipt = validate_schedule(resources, bank, prepared, ends, repeats=request["repeats"])
            receipt.update(rows=rows, local_rows=local, skew=skew)
            records.append(receipt)
            del ends
        del prepared
    return {
        "passed": all(record["passed"] for record in records),
        "bank": name,
        "projection": request["projection"],
        "records": records,
    }


def make_rpc(original):
    original = getattr(original, "_qfn_w4_prefill_probe_base", original)

    def rpc(self, output=None):
        if output is None or not isinstance(output, str) or not output.lstrip().startswith("{"):
            return original(self, output)
        import torch

        try:
            request = parse_request(output)
            receipt = run_action(self, request, torch)
            return {
                "rank": getattr(self, "rank", None),
                "action": request["action"],
                "version": request["version"],
                **receipt,
            }
        except BaseException as error:
            return {"rank": getattr(self, "rank", None), "passed": False, "error": f"{type(error).__name__}: {error}"}

    rpc._qfn_w4_prefill_probe_base = original
    return rpc


def replacements():
    from vllm_ascend._310p.worker_310p import NPUWorker310

    return {TARGET: make_rpc(NPUWorker310.qfn_cpu_profile_stop)}
