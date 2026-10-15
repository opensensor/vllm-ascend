# SPDX-License-Identifier: Apache-2.0
"""Paused private GDN projection gate RPC. No model method is replaced."""

import json

import torch

from tools.qwen4exp import gdn_projection_gate as gate

RPC_TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"
STATE_ATTRIBUTE = "_qfn_gdn_projection_probe"


def parse_request(payload):
    request = json.loads(payload)
    if not isinstance(request, dict) or set(request) - {"action", "module", "rows", "repeats", "replays"}:
        raise ValueError("GDN probe accepts action, module, rows, repeats and replays only")
    if request.get("action") not in ("preflight", "load", "validate"):
        raise ValueError("unknown GDN probe action")
    module = request.get("module")
    if module is not None and (not isinstance(module, str) or not module):
        raise ValueError("GDN module must be a nonempty exact resident module path")
    rows = request.get("rows", list(gate.ROWS))
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) > len(gate.ROWS)
        or any(type(value) is not int or value not in gate.ROWS for value in rows)
        or len(set(rows)) != len(rows)
    ):
        raise ValueError("GDN rows must be a unique subset of 3, 6, 64")
    repeats, replays = request.get("repeats", 3), request.get("replays", 5)
    if type(repeats) is not int or not 1 <= repeats <= 5 or type(replays) is not int or not 1 <= replays <= 10:
        raise ValueError("GDN repeats require 1..5 and replays require 1..10")
    return dict(action=request["action"], module=module, rows=rows, repeats=repeats, replays=replays)


def _format_weight(weight):
    # Lazy import only in an initialized worker, never during CPU collection.
    import torch_npu

    return torch_npu.npu_format_cast(weight, 29)


def run_action(worker, request, torch_module=torch):
    try:
        return _run_action(worker, request, torch_module)
    except BaseException as error:
        state = getattr(worker, STATE_ATTRIBUTE, None)
        if state is not None and state["failed"]:
            state.setdefault("failure", error)
            state.setdefault("failure_traceback", error.__traceback__)
        raise


def _run_action(worker, request, torch_module):
    if getattr(worker.model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("pause(wait) and drain serving before the GDN gate")
    if any(
        getattr(worker, name, None) is not None
        for name in ("profiler", "_qfn_cpu_profile", "_qfn_native_profile_session")
    ):
        raise RuntimeError("stop both profilers before the private GDN gate")
    if torch_module.npu.is_current_stream_capturing():
        raise RuntimeError("GDN gate cannot be invoked inside capture")
    state = getattr(worker, STATE_ATTRIBUTE, None)
    if state is None:
        state = dict(
            requested_module=request["module"], failed=False, preflight=False, loaded=False, resources={}, receipt=None
        )
        setattr(worker, STATE_ATTRIBUTE, state)
    if state["failed"]:
        raise RuntimeError("GDN gate failed; retain all owners for coordinated recovery")
    if state["requested_module"] != request["module"]:
        raise RuntimeError("one GDN module per worker owner; do not replace or unload it")
    if state["receipt"] is not None:
        raise RuntimeError("GDN gate completed; one-shot owner cannot be reused")
    name, layer, linear = gate.resident_module(worker, request["module"])
    if "module" in state and (name != state["module"] or layer is not state["layer"]):
        raise RuntimeError("resident GDN module identity changed after preflight")
    memory = gate.memory_preflight(torch_module)
    if request["action"] == "preflight":
        state.update(module=name, layer=layer, preflight=memory["passed"])
        return dict(**memory, module=name)
    if not state["preflight"]:
        raise RuntimeError("separate all-rank GDN preflight is required before allocating")
    if not memory["passed"]:
        return dict(**memory, module=name)
    if request["action"] == "load":
        if state["loaded"]:
            return dict(passed=True, module=name, already_loaded=True)
        state["failed"] = True
        gate.prepare(layer, linear, state["resources"], format_weight=_format_weight)
        torch_module.npu.synchronize()
        state["loaded"], state["failed"] = True, False
        return dict(passed=True, module=name, packed_weight_bytes=gate.HIDDEN * (gate.QKV_WIDTH + gate.Z_WIDTH) * 2)
    if not state["loaded"]:
        raise RuntimeError("load one private GDN weight before validation")
    state["failed"] = True
    receipt = gate.validate(
        state["resources"],
        rows=request["rows"],
        repeats=request["repeats"],
        replays=request["replays"],
        torch_module=torch_module,
    )
    torch_module.npu.synchronize()
    if receipt.get("passed") is True:
        state["receipt"], state["failed"] = receipt, False
    return dict(**receipt, module=name)


def make_rpc(original):
    original = getattr(original, "_qfn_gdn_projection_probe_base", original)

    def rpc(self, output=None):
        if not isinstance(output, str) or not output.lstrip().startswith("{"):
            return original(self, output)
        try:
            request = parse_request(output)
            return dict(rank=getattr(self, "rank", None), action=request["action"], **run_action(self, request))
        except BaseException as error:
            return dict(rank=getattr(self, "rank", None), passed=False, error=f"{type(error).__name__}: {error}")

    rpc._qfn_gdn_projection_probe_base = original
    return rpc


def replacements():
    from vllm_ascend._310p.worker_310p import NPUWorker310

    return {RPC_TARGET: make_rpc(NPUWorker310.qfn_cpu_profile_stop)}
