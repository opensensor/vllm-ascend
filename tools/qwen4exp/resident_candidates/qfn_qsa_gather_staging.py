# SPDX-License-Identifier: Apache-2.0
"""Private K/V staging gates and exact-source model binding for native8 K only.

Installed V gathers remain unchanged: dense V regressed in the component gate.
"""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path

import torch

from tools.qwen4exp import qsa_gather_staging

RPC_TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"
MODEL_TARGET = qsa_gather_staging.GATHER_MODULE + ":_qsa_gather_nz_310"
STATE_ATTRIBUTE = "_qfn_qsa_gather_staging_probe"
QUALIFIED_SOURCE_SHA256 = "40a60228fd3469237305d8450970a0b6427127802d95aac2d9a8f786893e6240"
REQUIRED_ROWS = frozenset((1, 3, 64))
MEMORY_REQUIRED_BYTES = 1 << 30


def parse_request(payload):
    request = json.loads(payload)
    if not isinstance(request, dict) or set(request) - {"action", "build", "rows", "repeats", "output"}:
        raise ValueError("QSA staging probe accepts action, build, rows, repeats and profile output only")
    if request.get("action") not in {"preflight", "load", "validate", "timing", "profile", "timing_admitted"}:
        raise ValueError("unknown QSA staging probe action")
    if not isinstance(request.get("build"), str) or not Path(request["build"]).is_absolute():
        raise ValueError("QSA staging probe requires an absolute immutable build directory")
    rows, repeats = request.get("rows", [1, 3, 64]), request.get("repeats", 3)
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 4
        or any(type(value) is not int or not 0 < value <= qsa_gather_staging.MAX_ROWS for value in rows)
        or len(set(rows)) != len(rows)
    ):
        raise ValueError("QSA staging probe rows require 1..4 unique integers within 1..64")
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("QSA staging probe repeats require 1..5")
    output = request.get("output")
    if request["action"] == "profile":
        if not isinstance(output, str) or not Path(output).is_absolute() or repeats > 3:
            raise ValueError("QSA profile requires absolute output and repeats within1..3")
    elif output is not None:
        raise ValueError("output is accepted only for QSA profile")
    result = dict(action=request["action"], build=str(Path(request["build"]).resolve()), rows=rows, repeats=repeats)
    if output is not None:
        result["output"] = str(Path(output).resolve())
    return result


def memory_preflight(torch_module, rows):
    free, total = torch_module.npu.mem_get_info()
    # Covers three H2/T64 NZ outputs (384MiB), private cache and full
    # attention temporaries/streams, plus substantial allocator headroom.
    required = MEMORY_REQUIRED_BYTES
    return dict(
        passed=free >= required,
        free_bytes=free,
        total_bytes=total,
        required_bytes=required,
        scope="private QSA staging numerical/timing gate memory admission",
    )


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
        raise RuntimeError("pause(wait) and drain requests before QSA staging gates")
    if any(
        getattr(worker, name, None) is not None
        for name in ("profiler", "_qfn_cpu_profile", "_qfn_native_profile_session")
    ):
        raise RuntimeError("stop both profilers before QSA staging gates")
    if torch_module.npu.is_current_stream_capturing():
        raise RuntimeError("QSA staging gates are strictly eager-only")
    state = getattr(worker, STATE_ATTRIBUTE, None)
    if state is None:
        state = dict(
            build=request["build"],
            resources={},
            failed=False,
            loaded=False,
            preflight=None,
            validated_rows=set(),
            validation_receipts=[],
        )
        setattr(worker, STATE_ATTRIBUTE, state)
    if state["build"] != request["build"]:
        raise RuntimeError("QSA staging native build is already owned; do not replace or unload it")
    if state["failed"]:
        raise RuntimeError("QSA staging probe failed; retain resources and tensor frames for inspection")
    if request["action"] == "preflight":
        qsa_gather_staging.verify_build(request["build"])
        receipt = memory_preflight(torch_module, request["rows"])
        state["preflight"] = max(request["rows"]) if receipt["passed"] else None
        return receipt
    if state["preflight"] is None or max(request["rows"]) > state["preflight"]:
        raise RuntimeError("all-rank QSA staging memory preflight is required for this capacity")
    memory = memory_preflight(torch_module, request["rows"])
    if not memory["passed"]:
        return memory
    if request["action"] == "load":
        if state["loaded"]:
            return dict(passed=True, already_loaded=True, scope="private QSA staging native ownership")
        state["failed"] = True
        qsa_gather_staging.load_resources(request["build"], owner=state["resources"])
        state["loaded"], state["failed"] = True, False
        return dict(passed=True, loaded=True, memory=memory, scope="private QSA staging native ownership")
    if not state["loaded"]:
        raise RuntimeError("load QSA staging resources before numerical gates")
    if request["action"] in {"timing", "profile", "timing_admitted"} and not set(request["rows"]).issubset(
        state["validated_rows"]
    ):
        raise RuntimeError("validate each QSA staging shape before timing")
    if request["action"] == "profile" and state.get("profile_session") is not None:
        raise RuntimeError("private QSA profiling is one-shot; previous session is retained")
    state["failed"] = True
    device = torch_module.device("npu", torch_module.npu.current_device())
    with torch_module.inference_mode():
        if request["action"] == "profile":
            from tools.qwen4exp.qsa_gather_diagnostic import profile

            receipt = profile(
                worker,
                state,
                device,
                rows=request["rows"],
                repeats=request["repeats"],
                output=request["output"],
                torch_module=torch_module,
            )
        else:
            function = qsa_gather_staging.validate if request["action"] == "validate" else qsa_gather_staging.timing
            kwargs = dict(rows=request["rows"], repeats=request["repeats"])
            if request["action"] == "timing_admitted":
                kwargs["diagnostic_routes"] = True
            receipt = function(state["resources"]["gather"], device, **kwargs)
    torch_module.npu.synchronize()
    if receipt.get("passed") is True:
        state["failed"] = False
        if request["action"] == "validate":
            state["validated_rows"].update(request["rows"])
            state["validation_receipts"].append(receipt)
    return receipt


def build_forward(original, resource, *, source=None, state=None):
    original = getattr(original, "_qfn_qsa_gather_staging_base", original)
    source = textwrap.dedent(inspect.getsource(original)) if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_SOURCE_SHA256:
        raise RuntimeError("QSA staging requires exact qualified gather source")

    def forward(cache, selection, block_table, *, head_dim, transpose_output, output=None):
        kwargs = dict(head_dim=head_dim, transpose_output=transpose_output, output=output)
        if transpose_output is not True:
            return original(cache, selection, block_table, **kwargs)
        if torch.is_grad_enabled() or qsa_gather_staging._is_capturing():
            return original(cache, selection, block_table, **kwargs)
        key = qsa_gather_staging.geometry(cache, selection, block_table, head_dim, transpose_output, output)
        if key is None:
            return original(cache, selection, block_table, **kwargs)
        if state is not None and state["failed"]:
            raise RuntimeError("QSA staging model owner failed; retain resources")
        if key not in resource._configs and len(resource._configs) >= qsa_gather_staging.MAX_CONFIGS:
            return original(cache, selection, block_table, **kwargs)
        try:
            return resource.launch_admitted(key, cache, selection, block_table, variant=8, output=output)
        except BaseException as error:
            if state is not None:
                state["failed"] = True
                state.setdefault("failure", error)
                state.setdefault("failure_traceback", error.__traceback__)
            raise

    forward._qfn_qsa_gather_staging_base = original
    return forward


def _qualified_receipts(state):
    covered, attention = set(), set()
    for receipt in state.get("validation_receipts", ()):
        if receipt.get("passed") is not True:
            continue
        for case in receipt.get("cases", ()):
            if all(case.get(flag) is True for flag in ("exact", "inputs_preserved", "output_reused")):
                covered.add(
                    tuple(case.get(k) for k in ("rows", "heads", "table_width", "transpose", "step", "variant"))
                )
        for case in receipt.get("attention", ()):
            if case.get("exact") is True and case.get("inputs_preserved") is True:
                attention.add(tuple(case.get(k) for k in ("heads", "parallel", "variant")))
    expected = {
        (row, head, width, trans, step, variant)
        for row in REQUIRED_ROWS
        for head in (1, 2)
        for width in (2048, 4096)
        for trans in (False, True)
        for step in range(len(qsa_gather_staging.COUNTS))
        for variant in (4, 8)
    }
    expected_attention = {
        (head, parallel, variant) for head in (1, 2) for parallel in (False, True) for variant in (4, 8)
    }
    return expected.issubset(covered) and expected_attention.issubset(attention)


def make_replacements(worker):
    if getattr(worker.model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("drain requests before preparing QSA staging binding")
    state = getattr(worker, STATE_ATTRIBUTE, None)
    if (
        not isinstance(state, dict)
        or state.get("failed")
        or not state.get("loaded")
        or not REQUIRED_ROWS.issubset(state.get("validated_rows", ()))
        or state.get("resources", {}).get("eager_only") is not True
        or not _qualified_receipts(state)
    ):
        raise RuntimeError("QSA staging requires complete physical-byte and unchanged-attention receipts")
    resource = state["resources"].get("gather")
    if not callable(resource) or not callable(getattr(resource, "launch_admitted", None)):
        raise RuntimeError("qualified QSA staging resource requires explicit paused wrapper upgrade")
    return {MODEL_TARGET: build_forward(_gather_module()._qsa_gather_nz_310, resource, state=state)}


def _gather_module():
    import importlib

    return importlib.import_module(qsa_gather_staging.GATHER_MODULE)


def make_rpc(original):
    original = getattr(original, "_qfn_qsa_gather_staging_probe_base", original)

    def rpc(self, output=None):
        if not isinstance(output, str) or not output.lstrip().startswith("{"):
            return original(self, output)
        try:
            request = parse_request(output)
            return dict(rank=getattr(self, "rank", None), action=request["action"], **run_action(self, request))
        except BaseException as error:
            return dict(rank=getattr(self, "rank", None), passed=False, error=f"{type(error).__name__}: {error}")

    rpc._qfn_qsa_gather_staging_probe_base = original
    return rpc


def replacements():
    from vllm_ascend._310p.worker_310p import NPUWorker310

    return {RPC_TARGET: make_rpc(NPUWorker310.qfn_cpu_profile_stop)}
