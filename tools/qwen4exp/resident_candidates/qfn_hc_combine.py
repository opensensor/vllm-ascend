# SPDX-License-Identifier: Apache-2.0
"""Private HC gate RPC and separately admitted, exact-source model binding."""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path
from types import FunctionType

import torch

from tools.qwen4exp import hc_combine
from tools.qwen4exp.native_descriptor_admission import descriptors_available

RPC_TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"
MODEL_TARGET = "vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine"
STATE_ATTRIBUTE = "_qfn_hc_combine_probe"
QUALIFIED_SOURCE_SHA256 = "bc5999d414ff84cedc1885022ad2e98fd5ae9df405d4840a5878ce8af85297ce"
REQUIRED_ROWS = frozenset((68, 640, 2048))
MIN_PREFILL_ROWS = 64
MEMORY_RESERVE_BYTES = 256 << 20
BYTES_PER_ROW_BUDGET = 256 << 10
RESIDUAL_CAST = "    residual = hyper_input.view(num_tokens, self.hc_count, self.hidden_size).to(self.compute_dtype)"
ORIGINAL_EPILOGUE = """    block = block_output.to(self.compute_dtype).unsqueeze(-2)
    out = residual + block * injection.unsqueeze(-1)
    return out.flatten(-2).to(self.params_dtype)"""
NATIVE_EPILOGUE = """    out = _combine(residual, block_output.to(self.compute_dtype), injection.to(self.compute_dtype))
    return out.flatten(-2)"""


def parse_request(payload):
    request = json.loads(payload)
    if not isinstance(request, dict) or set(request) - {"action", "build", "rows", "repeats"}:
        raise ValueError("HC probe accepts action, build, rows and repeats only")
    if request.get("action") not in {"preflight", "load", "validate", "timing"}:
        raise ValueError("unknown HC probe action")
    if not isinstance(request.get("build"), str) or not Path(request["build"]).is_absolute():
        raise ValueError("HC probe requires an absolute immutable build directory")
    rows, repeats = request.get("rows", [68]), request.get("repeats", 3)
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 4
        or any(type(value) is not int or not 0 < value <= hc_combine.MAX_ROWS for value in rows)
        or len(set(rows)) != len(rows)
    ):
        raise ValueError("HC probe rows require 1..4 unique integers within 1..2560")
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("HC probe repeats require 1..5")
    return dict(action=request["action"], build=str(Path(request["build"]).resolve()), rows=rows, repeats=repeats)


def memory_preflight(torch_module, rows):
    free, total = torch_module.npu.mem_get_info()
    # Covers FP32 oracle residual/product/add, FP16 inputs/output/expected,
    # widened block/gates and allocator headroom; does not count CPU snapshots.
    required = max(rows) * BYTES_PER_ROW_BUDGET + MEMORY_RESERVE_BYTES
    return dict(
        passed=free >= required,
        free_bytes=free,
        total_bytes=total,
        required_bytes=required,
        scope="private HC numerical/timing gate memory admission",
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
        raise RuntimeError("pause(wait) and drain requests before HC gates")
    if any(
        getattr(worker, name, None) is not None
        for name in ("profiler", "_qfn_cpu_profile", "_qfn_native_profile_session")
    ):
        raise RuntimeError("stop both profilers before HC gates")
    if torch_module.npu.is_current_stream_capturing():
        raise RuntimeError("HC gates are strictly eager-only")
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
        raise RuntimeError("HC native build is already owned; do not replace or unload it")
    if state["failed"]:
        raise RuntimeError("HC probe failed; retain resources and tensor frames for inspection")
    if request["action"] == "preflight":
        hc_combine.verify_build(request["build"])
        receipt = memory_preflight(torch_module, request["rows"])
        state["preflight"] = max(request["rows"]) if receipt["passed"] else None
        return receipt
    if state["preflight"] is None or max(request["rows"]) > state["preflight"]:
        raise RuntimeError("all-rank HC memory preflight is required for this capacity")
    memory = memory_preflight(torch_module, request["rows"])
    if not memory["passed"]:
        return memory
    if request["action"] == "load":
        if state["loaded"]:
            return dict(passed=True, already_loaded=True, scope="private HC native ownership")
        state["failed"] = True
        hc_combine.load_resources(request["build"], owner=state["resources"])
        state["loaded"], state["failed"] = True, False
        return dict(passed=True, loaded=True, memory=memory, scope="private HC native ownership")
    if not state["loaded"]:
        raise RuntimeError("load HC resources before numerical gates")
    if request["action"] == "timing" and not set(request["rows"]).issubset(state["validated_rows"]):
        raise RuntimeError("validate each HC shape before timing")
    state["failed"] = True
    device = torch_module.device("npu", torch_module.npu.current_device())
    function = hc_combine.validate if request["action"] == "validate" else hc_combine.timing
    with torch_module.inference_mode():
        receipt = function(state["resources"]["combine"], device, rows=request["rows"], repeats=request["repeats"])
    torch_module.npu.synchronize()
    if receipt.get("passed") is True:
        state["failed"] = False
        if request["action"] == "validate":
            state["validated_rows"].update(request["rows"])
            state["validation_receipts"].append(receipt)
    return receipt


def _eligible_model(module, block, residuals):
    if (
        not module.use_combine
        or module.hc_count != hc_combine.STREAMS
        or module.hidden_size != hc_combine.WIDTH
        or module.compute_dtype != torch.float32
        or module.params_dtype != torch.float16
        or torch.is_grad_enabled()
        or not isinstance(residuals, (tuple, list))
        or len(residuals) != 2
    ):
        return False
    hyper, xn = residuals
    if not all(isinstance(value, torch.Tensor) for value in (hyper, xn, block, module.block_inject_weight)):
        return False
    if hyper.ndim != 2 or not MIN_PREFILL_ROWS <= hyper.shape[0] <= hc_combine.MAX_ROWS:
        return False
    rows = hyper.shape[0]
    return (
        hyper.shape == xn.shape == (rows, hc_combine.STREAMS * hc_combine.WIDTH)
        and block.shape == (rows, hc_combine.WIDTH)
        and hyper.dtype == module.block_inject_weight.dtype == torch.float16
        and xn.dtype in (torch.float16, torch.float32)
        and block.dtype in (torch.float16, torch.float32)
        and module.block_inject_weight.shape == (hc_combine.STREAMS, hc_combine.STREAMS * hc_combine.WIDTH)
        and hc_combine._on_npu(hyper.device)
        and all(
            value.device == hyper.device and value.is_contiguous()
            for value in (hyper, xn, block, module.block_inject_weight)
        )
    )


def build_forward(original, resource, *, source=None, state=None):
    original = getattr(original, "_qfn_hc_combine_base", original)
    source = textwrap.dedent(inspect.getsource(original)) if source is None else source
    if (
        hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_SOURCE_SHA256
        or source.count(RESIDUAL_CAST) != 1
        or source.count(ORIGINAL_EPILOGUE) != 1
    ):
        raise RuntimeError("HC combine binding requires exact qualified combine source")
    body = source.replace(RESIDUAL_CAST, RESIDUAL_CAST.replace(".to(self.compute_dtype)", ""))
    body = body.replace(ORIGINAL_EPILOGUE, NATIVE_EPILOGUE)
    factory = "def factory(_combine):\n" + textwrap.indent(body, "    ") + "    return combine\n"
    namespace = dict(original.__globals__)
    exec(compile(factory, "<qfn-qualified-hc-combine>", "exec"), namespace)
    compiled = namespace["factory"](resource)
    implementation = FunctionType(
        compiled.__code__, original.__globals__, compiled.__name__, compiled.__defaults__, compiled.__closure__
    )

    def forward(self, block_output, residuals):
        if not _eligible_model(self, block_output, residuals):
            return original(self, block_output, residuals)
        if hc_combine._is_capturing():
            return original(self, block_output, residuals)
        if state is not None and state["failed"]:
            raise RuntimeError("HC model resource failed; retain owner for inspection")
        key = (residuals[0].shape[0], residuals[0].device)
        if not descriptors_available(
            resource, (key,), capacity=hc_combine.MAX_CONFIGS, is_capturing=hc_combine._is_capturing
        ):
            return original(self, block_output, residuals)
        try:
            return implementation(self, block_output, residuals)
        except BaseException as error:
            if state is not None:
                state["failed"] = True
                state.setdefault("failure", error)
                state.setdefault("failure_traceback", error.__traceback__)
            raise

    forward._qfn_hc_combine_base = original
    return forward


def make_replacements(worker):
    if getattr(worker.model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("drain model requests before preparing HC binding")
    state = getattr(worker, STATE_ATTRIBUTE, None)
    if (
        not isinstance(state, dict)
        or state.get("failed")
        or not state.get("loaded")
        or not REQUIRED_ROWS.issubset(state.get("validated_rows", ()))
        or state.get("resources", {}).get("eager_only") is not True
    ):
        raise RuntimeError("HC binding requires healthy eager resources and validated rows 68, 640, 2048")
    resource = state["resources"].get("combine")
    if not callable(resource):
        raise RuntimeError("qualified HC combine resource is absent")
    covered = {
        (case.get("rows"), case.get("step"))
        for receipt in state.get("validation_receipts", ())
        if receipt.get("passed") is True
        for case in receipt.get("cases", ())
        if case.get("exact") is True and case.get("inputs_preserved") is True
    }
    if not {(rows, step) for rows in REQUIRED_ROWS for step in range(hc_combine.MIN_VALIDATION_CASES)}.issubset(
        covered
    ):
        raise RuntimeError("HC binding requires complete finite, rounding and exceptional-value parity receipts")
    return {MODEL_TARGET: build_forward(_hc_type().combine, resource, state=state)}


def _hc_type():
    from vllm_ascend.models.qwen4_exp.model import _GatedResidual

    return _GatedResidual


def make_rpc(original):
    original = getattr(original, "_qfn_hc_combine_probe_base", original)

    def rpc(self, output=None):
        if not isinstance(output, str) or not output.lstrip().startswith("{"):
            return original(self, output)
        try:
            request = parse_request(output)
            return dict(rank=getattr(self, "rank", None), action=request["action"], **run_action(self, request))
        except BaseException as error:
            return dict(rank=getattr(self, "rank", None), passed=False, error=f"{type(error).__name__}: {error}")

    rpc._qfn_hc_combine_probe_base = original
    return rpc


def replacements():
    from vllm_ascend._310p.worker_310p import NPUWorker310

    return {RPC_TARGET: make_rpc(NPUWorker310.qfn_cpu_profile_stop)}
