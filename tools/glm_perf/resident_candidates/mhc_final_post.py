# SPDX-License-Identifier: Apache-2.0
"""Bind only the final, unrounded GLM stream mixer during an idle experiment."""

import torch

from tools.glm_perf.instance_bindings import InstanceBindings
from tools.glm_perf.mhc_post_native import FINAL_POST_ROWS, GLM_HIDDEN_SIZE, supported_inputs
from tools.glm_perf.resident_rpc_guard import WORKER_PREFIX, guard_replacements


def bind_final_mixer(bindings, runner, native, state):
    if not getattr(native, "final_only", False):
        raise ValueError("final mixer requires the separate unrounded native helper")
    root = getattr(runner.model, "runnable", runner.model)
    selected = [
        layer
        for layer in root.modules()
        if hasattr(layer, "mhc_post_op")
        and layer.layer_idx == layer.num_hidden_layers - 1
        and not getattr(layer, "is_mtp_layer", False)
    ]
    if len(selected) != 1:
        raise ValueError("require exactly one loaded target final mHC layer")
    layer = selected[0]
    if (
        layer.n != 4
        or layer.hidden_size != GLM_HIDDEN_SIZE
        or not getattr(layer.mhc_fused_post_pre_op, "use_310p_fp16_mhc_state", False)
    ):
        raise ValueError("final mixer is qualified only for the FP16-rounded four-stream GLM state")
    device = layer.hc_attn_fn.device
    for rows in FINAL_POST_ROWS:
        key = (torch.Size((rows, GLM_HIDDEN_SIZE)), device)
        if key not in native.configs:
            native.configs[key] = torch.tensor([rows, GLM_HIDDEN_SIZE], dtype=torch.int64, device=device)
    owner = layer.mhc_post_op
    original = owner._forward_method.__func__

    def post(self, x, residual, post_layer_mix, comb_res_mix):
        if state.get("actual_gate") and not state["actual_gate"]["passed"]:
            return original(self, x, residual, post_layer_mix, comb_res_mix)
        if supported_inputs(x, residual, post_layer_mix, comb_res_mix, final_only=True):
            state["calls_by_rows"][x.shape[0]] = state["calls_by_rows"].get(x.shape[0], 0) + 1
            output = native(x, residual, post_layer_mix, comb_res_mix)
            if state.get("verify_first_request", False) and not state.get("actual_gate"):
                # Explicit controlled activation gate, before the timed large
                # requests. No device reads remain after this one-time check.
                expected = original(self, x, residual, post_layer_mix, comb_res_mix)
                a, b = output.cpu(), expected.cpu()
                xc, rc, pc, cc = (value.float().cpu() for value in (x, residual, post_layer_mix, comb_res_mix))
                magnitude = (rc.unsqueeze(2).abs() * cc.unsqueeze(-1).abs()).sum(1)
                magnitude.add_(pc.abs() * xc.unsqueeze(1).abs())
                error = (a - b).abs()
                bound = 16 * torch.finfo(torch.float32).eps * magnitude + torch.finfo(torch.float32).tiny
                passed = bool(torch.isfinite(a).all()) and not bool((error > bound).any())
                state["actual_gate"] = dict(
                    rows=x.shape[0],
                    passed=passed,
                    max_abs=float(error.max()) if bool(torch.isfinite(error).all()) else None,
                    scope="16 FP32 epsilon times absolute contribution sum; not language quality",
                )
                if not passed:
                    # Return the established output and report all ranks;
                    # reject the experiment without killing the executor.
                    return expected
            return output
        return original(self, x, residual, post_layer_mix, comb_res_mix)

    bindings.bind(owner, "_forward_method", post)
    return layer.layer_idx


def extend_replacements(changes, native, *, verify_first_request=False):
    bindings = InstanceBindings()
    state = {"layer": None, "calls_by_rows": {}, "verify_first_request": verify_first_request, "actual_gate": None}
    capture_base, apply_base, status_base = (
        changes[WORKER_PREFIX + name] for name in ("resident_capture", "resident_apply", "resident_status")
    )

    def capture(self):
        try:
            if self._resident_session().graphs_dirty:
                state["layer"] = bind_final_mixer(bindings, self.model_runner, native, state)
            result = capture_base(self)
            if "error" in result:
                bindings.restore()
            else:
                torch.npu.reset_peak_memory_stats()
            return result
        except Exception as error:
            bindings.restore()
            self._resident_session().graphs_dirty = True
            return self._resident_error(error)

    def apply(self, generation):
        # A same-source mode change retains the existing graph/binding. A new
        # source must retire this instance binding before installing its own.
        session = self._resident_session()
        if session.pending is not None and session.current is not None:
            control = session.pending[0]
            if control.digest != session.current.digest or control.recapture:
                bindings.restore()
        return apply_base(self, generation)

    def status(self):
        row = status_base(self)
        row["native_final_mhc_post"] = {
            "layer": state["layer"],
            "calls_by_rows": dict(state["calls_by_rows"]),
            "actual_gate": state["actual_gate"],
            "scope": "unrounded FP32 final mixer only; intermediate mixers and decode unchanged",
        }
        return row

    result = dict(changes)
    for name, function in (("resident_capture", capture), ("resident_apply", apply), ("resident_status", status)):
        result[WORKER_PREFIX + name] = function
    return guard_replacements(result)
