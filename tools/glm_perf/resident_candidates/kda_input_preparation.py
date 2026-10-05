# SPDX-License-Identifier: Apache-2.0
"""Opt-in KDA input experiments; preserve the current recurrent state logic.

The resident harness replaces only _run_recurrent. Its five input assignments
are replaced in a private function, after checking their exact syntax. No live
module is edited here; the harness owns pause, recapture, and rollback.
"""

import ast
import inspect
import textwrap

import torch


def normalize_qk(q, k, normalize):
    if q.shape == k.shape and q.ndim == 4 and q.shape[0] == 1 and q.dtype == k.dtype and q.device == k.device:
        packed = normalize(torch.cat((q, k), dim=0))
        return packed[:1], packed[1:]
    return normalize(q), normalize(k)


def gate_beta_reference(raw_gate, beta_raw, scale, bias, lower_bound):
    return (
        (lower_bound * torch.sigmoid(scale * (raw_gate.float() + bias))).squeeze(0).contiguous(),
        beta_raw.float().sigmoid().squeeze(0).to(torch.float16).contiguous(),
    )


def make_preparer(normalize, safe_gate, *, batch_qk=False, gate_beta=None):
    def prepare(layer, q, k, v, raw_gate, beta_raw):
        if batch_qk:
            q, k = normalize_qk(q, k, normalize)
        else:
            q, k = normalize(q), normalize(k)
        cached = getattr(layer, "_kda_gate_weights", None)
        if (
            gate_beta is not None
            and cached is not None
            and cached[0] is layer.A_log
            and cached[1] is layer.dt_bias
            and gate_beta.supports(raw_gate, beta_raw, cached[2], cached[3], float(layer.kda_lower_bound))
        ):
            gk, beta = gate_beta(raw_gate, beta_raw, cached[2], cached[3], float(layer.kda_lower_bound))
        else:
            gk = safe_gate(layer, raw_gate).squeeze(0).contiguous()
            beta = beta_raw.float().sigmoid().squeeze(0).to(torch.float16).contiguous()
        return (
            q.squeeze(0).to(torch.float16).contiguous(),
            k.squeeze(0).to(torch.float16).contiguous(),
            v.squeeze(0).to(torch.float16).contiguous(),
            gk,
            beta,
        )

    return prepare


def replace_input_prelude(original, prepare):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    offset = int(isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant))
    expected = ast.parse(
        "q = _l2norm_310p(q).squeeze(0).to(torch.float16).contiguous()\n"
        "k = _l2norm_310p(k).squeeze(0).to(torch.float16).contiguous()\n"
        "v = v.squeeze(0).to(torch.float16).contiguous()\n"
        "gk = _safe_gate_for_layer(self_attn, raw_gate).squeeze(0).contiguous()\n"
        "beta = beta_raw.float().sigmoid().squeeze(0).to(torch.float16).contiguous()\n"
    ).body
    if [ast.dump(node) for node in function.body[offset : offset + 5]] != [ast.dump(node) for node in expected]:
        raise ValueError("KDA preparation changed: re-audit the candidate before applying")
    function.body[offset : offset + 5] = ast.parse(
        "q, k, v, gk, beta = _candidate_prepare(self_attn, q, k, v, raw_gate, beta_raw)"
    ).body
    function.decorator_list = []
    scope = dict(original.__globals__, _candidate_prepare=prepare)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(original), "exec"), scope)
    return scope[original.__name__]


def replacements(native_resources=None):
    # Worker-only lazy import: CPU helpers above do not import vLLM or torch_npu.
    from vllm_ascend.models.glm5next_w2 import kda_310

    prepare = make_preparer(kda_310._l2norm_310p, kda_310._safe_gate_for_layer, batch_qk=True)
    return {
        "vllm_ascend.models.glm5next_w2.kda_310:_run_recurrent": replace_input_prelude(kda_310._run_recurrent, prepare)
    }


def native_gate_replacements(native_resources):
    from vllm_ascend.models.glm5next_w2 import kda_310

    prepare = make_preparer(
        kda_310._l2norm_310p, kda_310._safe_gate_for_layer, gate_beta=native_resources["kda_gate_beta_v1"]
    )
    return {
        "vllm_ascend.models.glm5next_w2.kda_310:_run_recurrent": replace_input_prelude(kda_310._run_recurrent, prepare)
    }
