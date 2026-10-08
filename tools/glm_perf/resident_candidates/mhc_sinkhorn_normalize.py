# SPDX-License-Identifier: Apache-2.0
"""Resident decode-only normalization fusion; requires its validated native resource."""

import inspect

DECODE_QUALIFIED_ROWS = (1, 2, 3, 4, 5, 6, 7, 8)
NORMALIZATION_BLOCK = """    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
"""


def wrap_pre(original, normalize, *, qualified_rows=DECODE_QUALIFIED_ROWS):
    if not qualified_rows or any(type(rows) is not int or rows <= 0 for rows in qualified_rows):
        raise ValueError("normalization requires explicit positive qualified row counts")
    original = getattr(original, "__glm_resident_original__", original)
    source = inspect.getsource(original)
    if source.count(NORMALIZATION_BLOCK) != 1:
        raise ValueError("upstream mHC normalization changed; refuse partial replacement")
    # Retain the complete current upstream implementation, signature, and
    # surrounding arithmetic. Change only the verified normalization block.
    replacement = """    if (hc_mult == 4 and num_tokens in _resident_normalize_rows
            and sinkhorn_repeat == 20 and hc_sinkhorn_eps == 1e-6):
        comb_mix = _resident_normalize(torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps)
    else:
""" + "".join("    " + line + "\n" for line in NORMALIZATION_BLOCK.rstrip().splitlines())
    namespace = dict(
        original.__globals__, _resident_normalize=normalize, _resident_normalize_rows=frozenset(qualified_rows)
    )
    exec(compile(source.replace(NORMALIZATION_BLOCK, replacement), "<resident-mhc-normalize>", "exec"), namespace)
    wrapped = namespace[original.__name__]
    wrapped.__glm_resident_original__ = original
    return wrapped


def replacements(native_resources):
    from vllm_ascend.patch.worker import patch_mhc_norm

    operation = native_resources["sinkhorn_normalize_v1"]
    return {
        "vllm_ascend.patch.worker.patch_mhc_norm:_mhc_pre_torch": wrap_pre(patch_mhc_norm._mhc_pre_torch, operation)
    }
