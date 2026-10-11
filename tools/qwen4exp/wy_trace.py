# SPDX-License-Identifier: Apache-2.0
"""Explicit diagnostic snapshots and FP64 WY oracle; never used in a hot path."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CHUNK_SIZE = 64
PREPARED_NAMES = ("q_kernel", "k_kernel", "w", "u", "cumulative_g")
INPUT_NAMES = ("q", "k", "v", "g", "beta")


def wy_fp64(q, k, v, g, beta, chunk_size=CHUNK_SIZE):
    """Independent triangular forward solve with value-head expansion in FP64.

    Inputs must already carry the selected Q/K normalization. No re-normalizing
    or gate re-rounding occurs. Outputs intentionally retain FP64 for diagnosis.
    """
    if q.shape != k.shape or k.ndim != 4 or v.ndim != 4:
        raise ValueError("WY oracle requires BTHD Q/K/V")
    batch, tokens, kh, dim = k.shape
    vh = v.shape[2]
    if tokens % chunk_size or vh % kh or g.shape != (batch, tokens, vh) or beta.shape != g.shape:
        raise ValueError("invalid WY chunk/head/gate geometry")
    key = k.double().repeat_interleave(vh // kh, 2).transpose(1, 2).contiguous()
    value = v.double().transpose(1, 2).contiguous()
    gate = g.double().transpose(1, 2).reshape(batch, vh, tokens // chunk_size, chunk_size).cumsum(-1)
    beta_head = beta.double().transpose(1, 2).contiguous()
    output_w, output_u = torch.empty_like(key), torch.empty_like(value)
    for chunk in range(tokens // chunk_size):
        first = chunk * chunk_size
        keys = key[:, :, first : first + chunk_size]
        values = value[:, :, first : first + chunk_size]
        gates = gate[:, :, chunk]
        betas = beta_head[:, :, first : first + chunk_size]
        gram = keys @ keys.transpose(-1, -2)
        w_rows, u_rows = [], []
        for row in range(chunk_size):
            u = values[:, :, row] * betas[:, :, row, None]
            w = keys[:, :, row] * (betas[:, :, row] * gates[:, :, row].exp()).unsqueeze(-1)
            for prior in range(row):
                coefficient = -betas[:, :, row] * gram[:, :, row, prior] * (gates[:, :, row] - gates[:, :, prior]).exp()
                u = u + coefficient.unsqueeze(-1) * u_rows[prior]
                w = w + coefficient.unsqueeze(-1) * w_rows[prior]
            w_rows.append(w)
            u_rows.append(u)
        output_w[:, :, first : first + chunk_size] = torch.stack(w_rows, 2)
        output_u[:, :, first : first + chunk_size] = torch.stack(u_rows, 2)
    return (
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        output_w,
        output_u,
        gate.reshape(batch, vh, tokens),
    )


def comparison(reference, candidate, *, atol, rtol):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return dict(passed=False, shape_or_dtype_mismatch=True)
    a, b = reference.detach().cpu().double(), candidate.detach().cpu().double()
    delta = (a - b).abs()
    bad = ~torch.isfinite(a) | ~torch.isfinite(b) | (delta > atol + rtol * a.abs())
    indices = bad.nonzero()
    return dict(
        passed=not bool(bad.any()),
        mismatches=int(bad.sum()),
        elements=a.numel(),
        max_abs_error=(float(delta.max()) if torch.isfinite(delta).all() else None) if delta.numel() else 0.0,
        first_mismatch=indices[0].tolist() if len(indices) else None,
    )


def capture_wy_trace(
    directory,
    inputs,
    initial_state,
    *,
    reference_prepare,
    candidate_prepare,
    downstream,
    qk_normalized,
    atol=0.003,
    rtol=0.02,
):
    """Save exact inputs plus both outputs/states even if the comparison fails.

    This function explicitly copies diagnostic tensors to CPU. Invoke only in
    an isolated diagnostic job, never in serving execution. Each variant gets
    cloned inputs/state so mutations cannot contaminate the other variant.
    ``downstream(prepared, initial_state)`` must return output and final state.
    """
    if len(inputs) != len(INPUT_NAMES) or type(qk_normalized) is not bool:
        raise ValueError("five exact input tensors and normalization identity required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    saved = dict(zip(INPUT_NAMES, [t.detach().clone() for t in inputs]))
    saved["initial_state"] = initial_state.detach().clone()
    result = dict(
        qk_normalized=qk_normalized,
        chunk_size=CHUNK_SIZE,
        atol=atol,
        rtol=rtol,
        hardware_validated=False,
        errors={},
        comparisons={},
        input_mutation={},
    )
    prepared_variants = {}
    for name, prepare in (("reference", reference_prepare), ("candidate", candidate_prepare)):
        local_inputs = tuple(saved[k].clone() for k in INPUT_NAMES)
        local_state = saved["initial_state"].clone()
        try:
            prepared = tuple(prepare(*local_inputs, CHUNK_SIZE))
            if len(prepared) != len(PREPARED_NAMES):
                raise ValueError("WY preparation must return all five stages")
            prepared_variants[name] = tuple(t.detach().clone() for t in prepared)
            for stage, tensor in zip(PREPARED_NAMES, prepared_variants[name]):
                saved[f"{name}_{stage}"] = tensor
            output, state = downstream(prepared, local_state)
            saved[f"{name}_output"] = output.detach().clone()
            saved[f"{name}_final_state"] = state.detach().clone()
        except Exception as error:
            result["errors"][name] = f"{type(error).__name__}: {error}"
        result["input_mutation"][name] = any(not torch.equal(t, saved[k]) for t, k in zip(local_inputs, INPUT_NAMES))
    for stage in (*PREPARED_NAMES, "output", "final_state"):
        names = [f"{variant}_{stage}" for variant in ("reference", "candidate")]
        if all(k in saved for k in names):
            result["comparisons"][stage] = comparison(*(saved[k] for k in names), atol=atol, rtol=rtol)
    oracle = wy_fp64(*(saved[k] for k in INPUT_NAMES))
    for stage, tensor in zip(PREPARED_NAMES, oracle):
        saved[f"fp64_{stage}"] = tensor
    result["first_divergence"] = next(
        (stage for stage, metric in result["comparisons"].items() if not metric["passed"]), None
    )
    result["passed"] = (
        not result["errors"]
        and not any(result["input_mutation"].values())
        and len(result["comparisons"]) == len(PREPARED_NAMES) + 2
        and all(metric["passed"] for metric in result["comparisons"].values())
    )
    path = directory / "tensors.npz"
    np.savez_compressed(path, **{name: value.detach().cpu().numpy() for name, value in saved.items()})
    result["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["tensors"] = {name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in saved.items()}
    (directory / "trace.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def wy_downstream_fp64(prepared, initial_state, chunk_size=CHUNK_SIZE):
    """Independent WY chunk output/state equations, value-major recurrent state.

    Q is used exactly as provided; callers apply their chosen query scaling
    before preparation. This oracle does not model native FP16 boundaries.
    """
    q, k, w, u, cumulative = (t.double() for t in prepared)
    batch, kh, tokens, kd = k.shape
    vh = u.shape[1]
    if tokens % chunk_size or vh % kh or initial_state.shape != (batch, vh, u.shape[-1], kd):
        raise ValueError("invalid downstream WY state/head/chunk shape")
    q, k = (t.repeat_interleave(vh // kh, 1) for t in (q, k))
    state = initial_state.double().clone()
    output = torch.empty_like(u)
    for first in range(0, tokens, chunk_size):
        section = slice(first, first + chunk_size)
        queries, keys, gates = q[:, :, section], k[:, :, section], cumulative[:, :, section]
        values = u[:, :, section] - w[:, :, section] @ state.transpose(-1, -2)
        decay = (gates.unsqueeze(-1) - gates.unsqueeze(-2)).exp().tril()
        output[:, :, section] = (queries @ state.transpose(-1, -2)) * gates.exp().unsqueeze(-1) + (
            (queries @ keys.transpose(-1, -2)) * decay
        ) @ values
        last = gates[:, :, -1]
        state = (
            state * last.exp()[..., None, None]
            + (values * (last.unsqueeze(-1) - gates).exp().unsqueeze(-1)).transpose(-1, -2) @ keys
        )
    return output, state
