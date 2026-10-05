# SPDX-License-Identifier: Apache-2.0
"""One diagnostic pass while the resident harness has serving paused."""

import torch

from tools.glm_perf.kpool_prefill import score_prefill_paged, select_prefill_request
from tools.glm_perf.resident_worker import ResidentWorkerExtension
from vllm_ascend.models.glm5next import kpool_ops


def audit(op):
    # Reproduce the failed 640-row test without any model execution or reload.
    generator = torch.Generator().manual_seed(310)
    rows, pools, br, rs, offset = 640, 2048, 160, 144, 16
    columns = (pools + br - 1) // br
    blocks, bs = columns + 1, br * rs + 256
    storage = torch.full((offset + blocks * bs,), torch.nan, dtype=torch.float16)
    cache_cpu = storage.as_strided((blocks, br, 128), (bs, rs, 1), offset)
    cache_cpu[:-1].copy_(torch.randn(cache_cpu[:-1].shape, generator=generator).bfloat16().half())
    table = torch.arange(columns, dtype=torch.int32).flip(0).reshape(1, -1).npu()
    q = torch.randn(rows, 32, 128, generator=generator).bfloat16().half().npu()
    weights = torch.randn(rows, 32, generator=generator).npu()
    positions = (torch.arange(pools - rows, pools, dtype=torch.int32) * 4 - 1).npu()
    storage = storage.npu()
    cache = storage.as_strided((blocks, br, 1, 128), (bs, rs, rs, 1), offset)
    ids = torch.arange(pools, device=q.device)
    keys = cache[table[0, ids // br].long(), ids % br, 0]
    full_rot = kpool_ops.hadamard128(q).bfloat16().half()
    chunk_rot = torch.cat([kpool_ops.hadamard128(q[i : i + 128]).bfloat16().half() for i in range(0, rows, 128)])
    rotation_mismatches = int((full_rot != chunk_rot).sum().item())
    del full_rot, chunk_rot
    baseline = kpool_ops.score_kpool(q, weights, keys)
    native = torch.cat(
        [
            score_prefill_paged(
                q[i : i + 128], weights[i : i + 128], cache, table, positions[i : i + 128], pools, native_op=op
            )
            for i in range(0, rows, 128)
        ]
    )
    baseline.masked_fill_(ids[None, :] >= ((positions + 1) // 4)[:, None], -torch.inf)
    selected_base = baseline.topk(512).indices
    selected_native = native.topk(512).indices
    mismatch_rows = (selected_base.sort().values != selected_native.sort().values).any(1)
    bad_rows = mismatch_rows.nonzero().flatten().cpu().tolist()
    valid = torch.isfinite(native) & torch.isfinite(baseline)
    absolute = (native[valid] - baseline[valid]).abs()
    output = {
        "rotation_mismatches": rotation_mismatches,
        "score_max_abs_error": float(absolute.max().item()),
        "score_mean_abs_error": float(absolute.mean().item()),
        "changed_selection_rows": len(bad_rows),
        "examples": [],
    }
    for row in bad_rows[:3]:
        b, n = baseline[row].cpu(), native[row].cpu()
        b_ids, n_ids = selected_base[row].cpu(), selected_native[row].cpu()
        dropped = b_ids[~torch.isin(b_ids, n_ids)]
        added = n_ids[~torch.isin(n_ids, b_ids)]
        ref = (
            (kpool_ops.hadamard128(q[row : row + 1].cpu()).bfloat16().float()[0] @ keys.cpu().float().T).relu()
            * weights[row, :, None].cpu()
        ).sum(0)
        changed = torch.cat([dropped, added])[:12]
        output["examples"].append(
            {
                "row": row,
                "dropped": dropped.tolist(),
                "added": added.tolist(),
                "ids": changed.tolist(),
                "baseline_scores": b[changed].tolist(),
                "native_scores": n[changed].tolist(),
                "cpu_scores": ref[changed].tolist(),
                "cutoff_gap": float((b.topk(513).values[511] - b.topk(513).values[512]).item()),
            }
        )
    del baseline, native, selected_base, selected_native, absolute, valid

    def old():
        gathered = cache[table[0, ids // br].long(), ids % br, 0]
        return kpool_ops.score_and_select_kpool_tokens(q, weights, gathered, positions, 2048, 4)

    def new():
        return select_prefill_request(q, weights, cache, table, positions, pools, 2048, 4, native_op=op)

    timings = {"baseline": [], "candidate": []}
    for fn in (old, new):
        fn()
    torch.npu.synchronize()
    for repeat in range(5):
        order = [("baseline", old), ("candidate", new)]
        for name, fn in order if repeat % 2 == 0 else reversed(order):
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record()
            result = fn()
            end.record()
            end.synchronize()
            timings[name].append(begin.elapsed_time(end))
            del result
    output["selector_ms"] = timings
    return output


def replacements(native_resources):
    op = native_resources["prefill_v1"]
    result = {}

    def status(self):
        if not result:
            try:
                result.update(audit(op))
            except Exception as exc:
                result["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
        receipt = ResidentWorkerExtension.resident_status(self)
        receipt["prefill_audit"] = result
        return receipt

    return {"vllm_ascend._310p.worker_310p:NPUWorker310.resident_status": status}
