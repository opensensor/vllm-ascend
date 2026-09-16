#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D1 micro-benchmark for the Qwen4Exp host PLE table transports (plan T4.1).

Compares the two host PLE table transports on a small synthetic table so the
transport switch can be decided on real Ascend 310P hardware at D1:

* transport (b) shared-mmap  -- always runs (host-testable);
* transport (a) pinned-UVA   -- runs only where UVA / pinned memory is available
  (the on-device gather is D1-verified); reported as unavailable otherwise.

The benchmark builds a synthetic table, times a batched random row gather across
several iterations, and reports rows/s and effective GiB/s. It never allocates
the real 95.43 GiB table -- keep ``--rows`` / ``--dim`` small on host.

Examples::

    python3 tools/qwen38_1m/ple_transport_bench.py --rows 65536 --dim 256
    python3 tools/qwen38_1m/ple_transport_bench.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import torch

# Allow running as a plain script (python tools/qwen38_1m/ple_transport_bench.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vllm_ascend.models.qwen4_exp.dtype_policy import (  # noqa: E402
    ASCEND_QWEN4EXP_DTYPE_POLICY,
)
from vllm_ascend.models.qwen4_exp.ngram_embedding import (  # noqa: E402
    AscendPLEPinnedHostEmbeddingMethod,
    AscendPLESharedMmapEmbeddingMethod,
    PLETransportUnavailableError,
)

_BYTES_PER_GIB = 1024**3


def _synthetic_table(rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
    base = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
    return (base * 0.001 + cols * 0.01).to(dtype)


def _time_gather(method, ids: torch.Tensor, iters: int) -> dict:
    # Warm up (first touch faults pages / builds views).
    method.gather_rows(ids)
    start = time.perf_counter()
    total_rows = 0
    for _ in range(iters):
        rows = method.gather_rows(ids)
        total_rows += rows.shape[0]
    elapsed = time.perf_counter() - start
    row_bytes = method.embedding_dim * method.itemsize
    moved_bytes = total_rows * row_bytes
    return {
        "iters": iters,
        "num_ids": int(ids.numel()),
        "elapsed_s": elapsed,
        "rows_per_s": (total_rows / elapsed) if elapsed > 0 else float("inf"),
        "gib_per_s": (moved_bytes / _BYTES_PER_GIB / elapsed) if elapsed > 0 else float("inf"),
    }


def run(rows: int, dim: int, num_ids: int, iters: int, seed: int) -> dict:
    dtype = ASCEND_QWEN4EXP_DTYPE_POLICY.cast_site("ngram_embedding")
    generator = torch.Generator().manual_seed(seed)
    table = _synthetic_table(rows, dim, dtype)
    ids = torch.randint(0, rows, (num_ids,), generator=generator, dtype=torch.long)

    table_bytes = rows * dim * (torch.finfo(dtype).bits // 8)
    result: dict = {
        "rows": rows,
        "dim": dim,
        "dtype": str(dtype),
        "table_bytes": table_bytes,
        "transports": {},
    }

    # Transport (b): shared mmap (always available on host).
    shm_path = os.path.join("/dev/shm", f"ple_bench_{uuid.uuid4().hex}.bin")
    mmap_method = AscendPLESharedMmapEmbeddingMethod(
        rows,
        dim,
        shm_path=shm_path,
        create=True,
        table_source=table,
        # Bench uses a tiny synthetic table; skip the production host reserve.
        reserve_bytes=0,
    )
    try:
        result["transports"]["shared_mmap"] = {
            "available": True,
            **_time_gather(mmap_method, ids, iters),
        }
    finally:
        mmap_method.close()
        if os.path.exists(shm_path):
            os.unlink(shm_path)

    # Transport (a): pinned-UVA (device path; runs only where available).
    if AscendPLEPinnedHostEmbeddingMethod.is_available():
        try:
            pinned_method = AscendPLEPinnedHostEmbeddingMethod(
                rows,
                dim,
                table_source=table,
                reserve_bytes=0,
            )
            try:
                result["transports"]["pinned_uva"] = {
                    "available": True,
                    **_time_gather(pinned_method, ids, iters),
                }
            finally:
                pinned_method.close()
        except PLETransportUnavailableError as exc:
            result["transports"]["pinned_uva"] = {"available": False, "reason": str(exc)}
    else:
        result["transports"]["pinned_uva"] = {
            "available": False,
            "reason": "UVA unavailable on this host; on-device path is D1-verified",
        }

    return result


def _human(result: dict) -> str:
    lines = [
        f"PLE transport micro-bench: {result['rows']} rows x {result['dim']} dim "
        f"({result['dtype']}), table={result['table_bytes'] / _BYTES_PER_GIB:.4f} GiB",
    ]
    for name, stats in result["transports"].items():
        if not stats.get("available"):
            lines.append(f"  {name:<12} unavailable ({stats.get('reason', 'n/a')})")
            continue
        lines.append(
            f"  {name:<12} {stats['rows_per_s'] / 1e6:8.3f} M rows/s  "
            f"{stats['gib_per_s']:7.3f} GiB/s  "
            f"({stats['num_ids']} ids x {stats['iters']} iters in {stats['elapsed_s'] * 1e3:.2f} ms)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1 << 16, help="synthetic table rows")
    parser.add_argument("--dim", type=int, default=256, help="embedding dim")
    parser.add_argument("--num-ids", type=int, default=4096, help="row ids per gather")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    result = run(args.rows, args.dim, args.num_ids, args.iters, args.seed)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_human(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
