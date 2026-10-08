# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["group", "kda", "cap"])
parser.add_argument("label")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--non-safe-only", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
out = root / (args.mode + "-" + args.label + ".json")
torch.set_num_threads(4)
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
enable_custom_op()
records = []


def record(name, fn, repeats=5):
    first = fn()
    values = first if isinstance(first, tuple) else (first,)
    digests = []
    for x in values:
        cpu = x.detach().cpu().contiguous()
        assert torch.isfinite(cpu).all(), name
        digests.append(hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
    fn()
    torch.npu.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    second = result if isinstance(result, tuple) else (result,)
    for x, digest in zip(second, digests):
        assert (
            hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == digest
        ), name + " nondeterministic"
    row = {"case": name, "sha256": digests, "median_ms": statistics.median(times), "samples_ms": times}
    baseline = root / (args.mode + ("-baseline_non_safe.json" if args.non_safe_only else "-baseline.json"))
    if not args.label.startswith("baseline") and baseline.exists():
        ref = next(r for r in json.loads(baseline.read_text()) if r["case"] == name)
        row["exact_baseline"] = ref["sha256"] == digests
        row["baseline_ms"] = ref["median_ms"]
    records.append(row)
    out.write_text(json.dumps(records, indent=2))
    print(json.dumps(row), flush=True)
    if args.limit and len(records) >= args.limit:
        raise SystemExit(0)


if args.mode == "group":
    from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz, _pack_codes_nz_w3

    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    cases = []
    for bits in (2, 3, 4):
        for layout in ("canonical", "nz"):
            cases.append((bits, layout, 256, 256, [0, 1, 127, 128, 129, 17, 0, 238], 640, "mixed"))
            cases.append((bits, layout, 256, 256, [1, 0, 2, 1, 3, 0, 1, 0], 16, "decode"))
    for bits, k in ((3, 4096), (3, 2048), (4, 4096), (2, 2048)):
        for routing in ("uniform", "hot"):
            counts = [18] * 56 + [17] * 16 if routing == "uniform" else [640] * 2 + [0] * 70
            cases.append((bits, "nz", 4096, k, counts, 5120, routing))
    for bits, layout, n, k, counts, rows, routing in cases:
        gen = torch.Generator().manual_seed(310 + bits + n + k + rows)
        experts = len(counts)
        codes = torch.randint(0, 256, (experts, n, k * bits // 8), dtype=torch.uint8, generator=gen)
        if layout == "nz":
            pack = _pack_codes_nz_w3 if bits == 3 else _pack_codes_nz
            codes = torch.stack([pack(codes[e], k) for e in range(experts)]).view(torch.int8)
        packed = codes.npu()
        scales = (torch.rand(experts, n // 32, k // 32, generator=gen) * 0.02 + 0.005).npu()
        x = (torch.randn(rows, k, generator=gen) * 0.1).half().npu()
        ends = torch.tensor(counts, dtype=torch.int64).cumsum(0).npu()
        record(
            f"w{bits}_{layout}_{n}_{k}_{routing}",
            lambda x=x, packed=packed, scales=scales, ends=ends: op(x, packed, scales, ends),
            3,
        )
        del packed, codes, scales, x, ends
        torch.npu.empty_cache()
elif args.mode == "cap":
    from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz, _pack_codes_nz_w3

    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    for bits in (2, 3, 4):
        gen = torch.Generator().manual_seed(890 + bits)
        k = n = 256
        experts = 8
        codes = torch.randint(0, 256, (experts, n, k * bits // 8), dtype=torch.uint8, generator=gen)
        pack = _pack_codes_nz_w3 if bits == 3 else _pack_codes_nz
        packed = torch.stack([pack(codes[e], k) for e in range(experts)]).view(torch.int8).npu()
        scales = (torch.rand(experts, n // 32, k // 32, generator=gen) * 0.02 + 0.005).npu()
        for rows in (10240, 20480):
            x = (torch.randn(rows, k, generator=gen) * 0.1).half().npu()
            ends_cpu = torch.arange(1, experts + 1) * (rows // (experts * 4))
            ends = ends_cpu.npu()
            expected = torch.cat(
                [
                    op(
                        x[start : start + 5120],
                        packed,
                        scales,
                        (ends_cpu - start).clamp(0, min(5120, rows - start)).npu(),
                    )
                    for start in range(0, rows, 5120)
                ]
            )
            actual = op(x, packed, scales, ends)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            record(
                f"w{bits}_routes{rows}",
                lambda x=x, packed=packed, scales=scales, ends=ends: op(x, packed, scales, ends),
                3,
            )
else:
    for length, varlen, safe, initial in (
        (64, False, True, False),
        (65, True, True, True),
        (134, True, True, True),
        (640, True, True, True),
        (64, True, False, True),
    ):
        if args.non_safe_only and safe:
            continue
        gen = torch.Generator().manual_seed(641 + length)
        heads, width = 16, 128
        shape = (1, length, heads, width)
        q = (torch.randn(shape, generator=gen) * 0.04).half().npu()
        k = (torch.randn(shape, generator=gen) * 0.04).half().npu()
        v = (torch.randn(shape, generator=gen) * 0.04).half().npu()
        raw = (-7 + torch.randn(shape, generator=gen) * 0.03).float().npu()
        if not safe:
            raw = (-torch.rand(shape, generator=gen) * 0.005).float().npu()
        beta = (torch.rand(1, length, heads, generator=gen) * 0.2 + 0.05).float().npu()
        a_log = (torch.randn(heads, generator=gen) * 0.1).npu()
        bias = (torch.randn(heads * width, generator=gen) * 0.1).npu()
        cu = [0, 0, 64, length] if length == 134 else [0, length]
        indices = [[seq, chunk] for seq, (a, b) in enumerate(zip(cu, cu[1:])) for chunk in range((b - a + 63) // 64)]
        state = (
            (torch.randn(len(cu) - 1 if varlen else 1, heads, width, width, generator=gen) * 0.01).npu()
            if initial
            else None
        )
        kwargs = dict(
            layout="BSND", safe_gate=safe, use_gate_in_kernel=safe, lower_bound=-5.0, A_log=a_log, dt_bias=bias
        )
        if varlen:
            kwargs["cu_seqlens"] = cu
        record(
            f"gate_{length}_{varlen}_{safe}_{initial}",
            lambda raw=raw, kwargs=kwargs: torch.ops._C_ascend.kda_gate_cumsum(raw, 64, **kwargs),
        )
        if varlen:
            kwargs["chunk_indices"] = [value for pair in indices for value in pair]

        def full(q=q, k=k, v=v, raw=raw, beta=beta, state=state, kwargs=kwargs):
            outputs = torch.ops._C_ascend.chunk_kda_fwd(
                q,
                k,
                v,
                raw,
                beta,
                128**-0.5,
                64,
                initial_state=state,
                output_final_state=True,
                disable_recompute=False,
                return_intermediate_states=False,
                state_v_first=True,
                **kwargs,
            )
            return outputs[0], outputs[1]

        record(f"full_{length}_{varlen}_{safe}_{initial}", full)
