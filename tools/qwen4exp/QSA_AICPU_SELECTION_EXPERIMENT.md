# Qwen3.8 QSA AI CPU selection experiment

This is an isolated candidate for the cold-prefill QSA selector. It does not
change serving dispatch or the qualified launcher. The isolated 310P gate ran
on 2026-10-04; further NPU work is deferred at the user's request.

## Question and comparison

At 2,048 queries and 10,000 visible groups, the measured index stage was
61.476 ms per QSA layer: 24.934 ms scoring and 37.682 ms selection. QSA
attention was about 246 ms per layer. The AI CPU can read the FP32 score tensor
from device memory without sending it over PCIe, but a scalar selector may be
slower than the current AI Core path. Measure the entire selector call, including
launch and synchronization, before considering a server change.

The candidate consumes the already-scored and masked contiguous FP32
`[queries, groups]` tensor and returns INT32 `[queries, topk]` group IDs.
It uses `nth_element` followed by sorting the selected K values, comparing by
score descending and group ID ascending. This implements the exact stable
descending order, including ties across the K boundary and `-inf` padding.
NaNs are rejected. Scores produced by the QSA ReLU/mask path should not have
NaNs; an actual NaN must be investigated instead of silently assigned a rank.

The existing prefill policy uses `_fast_topk_indices`. It sorts the returned K
indices exactly but a tie at the cutoff can change *which* K are returned.
The tested synthetic cases happened to match stable CPU argsort on both paths;
that does not establish exactness of the current fast policy for all ties.

## Host gate (no NPU)

Run from the repository root:

```bash
python3 -m pytest --noconftest -q tests/ut/qwen38_1m/test_qsa_exact_topk_aicpu.py
python3 tools/qwen4exp/benchmark_qsa_exact_topk_aicpu_310.py --dry-run
g++ -O3 -std=c++17 -pthread tools/qwen4exp/qsa_exact_topk_host.cpp -o /tmp/qsa_exact_topk_host
/tmp/qsa_exact_topk_host 2048 10000 512 5 1 ties
/tmp/qsa_exact_topk_host 2048 10000 512 5 4 random
/tmp/qsa_exact_topk_host 2048 10000 512 5 8 random
```

The C++ host timings characterize the algorithm only. This host is a Ryzen 9
7950X, **not** the target Threadripper. The four-thread 2,048×10,000×512
trial took roughly 47–49 ms for random/tied scores, above the earlier
37.682 ms device selection measurement. A tiny four-thread launch took about
0.06 ms, so thread startup does not explain the gap. An eight-thread trial
took roughly 24–25 ms; the AI CPU candidate now presents eight row shards to
`ParallelFor`, which can use only the workers available on the device. None
of these Ryzen timings predicts the 310P AI CPU result. In particular, a 310P
with fewer available AI CPU workers may retain the slower four-way behavior.

## Isolated 310P gate

Use an isolated runtime and unique OPP vendor. The operator is in
`csrc/attention/qsa_exact_topk_aicpu_v310/`; the PyTorch symbol is
`torch.ops._C_ascend.npu_qsa_exact_topk_aicpu_310`. Build the custom OPP with
the matching CANN 9.1 toolchain and rebuild the extension from the same source
snapshot. One example from `csrc/` is:

```bash
bash build.sh --pkg --soc=ascend310p \
  --ops=qsa_exact_topk_aicpu_v310 --vendor_name=qsa_exact_topk_probe -j8 -O3
```

Keep the qualified serving OPP and launcher untouched. Confirm that the
experimental host API and AI CPU kernel come from the same package, then run
the standalone gate on one idle 310P:

```bash
python3 tools/qwen4exp/benchmark_qsa_exact_topk_aicpu_310.py \
  --output /path/to/new/qsa-aicpu-selection.jsonl
```

The gate covered decode-like 3×5,856 and prefill 64/256/2,048×10,000 scores,
with random, cutoff-tie, and masked tails. The isolated custom vendor and
matching extension ran on 310P device 0. The candidate matched stable CPU
argsort in all 12 cases, and device logs confirmed AI CPU kernel execution.
Each median below is per call, from three trials of two calls followed by a
device synchronization. Scores were precomputed on device; this is an isolated
selector comparison, not full-model TTFT.

| Queries × groups | Pattern | AI CPU exact (ms) | Current fast topk (ms) | AI CPU / current |
| --- | --- | ---: | ---: | ---: |
| 3 × 5,856 | random / ties / masked | 0.552 / 0.638 / 0.429 | 0.322 / 0.320 / 0.310 | 1.72 / 2.00 / 1.38 |
| 64 × 10,000 | random / ties / masked | 3.770 / 3.716 / 2.765 | 1.182 / 1.301 / 1.266 | 3.19 / 2.86 / 2.18 |
| 256 × 10,000 | random / ties / masked | 14.995 / 15.314 / 10.985 | 3.660 / 4.059 / 3.855 | 4.10 / 3.77 / 2.85 |
| 2,048 × 10,000 | random / ties / masked | 114.616 / 122.922 / 86.208 | 26.660 / 29.222 / 27.953 | 4.30 / 4.21 / 3.08 |

The raw trial times and parity flags are in
[`selector_gate.jsonl`](../../artifacts/qwen38-qsa-aicpu-20261004/selector_gate.jsonl).
This exact AI CPU selector is slower at every measured shape; do not integrate
it into serving. These results do not rule out AI CPU work on smaller,
branch-heavy device metadata or a different selection algorithm. No model
server or full-prompt TTFT test was run for this candidate.

Source changes are in the shared main checkout. A temporary build snapshot on
the Threadripper at `/srv/ai/src/qsa-aicpu-probe-20261003` produced the tested
AI CPU-only OPP package at
`csrc/build_out/cann-ops-transformer-qsa_exact_topk_probe_linux-x86_64.run`.
Its tested SHA256 was
`5c03899300041ee4b62666a7fcf0b5b191e96507893b3d7129ad4db784e34587`.
The build system skips AI Core metadata generation for AI CPU-only packages and
now relinks the AI CPU shared library when its kernel object changes. The
matching PyTorch extension was built and staged with the custom vendor.

After the device run, shape recovery for flattened AI CPU descriptors was
extracted into the shared host-tested helper. The local regression gate now
passes 14 tests. This refactor has not been rebuilt or retested on the NPU;
the raw device measurements correspond to the earlier equivalent kernel.
