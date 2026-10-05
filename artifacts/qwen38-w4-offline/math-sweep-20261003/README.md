# Qwen3.8 W4 math sweeps on 310P, 2026-10-03

These are operator diagnostics on `matteius-threadripper` (device 0, Ascend
310P3). The TP4 GLM service on port 8001 remained loaded; sampled AICore
utilization was 0% before and after the probes. No serving process was
restarted. Qwen was **not** serving. Device 0
reported 42,860 MB used of 47,254 MB before and after the probes, with the
resident GLM worker accounting for 41,474 MB. The remote run completed by
02:40:32 UTC. No candidate compiler build was used.

The Python runtime was
`/srv/ai/src/qwen38-head-unified-runtime-20261001` with torch
`2.13.0+cpu`, torch-npu `2.13.0.rc1`, and the coherent r2 Qwen custom OPP.
The host API library SHA-256 was
`7956e585b81ee6f7c14dfb396670c6f240499547b789c971fa92e0b537edb016`.
The existing one-expert benchmark script SHA-256 was
`4b8746d64f44b76748790d5d0bed74fb2d584aeeb89c110b0fbad1ba1a4c41ac`.
The route sweep script in this directory SHA-256 was
`abc9bea8cf5f63cfcdfab384c9be72911302bb3c163fcf5da1181f5f7079b3c3`.

## Method

Both sweeps used five trials of 12 NPU graph replays, with four independent
calls captured in each graph. Tables show median NPU event time per call in
microseconds. These are amortized graph times; the four calls can overlap, so
the difference between complete and prepared cases is **not** the isolated
packing latency. Projection cases passed eager and captured output parity
checks; the route suite also checked changing-input and peer-zero graph replay.

- [`single-expert.jsonl`](single-expert.jsonl) uses the existing
  `tools/qwen4exp/profile_projection_dtypes_310.py`. It holds arithmetic
  values equal across FP16, packed W4A16, and native W4A8, with uniform
  weight scale and zero offset. Shapes are gate/up `K=2560,N=1280` and down
  `K=640,N=2560`, at 1, 3, 32, and 128 rows. `prepared` excludes activation
  packing; `full` includes it.
- [`routes.jsonl`](routes.jsonl) uses [`sweep_routes.py`](sweep_routes.py).
  It has 32 synthetic local experts, varied per-G128 scales and offsets, and
  8 route slots per token. Patterns hold route count fixed while changing
  expert locality: `single`, `four`, `distinct`, and `tp4_sparse` (one local
  route per four slots). It checks exact prepared-versus-complete output and
  peer-zero output after changing input and route IDs in a captured graph.
  SwiGLU packing is measured separately on synthetic gate/up output.

The target checkpoint config has **10** route slots per token and 512 global
experts. With TP4/EP4, the expected local expert bank is 128. Thus the route
suite's 3- and 12-token cases have 24 and 96 slots, rather than the target
decode shapes of 30 and 120. The expert bank and route distribution are also
synthetic; the timings cannot be read as production C1 or C4 latency.

## Results

| One expert | Rows | Pack only | Prepared native W4A8 | Full native W4A8 | Packed W4A16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate/up | 3 | 7.7 | 47.4 | 49.7 | 115.4 |
| Gate/up | 128 | 64.0 | 397.6 | 457.4 | 480.9 |
| Down | 3 | 7.8 | 19.1 | 20.9 | 72.3 |
| Down | 128 | 20.3 | 317.2 | 331.4 | 259.2 |

The 128-row down case is a warning against treating native INT4 as uniformly
faster, even in this matched arithmetic fixture. W4A16 and W4A8 have different
activation policies on real model values, so this comparison does not qualify
a backend change.

| Route-aware case | Local routes / distinct experts | Gate/up prepared | Down prepared | Gate/up pack only | SwiGLU pack only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 tokens, TP4 sparse | 6 / 6 | 84.5 | 50.5 | 7.7 | 12.9 |
| 3 tokens, four experts | 24 / 4 | 144.9 | 85.2 | 7.7 | 12.9 |
| 3 tokens, distinct | 24 / 24 | 331.8 | 182.7 | 7.7 | 12.9 |
| 12 tokens, one expert | 96 / 1 | 369.5 | 282.7 | 11.6 | 34.2 |
| 12 tokens, distinct | 96 / 32 | 1005.5 | 454.8 | 11.6 | 34.2 |

At equal 24 local routes, four experts versus 24 distinct experts changes
gate/up from 144.9 to 331.8 microseconds. At equal 96 local routes, one
expert versus 32 distinct experts changes it from 369.5 to 1005.5
microseconds. The one-expert and four-expert order is not monotonic at all
shapes, so route count alone does not predict runtime. For 3-token TP4-sparse
gate/up, the prepared projection's five event trials span just 0.5
microseconds, whereas the 12-token sparse trials span 8.7 microseconds.

## Interpretation and limits

The present native W4A8 path already avoids FP16 weight unpack. These results
make its **expert and packed-weight schedule, tile reuse, and per-group
correction** more useful next inspection points than another broad GM-copy
reduction. Activation packing is measurable, especially the 34.2-microsecond
SwiGLU pack at 96 route rows, but the prepared projection is larger in every
route-aware case above. The older GLM PyTorch `uint8` bitwise sweep does not
measure any of these Ascend C kernels.

These synthetic fixtures omit the router, shared expert, QSA, GDN, TP
collectives, real checkpoint distribution, and full-model MTP scheduling.
The GLM server occupied all four devices, and its load may affect timings.
Neither table is a Qwen tokens/s result or a controlled candidate-kernel A/B.
The next useful experiment is an isolated kernel variant or instruction trace
at 30, 60, and 120 routes, holding route IDs and G128 metadata fixed, followed by
all-byte/offset parity, changing-route graph replay, and real-weight serving
gates before selecting a change.

## Reproduction

Source the host's Qwen hardware environment, preserve its `PYTHONPATH`, then
put the unified Qwen runtime, coherent r2 OPP, embedded vendor, and retained
fallback in the same order as
`examples/start_qwen38_flash_next_w4_310p.sh`. Set
`ASCEND_RT_VISIBLE_DEVICES=0` and `OMP_NUM_THREADS=1`; run from the unified
runtime directory. The measurement commands were:

```bash
python -m tools.qwen4exp.profile_projection_dtypes_310 \
  --output /tmp/qwen38-math-20261003-r2.jsonl \
  --rows 1 3 32 128 --projections gate_up down \
  --cases fp16_preformatted w4a16_packed_grouped \
  w4a8_native_fused_prepared w4a8_native_fused int4_fused_pack_only \
  --iterations 12 --repeats 5 --graph-unroll 4

python /tmp/qwen38-sweep_routes-20261003.py \
  --output /tmp/qwen38-routes-20261003-r1.jsonl \
  --tokens 1 3 6 12 --iterations 12 --repeats 5 --graph-unroll 4
```

The second script is a copy of `sweep_routes.py` in this directory. The first
attempt at the one-expert run failed before measurement because `PYTHONPATH`
replaced CANN's environment and an operator-build import failed. The completed run
preserved the hardware environment's `PYTHONPATH` entries.
