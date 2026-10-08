# GLM 310P graph with NZ-packed expert codes

Status: validated performance candidate, **not** Gate-A quality qualified.
The serving experiment used four Ascend 310P devices, the
`GLM-5.3-Flash-W4through32-noclip-310p` checkpoint, TP4, device-resident
MLA, `FULL_DECODE_ONLY` graph captures `[1,4]`, 512-token prefill chunks,
and the native BF16 mHC rounding change from commit `ca9cf3259`.
`tmp/serve-glm-rowtile-graph-nz-c1-c4-20261003.sh` differs from the earlier
canonical launch primarily by
`"ascend_glm_nz_packed_codes":true` in `--hf-overrides`. It uses the
known-good rowtile OPP, **not** the slower/non-finite singleton-L1 candidate.

## Why the earlier isolated numbers did not predict serving

The canonical serving profiler's grouped-W2 inputs were `UINT8`, for example
`[4096,4096]` activations and `[72,4096,2048]` W4 codes. The previous
isolated candidate benchmark had used NZ-packed `INT8`, a different operator
variant. The loader opt-in for NZ was omitted from the current graph launcher.
This was a launch configuration mismatch, not a graph or kernel regression.

A focused NPU sweep used 72 local experts, one-quarter local routes, and
512–4,096 total routed rows. Five synchronized repetitions per case showed
nearly flat cost as local rows grew, consistent with per-active-expert decode
dominating row math. At 4,096 total routes (1,024 local):

| Projection | Canonical `uint8` | NZ-packed `int8` | Reduction |
| --- | ---: | ---: | ---: |
| W4 gate/up | 239.73 ms | 72.22 ms | 69.9% |
| W2 down | 129.47 ms | 44.52 ms | 65.6% |

All eight canonical/NZ full-size outputs passed bitwise FP16 parity in
separate processes with identical logical codes, scales, inputs, route
boundaries, seed, and source package. The canonical compiled object SHA-256
was `09e04f4b3bf7ae197f255d52edbc2dac8bddda46a4a6bc15c5a726d3f1d5c836`;
the NZ object was
`fd115612355c0dce96983d302806c0422a68a84c9e0a196c5de8464c6118ce10`.
The sweep is synthetic: actual router histograms need their own capture.

## Serving observations

| Check | NZ graph result | Comparison/caveat |
| --- | ---: | --- |
| c1, 24-token stream | 2.13 decode tok/s | Canonical BF16 graph probe: 1.01 tok/s; capture-size set differed (`[1,4]` vs `[1]`). |
| c4, short exact-code requests | Four exact answers; 0.65 s per steady four-token step | About 6.2 aggregate decode tok/s; short sample. |
| c4, four 24-token requests | 96 tokens / 29.28 s = 3.28 end-to-end tok/s | Steady four-token steps were 0.71–0.84 s; three prefills arrived in a later scheduler wave. |
| 517-token prompt, 1 output token | 9.24 s client elapsed | Do not compare directly with the canonical *profiled* 20.31 s call. |
| 2,085-token exact-code retrieval | 52.46 s, exact 11-token answer | Matched canonical BF16 graph: 90.91 s, same prompt/checkpoint, but `[1]` capture rather than `[1,4]`. |
| 8,197-token exact-code retrieval | 204.44 s, exact 11-token answer | Older canonical pre-BF16 graph: 353.39 s; this is a combined-code-change comparison, not NZ-only. |

The strict 20-case graph quality suite completed all requests and remained
**17/20**, with the same `instr_reverse`, `instr_first`, and `code_slice`
failures as canonical and eager. Exact long retrievals and bitwise operator
parity are encouraging, but neither turns this into a Gate-A pass. Do not
promote NZ as production-qualified solely from speed results.

NZ checkpoint load took 261.2–261.7 s across ranks, versus about 197.3 s in
the prior canonical BF16 profile launch. Page cache and run conditions were
not controlled, so the roughly one-minute difference is an observation, not
an isolated repacking cost. The NZ graph captured in 4 s and used 0.28 GiB.

Evidence on the Threadripper host is under
`/home/matteius/experiments/glm-gate-a-20261002/`:

- `w2-grouped-uniform-quarter-canonical-20261003.pt`
- `w2-grouped-uniform-quarter-sweep-20261003.pt` (NZ)
- `w2-grouped-uniform-quarter-canonical-vs-nzpacked-20261003.json`
- `server-rowtile-graph-nz-c1-c4-20261003.log`

Local request records are `/tmp/glm-nz-graph-quality-20261003.jsonl`,
`/tmp/glm-nz-graph-coherence-2k-20261003.json`, and
`/tmp/glm-nz-graph-coherence-8k-20261003.json`.

Next performance target: capture a fresh NZ graph trace and split the
remaining approximately 0.47-s c1 step among grouped W2, kpool selection,
mHC, KDA, and collectives. The current 8K request still spent around
10–13.5 s on many later 512-token chunks, so W2 alone is not the remaining
prefill explanation.

## Dense-kpool graph replay check, 2026-10-03

An isolated graph-replay change skips the eager kpool selector when the
scheduler-owned pooled lengths are strictly below the sparse budget. It
clears the graph buffer to `-1`; QSA's negative-tail sentinel selects the
dense path and ignores those IDs. At the budget boundary and above, it runs
the original selector. The standalone kpool CPU tests passed (29 tests on
the Threadripper source snapshot), including both sides of that boundary.

The live TP4 NZ graph server captured `[1,4]` successfully and returned an
exact short answer, four distinct concurrent exact-code answers, and an
exact 2,437-token retrieval that exercised the sparse fallback. The 24-token
c1 stream measured 2.26 decode tok/s versus 2.13 in the prior NZ graph
probe; the short samples are not an isolated or statistically strong speed
comparison. The known strict-quality misses remain the same three cases,
so Gate-A is still 17/20 and **not passed**. Do not attribute a large decode
win to the kpool skip.
