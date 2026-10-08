# GLM NZ-graph four-rank trace, 2026-10-03

The same four 310P devices ran GLM-5.3-Flash W4through32-noclip with
device-resident MLA, NZ-packed routed experts and FULL_DECODE_ONLY graph
capture at batch sizes 1 and 4. The source was the known-good
`/srv/ai/src/glm-w2-rowtile-20261002` snapshot; the model config and index
SHA-256 values were respectively
`bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`
and `df84e6a7eccd6f0896455f0b35e78b6f5ac3cfae00d88e0b350115bbf109fefb`.
The launch script is `tmp/serve-glm-rowtile-graph-nz-trace-20261003.sh`.

The c1 36-prompt/32-output-token request was valid at 2.211 decode tok/s.
The four-stream 32-token requests were valid at 4.868 aggregate decode
tok/s. These are client measurements with profiling enabled; the c4 profiler
stopped after 32 worker iterations, before all four requests completed.

| Approximate phase, rank 0 | Cross-rank envelope | Grouped W2/W4 tasks | AI-CPU BF16 Cast tasks | Collective tasks |
| --- | ---: | ---: | ---: | ---: |
| c1 prefill | 3,317 ms | 1,919 ms | 0 ms | 301 ms |
| c1 decode | 14,472 ms | 4,133 ms | 3,042 ms | 1.8 ms |
| c4 prefill | 8,990 ms | 4,110 ms | 98 ms | 664 ms |
| c4 decode (profiled) | 18,730 ms | 7,226 ms | 2,971 ms | 5.0 ms |

The phase boundaries are approximate. Task times are *attribution*, not an
additive critical path: task streams overlap. Decode collective arrival
spread medians were 0.167 ms at c1 and 0.227 ms at c4. Thus the old
collective-wait bottleneck did not survive graph capture; grouped projections
and mHC BF16 conversion are the next measured candidates. KDA and grouped
projection cost are substantial in prefill. See the complete all-rank
`summary-c1-phases.json` and `summary-c4-phases.json` for every category,
kernel shape and arrival statistic.

An isolated dependency-safe mHC probe rounded the residual before computing
the three pre outputs, then batched only those independent outputs into one
BF16 round trip. Results in `batched-mhc-round-legal-probe.json` were bitwise
equal for random c1/c4 tensors; median 0.498 versus 0.992 ms at c1 and
0.603 versus 1.059 ms at c4. This is a microbenchmark, **not** a serving
speedup.

The default-off `ascend_glm_mhc_batched_round` candidate captured the same
FULL_DECODE_ONLY graphs on the same checkpoint and grouped OPP. With the
same profiler configuration, one valid 32-token c1 request measured 2.410
tok/s versus baseline 2.211 (+9.0%), and four valid 32-token requests
measured 5.243 aggregate tok/s versus baseline 4.868 (+7.7%). The complete
20-case strict quality suite remained 17/20; all 20 final answers exactly
matched the saved baseline, including the same three misses. The unit suite
covering rounding parity and the residual-before-pre dependency passed 7/7
in the remote vLLM environment. These serving numbers are single paired
runs; the candidate has **not** been promoted. Candidate four-rank profiler
exports show the same reduction on every rank. In the completed rank-0 c1
operator statistic, AI-CPU Cast count was **26,080 → 14,560** and Cast task
time **3.198 → 1.852 s**. The 11,520-task reduction is exactly 360 fewer
tasks per profiled step across 32 steps: two fewer BF16 round trips, or four
Cast tasks, at each of the 90 mHC sublayers. The grouped-projection count
stayed 2,688 and its task time was 6.052 versus 5.972 s, within a small
single-run difference.

The isolated code and unit-test change is signed commit `2f2404a77`.
The experimental server runs from the separate
`/srv/ai/src/glm-mhc-batch-20261003` source tree; the known-good tree was
not modified. It remained healthy on port 8001 at the end of the test,
using about 41.5 GiB of process memory per NPU.

The all-rank analyzer's full captured windows (`baseline-summary-c1.json`,
`baseline-summary-c4.json`, `candidate-summary-c1.json`, and
`candidate-summary-c4.json`) use one analyzer version on both configurations:

| Capture | Baseline envelope | Candidate envelope | Rank-0 AI-CPU Cast tasks | Rank-0 AI-CPU Cast time | Rank-0 grouped time |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1 | 17.789 s | 16.867 s | 25,265 → 14,105 | 3.042 → 1.737 s | 6.052 → 5.972 s |
| c4 | 27.720 s | 25.948 s | 24,450 → 13,650 | 3.069 → 1.863 s | 11.337 → 10.928 s |

All four ranks had the same Cast counts within each capture. The analyzer's
cross-rank time envelope contains the profiled prefill and decode work; it is
not a critical-path decomposition. Its event clipping excludes one boundary
step relative to the CANN whole-capture `op_statistic.csv`, hence the lower
counts in this table.

An additional isolated NPU probe at 512 rows (`batched-mhc-round-prefill-probe.json`)
found bitwise-equal results and 12.649 versus 13.194 ms median for the legal
batching schedule (30 repetitions). The larger concatenation therefore did
not regress this specific operation, but a matched serving-prefill check is
still needed.

A server-tokenizer-calibrated 2K retrieval returned the exact code with 2,053
served prompt tokens, 45.37 s TTFT and 49.38 s total time. It passed its
prompt-target check. The prior baseline's 2,085-token retrieval took 52.46 s;
the prompt lengths differ, so this is only a prefill-regression spot check,
not an isolated speedup measurement.

NPU memory at the baseline server's idle state was about 42.7–42.9 GiB used
per device, with about 3.9–4.5 GiB free. vLLM reported 24,718 cacheable
tokens across the four-rank cache and served with `--max-model-len 16384`.
There may be room to raise the single-request limit, but no longer-context
capacity or quality claim follows from this headroom alone.
