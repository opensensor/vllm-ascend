# Six-chip Qwen cast, layout, and residual audit

The serving runtime was updated through drained resident transactions. Images,
six 262,144-token sequence slots, MTP2, and decode graphs `[3,18]` were retained.
The 94°C hold, all-six-at-85°C resume, and separate 96°C cutoff stayed enabled.
No model weights or checkpoint files changed. The planner reports 1,607,731
cache tokens; six fully occupied 256K sessions have not been stress-qualified.

## Current status

The runtime is stopped for idle-memory attribution and cooling. Its last
candidate was `optimized_safe_hc`: histogram routing, qualified GDN RMS, short
speculative counts, and established HC arithmetic. The native residual was
temporarily removed while investigating independent projection differences.
Do not describe the stopped server as ready for requests.

The new residual ownership schedule is component-qualified, but its final
whole-model serving, image, and sustained thermal gates remain queued. The
historical 50% C1 target remains unmet: the last median was 28.277 tok/s against
19.262 tok/s, approximately 46.8% higher. The required threshold is 28.893 tok/s.

## Validated changes

- Expert histogram counts convert FP32 through INT32 before returning INT64.
  Keys and counts are exact within the existing 2^24 bound. The six-chip
  2560-token routing microbenchmark improved from about 1.70 ms for the
  comparison matrix to 0.25 ms for the fixed histogram. Histogram selection
  remains opt-in; the generic comparison default is preserved.
- Short Boolean speculative rows accumulate in INT32, then return INT64.
  This applies only to 310P NPU rows of width at most nine. Large reductions
  retain their original path. The actual proposer regression covers discarded
  requests, invalid tokens, backups, gathering, and unchanged inputs.
- GDN output RMS uses the existing native RMS operation and an instance-owned
  FP32 gamma cache for up to 18 decode rows. The original z projection,
  sigmoid, FP16 rounding, output projection, and TP reduction remain intact.
  All 36 real-weight GDN modules passed 2592 exact cases across six ranks,
  row sizes 3/6/18, and four input patterns.
- HC prefill retains the original four-row injection GEMM. The native residual
  replaces full-pipeline barriers with separate input-buffer and output-buffer
  ownership events, retaining vector arithmetic barriers, tile sizes, separate
  FP32 multiply/add, and the qualified saturating FP16 cast. It does not enable
  native arithmetic on small decode rows.

The residual ownership schedule passed 252 byte-exact synthetic cases on six
chips, including nonfinite values, overflow, signed zero, partial final tiles,
and changing inputs. With identical real-weight coefficients it matched both
reference arithmetic and the older kernel at all 98 HC layers on every rank
(588 cases). At 2560 rows, its isolated wall median was about 0.85 ms versus
1.66 ms. This is a component result, not a whole-model throughput multiplier.

CPU validation covers 89 distinct tests: 75 changed-candidate/proposer tests,
nine direct-residual wrapper tests, and five idle-probe admission tests. The wrapper/candidate receipt contains
26 tests, including the 17 candidate tests already in the 75-test receipt.

## Serving measurements before the barrier change

Same-shape trials used an uncached 128-token prompt with 256 output tokens and
an uncached 4096-token prompt with 16 output tokens. The short arm had one
warmup and two measured requests. These are preliminary observations; they
are not repeated reverse-order statistical A/B qualification.

| Variant | Short decode tok/s | Cold 4096-token TTFT |
| --- | --- | --- |
| Baseline | 28.269, 29.295 | 10.370 s |
| Histogram and shared HC operand | 27.724, 26.782 | 10.036 s |
| Histogram and native HC at all tested sizes | 28.677, 27.835 | 9.470 s |
| Histogram and native HC only for prefill | 29.136, 28.487 | 9.729 s |
| Previous row plus GDN RMS | 28.717, 28.167 | 9.500 s |

The prefill observations improved cold TTFT by roughly 6–9%. Small decode
changes did not establish a repeatable gain. The historical three-prompt
256-output-token median after the short-count change was 28.277 tok/s;
individual results were 30.963, 26.177, and 28.277. These are historical-control
comparisons, not a fresh paired two-card measurement.

Text coherence, fresh/cached screenshots, 1024×1024 image processing, six
cached prefix branches, and cancellation passed before the final short-count
and barrier changes. Those final combinations still require a serving rerun.
The retained serving trials reported zero checkpoint spills and restores.

## Physical transfer findings

The matching before/first-after capture used a 456-token cold prompt, 12 output
tokens, and 528 target INT4 matmul calls. Rank 0 had 18,423 versus 18,227 Cast
calls. Large 384×10240 FP32-to-FP16 casts fell from 297 to 101; route comparison
Equal calls fell from 113 to 15. TransData stayed at 12,248 calls and explicit
copy tasks were nearly unchanged. Summed Cast time fell from 126.2 to 71.9 ms,
but task durations overlap and profiling perturbs execution. This does not
establish a 43% wall-time or total-traffic improvement.

The subsequent GDN capture had 480 target INT4 matmul calls rather than 528.
Its raw 16,182 Cast and 11,053 TransData counts must not be interpreted as
like-for-like reductions without normalizing the number of model passes.
The earlier capture profiled six ranks; later captures profiled rank 0 only.
Driver bandwidth counters on ranks 4/5 were implausible in the earlier full
capture, and explicit copy payload/direction attribution remains incomplete.

The ten-second passive idle capture recorded DDR reads around 43–44 GB/s on
rank 0 with no inference kernels or explicit copy submissions. It contained
only two EVENT_RECORD and 36 PLACE_HOLDER_SQE device tasks. The parser reported
no ACL-to-NPU flow events. The CLI simultaneously reported 47% DDR bandwidth
usage, zero AI Core/AI CPU usage, and zero running/waiting requests. After all
workers exited, CLI DDR bandwidth usage briefly fell to zero, then returned
to 47% with no model or profiling process present. The third card reached
95°C while unloaded. This corrects the initial attribution to the serving
runtime: the source may be profiling, driver behavior, hardware, or telemetry.
Neither HCCL nor graph retirement is established as the cause.

The third card continued warming during idle, approaching the thermal hold
threshold. Worker shutdown completed with no remaining API/engine/TP process.
A clean-runtime idle comparison must precede further serving trials. No new
workload should be submitted until all six chips pass cooling admission.
The queued model-free probe is `tools/qwen4exp/benchmark_idle_ddr_310.py`;
it rejects admission above 72°C or with any missing sensor before spawning
children, and aborts experiments at 90°C. Its NPU phases remain untested.

## Rejected trials and accuracy boundaries

Joining or padding the injection projection changed its FP16 result at real
model weights. A separate NZ shadow of its original four logical rows also
failed exact parity, including a later 18-row layer. Neither is active.
The large INT32 comparison-matrix reduction was slower and was rejected.
Experimental fused GDN norm/gate kernels with real-weight failures are disabled.

The expanded independent HC projection gate also found output differences.
Subsequent checks found differences between repeated *baseline* projection
calls while saved normalization and mixed outputs remained equal. Both native
residual versions matched each other and reference arithmetic for identical
coefficients. Preserve those negative receipts: independent projection reruns
are not a valid isolated residual comparison when the baseline itself varies.

## Evidence and next gates

Raw profiling databases remain on the serving host. The JSON summaries include
SHA256 fingerprints. Exact remote sources are frozen as `.txt` to prevent
format hooks from rewriting the bytes used by the receipts. Canonical source
is in `tools/qwen4exp/native_hc_residual.cpp`; the executed host compiler/bridge
recipe is retained in `frozen-sources/build-hc-v3.sh.txt`. The optional resident
candidate requires the explicitly loaded `hc_residual_v3` resource.

Next, compare idle bandwidth after device initialization, bounded allocation,
HCCL initialization, eager all-reduce, graph capture, and graph retirement in
separate clean processes. Use CPU rendezvous between phases so an HCCL barrier
does not contaminate the idle measurement. Preserve all six temperature samples
and stop at the bounded experiment limit. HCCL mode changes need confirmation
of support in the installed 310P/CANN combination; no speculative SDK mode is
currently enabled. Huawei documents product-specific support for
[HCCL expansion modes](https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0096.html).

After idle attribution and cooling, rerun final images/prefix/cancellation and
matched cold/decode measurements. Then compare speculative depth with matching
graph sizes, keeping six slots and images. MTP3 requires `[4,24]`, a fresh
engine, and new state/graph qualification; it has not been enabled or measured.
Full six-by-256K and sustained thermal qualification remain open.

The broad `bash format.sh ci` check was run. It failed on existing repository
lint/forbidden-import findings and reformatted unrelated files; those unrelated
changes were restored. Scoped hooks for the delivered files are the release
check. The full unit suite remains unavailable in this local environment
because importing upstream vLLM requires missing `flash_linear_attention`.
