# Packed-W3 prefill after Cube512 QSA, 2026-10-04

The qualified Packed-W3 TP4 graph server on port 8001 uses the Cube512 QSA
package, the original NZ-packed W3 grouped kernel, 640-token prefill chunks,
and a configured 192K context. Its matched 7,877-token retrieval improved from
180.49 s to 99.04 s TTFT with the exact answer; the strict quality score stayed
17/20 with the same three misses. The full 192K prompt has not been tested.
The Cube512 result and package identities are in the adjacent
`qsa-cube512-20261004` study and commit `de6f8aa02`.

## Four-rank post-Cube profile

A two-iteration 640-token prefill capture used the same qualified code and
OPPs with profiling enabled. All four worker traces were parsed under capture
token `2026100412490692`; the summary is `profile-summary.json`, and the full
trace is on the NPU host at
`/home/matteius/experiments/glm-w3-20261004/profile-w3-cube512-prefill640-20261004`.
The correct 2K retrieval took 22.79 s TTFT during profiling. The two-step
cross-rank device task envelope was **15.688 s**; this is a profiled device
window, not the unprofiled serving TTFT.

| Rank-0 category | Calls | Summed device task time |
| --- | ---: | ---: |
| Grouped packed W2/W3/W4 | 168 | 5.315 s |
| KDA and convolution | 1,564 | 3.456 s |
| Transfer and layout | 6,968 | 2.889 s |
| Cube512 QSA | 11 | 0.693 s |

The transfer category includes 356 `CastAiCpu` calls on `[640,4,4096]`,
totalling **1.730 s**. KDA's `ChunkKdaFwd` shape `[1,640,16,128]` accounts
for **3.228 s** in 476 tasks. These category sums can overlap and do not form
an additive critical path. QSA was 5.162 s in the preceding vector-QSA
profile, so the next major work is grouped weight decode/matmul and KDA, with
BF16 state conversion also measurable.

## W3 vector-scale experiment: rejected

An isolated clean CANN build changed NZ W3 scale application from eight scalar
fractal scales to a vector gather and multiply. The first manual OPC binary
was invalid and returned zero rows; a clean package build produced a valid
binary. The clean candidate passed all **16/16 bitwise 310P cases**, including
W2/W3/W4 canonical and NZ layouts. It was slower on the two realistic NZ W3
cases: **1.308 to 1.650 ms** for the GLM gate shape and **8.870 to 11.213 ms**
with sixteen active experts (three synchronized repeats each). The canonical
GLM gate shape was flat at 5.273 vs 5.237 ms. The paired case records and
binary hashes are in `w3-control-16cases.json` and
`w3-vector-scale-16cases.json`. This is a no-go for serving; its local source
edits were reverted, and the server kept the qualified W3 OPP.

## BF16 state conversion experiment: rejected

An isolated 310P AI Core vector rounder was compiled to replace the current
scalar bit rounder for large mHC tensors. The 310P vector API does not support
the required 32-bit `And`; a shift-based version compiled but failed bitwise
NPU parity, including 16-element and prefill-size cases, even after explicit
vector pipeline barriers. It was never put in the serving package. The local
source and build-only staging source were restored. The current scalar AI
Core rounder is also slower than the native BF16 cast pair on `[640,4,4096]`
(34.31 vs 10.32 ms in a direct one-card comparison), so its 65,536-element
limit must remain in place.

The qualified server was restored and returned HTTP 200 from `/health` after
these isolated kernel tests. Neither kernel candidate was promoted.

## Opt-in FP16 mHC state: faster serving candidate

The existing `ascend_glm_mhc_fp16_state` HF override converts the mHC state
through FP16 instead of BF16. This avoids the AI CPU BF16 cast pair, but FP16
has a narrower exponent range and different rounding from the BF16 reference.
The exact tested launcher is `serve-selective-w3-prefill-fp16-mhc-20261004.sh`.
It changed only that override relative to the qualified Cube512/histogram
server; Packed-W3 and QSA OPPs, TP4 graph capture, 640-token chunks, 0.70 KV
fraction, and the 192K configured limit stayed the same. The server advertised
240,402 KV tokens, 1.22 times that limit.

| Gate | Qualified BF16 | Opt-in FP16 mHC |
| --- | ---: | ---: |
| Matched 7,877-token retrieval TTFT | 99.04 s | **85.74 s** |
| 8K retrieval answer | exact | exact |
| Strict answer suite | 17/20 | 17/20, same three misses |
| 256-token c1 decode | 3.548 tok/s | 3.653 tok/s |
| 256-token c4 aggregate decode | 7.939 tok/s | 8.093 tok/s |

All five short completions reached 256 tokens. Nineteen of the twenty strict
final strings were identical; the one changed response was the already-failed
`instr_reverse` case. A 32,453-token retrieval also returned the exact code
with 338.95 s TTFT. The paired 8K prompt served the same 7,877 tokens and
returned the same code on both servers. These are single serving runs, so the
small decode difference is not a firm throughput claim. The 8K TTFT gain is
13.4% over the qualified BF16 run, or 52.5% below the pre-Cube 180.49 s run.

Request records, summaries, and the exact launcher are alongside this README.
The opt-in server is running on port 8001 as PID `1072428`. This mode remains
experimental because FP16 can overflow above 65,504 and the full 192K prompt
has not been tested. The BF16 launcher is preserved on the NPU host at
`/home/matteius/experiments/glm-w3-20261004/serve-selective-w3-prefill-histogram-20261004.sh`.
