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
OPP packages with profiling enabled. All four worker traces were parsed under capture
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
server; Packed-W3 and QSA OPP packages, TP4 graph capture, 640-token chunks, 0.70 KV
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
This mode remains experimental because FP16 can overflow above 65,504 and the
full 192K prompt has not been tested. The BF16 launcher is preserved on the NPU host at
`/home/matteius/experiments/glm-w3-20261004/serve-selective-w3-prefill-histogram-20261004.sh`.

## Prefix reuse candidate for coding sessions

The launcher accepts `on` as its twelfth argument to enable hybrid prefix
caching. Its default is `off`. The earlier W2/W4 hybrid-cache run reused 7,040
tokens in a repeated 7,269-token prompt and cut warm TTFT from 169.97 s to
7.93 s, but that run had a 22,528-token configured window. A Kilo request
recomputed all 30,052 prompt tokens with zero cache hits and spent 386.06 s in
prefill, making prefix reuse especially valuable for coding sessions.

The first 192K trial started on port 8001 with the same FP16 mHC/Cube512
configuration, prefix caching enabled, and a 0.78 KV fraction. Startup
advertised **249,160 cache tokens**, 1.27 times the 196,608-token limit, with
6.65 GiB allocated to KV. An exact repeated 7,877-token retrieval answered
`BLUE-ORCHID-7319` both times: cold TTFT was **85.38 s**, warm TTFT **4.08 s**,
and the request-timing log reported **7,680 cached tokens** on the warm run.
The matched client record is `prefix-cache-8k-trial-20261004.jsonl`. This
validates short-prefix reuse at the 192K configuration; it does not validate
a full-length prompt or workspace headroom under concurrency.

The next 32,453-token retrieval shared the prior 8K prefix. It answered
exactly with 7,680 cached tokens and 259.85 s TTFT; its exact repeat answered
exactly with 32,000 cached tokens and 6.05 s TTFT. The source prompt and
results are in `prefix-cache-32k-trial-20261004.jsonl`. This pair establishes
reuse at coding-scale prompt length, but the first request was not fully cold
and therefore did not test a greater-than-five-minute silent prefill.

## Why cold prefill remains slow

A later 30,052-token Kilo request on the FP16 mHC server took 386.06 s in
prefill, with zero cached tokens and zero queue time. Its 47 context iterations
processed 640 tokens each except the 612-token tail. The first ten iterations
had a 7.98 s median and the last ten an 8.58 s median. Most of the cost is
repeated per chunk; the observed context growth added about 8% across this
request. This request is outside the earlier profiled two-step window, so the
attribution below uses that prior BF16 profile and is directional for the
current FP16 mode.

Each 640-token chunk executes 84 grouped packed-weight projections: gate/up
and down in each of 42 routed layers. The profile splits those layers into
22 W4, 12 W2, and eight W3. The grouped kernel unpacks each active expert's
entire matrix to FP16 in GM workspace and then reads the tile back for Cube
matmul; it does this again for the next chunk. If all 72 local experts are
active in each layer, the shapes in the trace imply **152.2 GB of decoded
FP16 writes per rank per chunk**, plus roughly the same amount of Cube
readback. That is a conditional traffic estimate, not measured bytes. With
5,120 top-8 routes across 288 experts, even routing averages only 17.8 rows
per expert, so the decode cost is amortized over small per-expert GEMMs. The
rank-0 grouped calls consumed 5.315 s in the two-step profile. Their kernel
metrics had median vector ratio 0.51 and MAC ratio 0.03, consistent with
decode and staging dominating the arithmetic. W3-only improvements affect
about 23% of this grouped time; the shared W2/W3/W4 path is the larger target.

The 310P KDA path also runs nine physically separate stages per call because
the unified AI cores cannot preserve the necessary mixed vector/Cube event
state across one launch. The full rank-0 trace has 34 KDA calls per chunk,
or 306 stage launches, accounting for about 3.32 s across two chunks. This is
the second cold-prefill target after grouped weight conversion. A prior
128-channel L1 streaming W2/W4 trial passed parity but was 2.21% slower in
the aggregate; simply moving the decoded tile to L1 is not yet a proven fix.

The same Kilo turn exposed a separate transport failure. Its first model token
arrived only after 386 s of prefill; the preceding 20,643-token turn began
generating after 262 s and succeeded. The upstream chat endpoint wraps a
generator in `StreamingResponse`, but the generator sends its first SSE event
only after receiving an engine result and has no prefill heartbeat. Kilo's
second assistant message stayed empty and was later aborted. A distinct cold
32,451-token direct client request reproduced this: the engine completed the
prompt in 338.27 s and generated nine tokens, while the client received only
the response headers. The server TCP socket then showed 4,425 bytes sent but
only 168 acknowledged, with 3,730 bytes retransmitted and a backlogged send
queue. This is strong evidence that the long silent prefill leaves the network
connection stale before the first SSE event; the exact device or timeout that
drops it has not been identified.

SSE comment heartbeats during prefill should keep the client connection active;
they cannot reduce cold TTFT.
The opt-in `SSEHeartbeatMiddleware` in `vllm_ascend/_310p/sse_heartbeat.py`
uses the supported vLLM `--middleware` hook and emits SSE comments every
15 seconds until the stream ends. The launcher accepts `on` as its thirteenth
argument to select it. Its CPU tests passed; the first prefix-cache server
does not use this middleware, keeping the cache comparison isolated.
