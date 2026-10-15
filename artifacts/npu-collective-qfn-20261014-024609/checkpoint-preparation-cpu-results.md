# Device checkpoint preparation and deferred count readback

## Retained scope

The existing `qwen38-flash-next` service on port 8001 now executes checkpoint
pre-copy decisions, grouped state copies, and accepted-count normalization on
its NPUs for unchanged MTP2 decode batches of one through four requests.
The Mamba cache policy, state layout, and recurrent math are unchanged.
No service restart or model graph recapture was required for this change.

`tools/qwen4exp/checkpoint_preparation.py` expresses the checkpoint operation
using the already qualified grouped-copy ABI. For destination D, acceptance A,
and block size B, synthetic running position `(D+1)*B-(A-1)` causes that kernel
to perform the original pre-copy with bias `A-1`. Non-copy and invalid rows use
inert metadata. Every possible acceptance bias is admitted against the current
compact page mapping before submission.

`checkpoint_preparation_runtime.py` owns four private metadata/copy graphs and
retains their native code, tables, and live-bank references. Source-checked
hooks replace the checkpoint count publication and bypass the accepted-count
wait/upload in the previously qualified input preparation. The engine and
worker still perform CPU scheduling and lifecycle work.

Normalized counts remain on the device during continued decode. A deferred
readback is materialized before insertion, compaction, reordering, preemption,
or a fallback consumer. Publication waits for both device acceptance and any
previous legacy D2H writer before copying into the CPU array. A mismatched row
identity fails closed. The original count publication remains available for
unsupported rows. Controls drain the service before changes.

## Validation

- 301 focused CPU tests passed; Ruff check and format checks passed.
- Each rank passed 196 live decision comparisons and 216 exact state-block
  comparisons before the fast checkpoint path was enabled.
- Ten model cases passed in verification mode, fast mode, and deferred-readback
  mode. These include checkpoint crossings, 8K input, and concurrency four.
- Twelve serving cases passed after each integration stage. The final rerun
  includes token budgets 1/2/3, penalties, logprobs, unequal request turnover,
  structured JSON, cancellation, and a subsequent request.
- Exact comparison covers component decisions and copied state banks. Model
  validation is a semantic smoke gate, not broad quality qualification or
  deterministic-generation parity. Generated token sequences differed between
  several suite runs, including repeated prompts within a single mode; these
  receipts do not establish whole-model numerical reproducibility.
- No model weights, worker PIDs, public model name, or port changed.

## CPU profile

Rank 0, same 128-output-token diagnostic; both runs had 52 iterations:

| Function/call | Previous retained configuration | Device checkpoint + deferred readback |
| --- | ---: | ---: |
| CPU acceptance synchronization helper | 52 | 2 |
| Original CPU checkpoint preprocessing | 52 | 2 |
| Host-buffer `copy_to_gpu` | 689 | 589 |
| Original device-commit retirement | 52 | 1 |

The remaining helper calls are fallback boundaries. The new retirement wrapper
runs 52 times and keeps counts on device when admitted. `materialize_counts`
is also called at harmless empty boundaries; its invocation count is not a
D2H-transfer count. Owner counters distinguish actual materializations.

The new pre-copy wrapper accumulated 108.33 ms over 52 calls, versus 3.66 ms
for the old CPU pre-copy function. CPU profiling includes native submission and
device waits, so these are not measurements of CPU arithmetic alone. This is
an explicit remaining cost: fuse the tensor decision operations and grouped
copy dispatch rather than assuming graph capture itself makes them free.

See `resident-live-v7/cpu-checkpoint-comparison.json` and both raw profile
folders for all four ranks. No profiler remains active.

## Short throughput A/B

| Mode | C1 median HTTP output tok/s | C4 aggregate HTTP output tok/s |
| --- | ---: | ---: |
| Checkpoint optimization off | 32.141 | 67.710 |
| Checkpoint + deferred counts on | 31.399 | 66.121 |

C1 is three 256-output-token trials; C4 is one finite batch of four 256-token
requests. These include prefill/API overhead; C4 includes request tails. The
candidate was about 2.3% lower in these short trials. There is no established
throughput improvement or confidence interval. It is retained under the user's
instruction to retain CPU reductions, with this regression reported explicitly.
Final boundary-writer ordering adds a wait only when materializing deferred
counts; serving checks were repeated after that correction, not the benchmark.

Receipts: `qsa-resident-v1/{decode-benchmark,concurrency}-checkpoint-v7-{off,on}-v1.json`.
At the pre-final status point each rank reported 942 avoided count waits,
2,080 avoided count uploads, 785 deferred readbacks, and 87 actual deferred
materializations. Those are cumulative experiment counters, not per-request
speedup measurements.

## Current service and restoration

- Model: `qwen38-flash-next`; port: 8001.
- Kilo entry: `vllm-310p-w4/qwen38-flash-next`.
- Stable log on hardware: `/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/server-current.log`.
- Run: `resident-live-v7`; final receipt: `resident-live-v7/final-checkpoint-status.json`.
- API/engine PIDs: 2167990 / 2168713.
- Worker PIDs: 2169038, 2169039, 2169040, 2169041.
- Post-benchmark temperatures: 76/75/80/79 C, all devices reported OK.

After restoring the earlier qualified owners, install checkpoint preparation
with `qfn-checkpoint-preparation-control-v7.py install --label FRESH`, run the
verification gate, then switch it `on`. On a fresh installation the deferred
readback controls are already installed, initially disabled; enable them with
`qfn-checkpoint-readback-control-v7.py on --label FRESH` and rerun serving gates.
The historical `upgrade` action was used only to upgrade the first live owner;
it is not required for a new owner made from the final source.

Both controls preserve the private token on the hardware host. No token is
archived here. Hot controls are not automatically restored by old launchers.

## Remaining work and ETA

Complete steady-state CPU bookkeeping removal remains unfinished. CPU request
admission/lifecycle, compact mapping, scheduling, attention metadata, PLE host
lookup, and graph submission still exist. This change removes the checkpoint
count round trip on admitted continuation rows; it is not an autonomous device
execution loop.

The next stages are fused checkpoint bookkeeping, bounded device-owned decode
windows with page ownership and cancellation contracts, and elimination of the
PLE host lookup dependency. A rough planning estimate is several more days for
the larger steady-state bookkeeping objective, with substantial uncertainty;
there is no validated completion date. Cache-policy replacement remains a
separate subsequent task under the user's requested ordering.
