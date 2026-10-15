# Grouped Mamba commit in the resident QFN service

The grouped native copy kernel is now bound to the actual model banks and
enabled in `qsa_combined_native_commit`. **CPU bookkeeping and the sampled-token
readback remain on the dependent execution path.** This is a qualified copy
implementation, not completion of the device-controlled iteration design.

The service remains `qwen38-flash-next` on port 8001. Workers
1638646/1638647/1638648/1638649 were retained. One drained model graph recapture
succeeded without restarting the worker group. General and prefix captures
remain `[3, 6, 9, 12]`; private commit graphs cover one through four requests.

## Live correctness

The resource binds 72 actual banks per rank: 36 FP16 convolution banks shaped
`[64, 5, 2560]` and 36 FP32 recurrent banks shaped `[64, 12, 128, 128]`.
Warmup and capture use no-copy metadata, preserving live state. Compact
postprocess mappings supply physical state slots; scheduler global IDs do not.
An event protects pinned metadata buffers from reuse before transfer completion.

Ten model requests exercised short, alignment-boundary, 8K, mixed C4, and return
to short workloads. Diagnostic mode checked 216 updates and 216 destination bank
contents per rank against independent CPU snapshots, including raw bitwise state
comparison. All four ranks reported no fallback or native failure. Diagnostic
mode submitted the graph on every step; normal mode submits only when host
admission finds a copy. Normal mode performs no receipt readback or replay
counter update. The CPU still computes and publishes next accepted counts.

Scoped regressions passed: 232 tests. Scoped repository hooks passed; the full
repository suite and secret-scan hook were not run. Existing baseline generated
token variability remains unresolved; state-copy parity is not a full model
quality qualification.

## Unprofiled end-to-end measurements

| Measurement | Native commit off | Native commit on |
| --- | --- | --- |
| C1 HTTP decode, median of three 256-token trials | 30.525 tok/s | 30.811 tok/s |
| C4 aggregate HTTP, four 256-token requests | 63.912 tok/s | 64.234 tok/s |
| Cold 8192-token prompt, one output token | 412.072 prompt tok/s | 410.547 prompt tok/s |

These small differences do not establish a serving speedup. Cold measurements
are one trial per mode, with zero cached prompt tokens. The roughly 29x private
copy-dispatch improvement does not apply to whole-model execution. Actual state
copies are infrequent, and the implementation has not removed the host transition
paid on every iteration. Both facts matter; this A/B does not isolate the cost
of removing CPU bookkeeping.

## Fresh profile and the remaining host dependency

The all-rank C1 prefix profile contains six complete steady iterations per rank.
Median iteration times span 85.20–85.63 ms. The draft-to-next-iteration interval
spans 12.52–13.78 ms, with 7.41–8.83 ms containing no recorded device task.
This interval is a target for investigation, not a demonstrated amount of CPU
overhead or recoverable time. Graph submission API time sums to only
0.083–0.111 ms per iteration.

The current dependency chain is:

1. Device sampling produces token IDs and acceptance information.
2. `qfn_acceptance_snapshot.make_bookkeeping` calls `sampled.cpu()` and retains
   CPU counts; `_bookkeeping_sync` parses outputs and updates host request state.
3. The state-update wrapper passes that CPU snapshot to `ResidentMambaCommit`.
   Host admission checks mapping bounds, decides whether copies are needed, and
   computes the counts published back to host bookkeeping.
4. When needed, pinned metadata and compact tables transfer to the NPU and the
   host submits the grouped copy graph. The kernel also computes copy decisions,
   but it is downstream of the retained host observation.
5. Host scheduler/input preparation and remaining PLE, draft routing, and
   selection callbacks still gate dependent work.

The next closure must consume device acceptance directly, commit state, and
construct the next inputs in an ordered device chain. Request identities,
compact mappings, speculative rejection, position updates, stop/cancellation,
and bounded cache ownership must remain consistent across that chain. Output
readback should observe completion without gating the next admitted step.
Merely moving the current Python bookkeeping into C or replaying this commit
graph on every iteration would not establish that property.

Rank 0 also spends kernel-duration sums of 20.67 ms per iteration in MatMulV2,
4.84 ms in TransData, and 3.64 ms in Cast. These sums are not independent
critical-path savings. No native commit copy occurred in this short steady
profile window, consistent with selective boundary execution.

## Evidence and lifecycle

Receipts, A/B results, and the all-rank profile summary are in
[`mamba-commit-live-v1/`](mamba-commit-live-v1/). The native resource is
`qfn_mamba_commit_v4`; normal flags are `enabled=true`, `verify=false`,
`all_steps=false`. All profilers stopped and released ownership after collection.

This is a resident integration, not a launcher default. A future restart must
restore the qualified combined candidate and then load, install, and enable the
native resource using `qfn-mamba-commit-control.py`. Preserve the frozen native
manifest assets. Do not copy the private control token into repository evidence.
