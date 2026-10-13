# Qwen live-coding follow-up: offline fixes

The live six-chip server remains unchanged. This work used local source, the
previously collected passive observations, and CPU tests. No inference,
profiling, worker RPC, pause, restart, or deployment was performed.

## Evidence and scope

Eleven completed coding requests from 17:38–17:51 UTC on October 10 had
427.33 seconds of prefill and 442.15 seconds of decode. Prefill accounted for
49.15% of those combined phases. New prompt tokens divided by logged prefill
time gave 333.31 tokens/s; median per-request decode was 17.0 tokens/s.
One request processed 30,896 new tokens alongside 37,120 cached tokens in
92.37 seconds. Queue time was negligible.

Passive metrics reported 4,203 draft rounds and 8,406 drafted tokens, with
2,925 acceptances in the first position and zero in the second. There were no
checkpoint spill/restore, graph-fallback, or runtime-error messages in the
inspected interval. These observations do not identify the precise device
traffic or prove sustained thermal stability. They also do not qualify
concurrent prefill/decode fairness: at most one running request was sampled.

The current profile already uses 2,560-token grouped prefill, built-in FP16
SwiGLU, and CANN v2 finalization. Those are retained. The image encoder,
weights, sharding, cache geometry, and thermal controls are unchanged.

## QSA metadata reuse

NZ K/V gathers previously prepared the same selection vectors and block
table separately for both kernels and each query tile. The batched attention
path now prepares contiguous INT32 metadata once on the producing stream.
Both gather streams share it after the existing readiness event, and retain
its allocation through their existing `record_stream` ownership protocol.
The unused NZ-path group-index cast was removed. Already contiguous INT32
inputs remain views; the public gather adapters still accept other integer
dtypes and strided inputs.

The CPU oracle executes the production orchestration with independent page
translation and selected-token attention. It covers two requests, different
page orders, causal tails, padding, three query tile sizes, serial/parallel
gathers, and three dtype combinations. Outputs match the FP16 reference
exactly. Dispatch counters observe five conversions for all-INT64 metadata,
three for INT32 indices/table plus INT64 geometry, and zero for prepared
INT32 metadata, independent of tile count. These are copy counts, not an NPU
latency measurement. A full INT32 selection can increase temporary metadata
residency compared with preparing one tile at a time; no host transfer is added.

## MTP graph identity

The serving implementation wraps each draft-model call in a breakable graph;
it does not capture the entire Python draft loop. Consequently, the loop's
attention-metadata assignment executes on every proposal. The initial idea
that this assignment itself was skipped does not describe this runtime and
was discarded.

The actual graph entry key contained only the padded batch descriptor. Draft
steps can share that descriptor while binding different persistent
`slot_mapping_group` and `query_start_loc_group` buffers. Replaying the first
step's graph for the next step retains the first step's captured operand
addresses. QSA's eager indexer can read the new metadata while captured cache
writes still address the previous step's slots.

The Ascend wrapper now specializes draft entry keys by the metadata map's
position in `draft_attn_metadatas`. It preserves every original descriptor
field and restores the outer descriptor even on capture failure. Target and
eager dispatch remain unchanged. Metadata object identities are used only to
find the step in the current proposal; they are not stored in graph keys.
All entries remain in the existing graph dictionary and are retired by its
existing `clear_graphs()` method. No additional synchronization, tensor copy,
environment variable, or global mutable state is introduced.

The CPU regression executes upstream's actual graph-entry dispatch with a
device-address oracle. The legacy shared key overwrites slot 5 with the
second proposal and leaves slot 6 empty. Separate step entries write both
slots correctly, retain their own persistent operands on subsequent proposals,
and survive new Python metadata maps. This establishes the address-collision
defect offline. It does **not** establish how much of the live zero second-
position acceptance is caused by that defect.

More entries can consume additional graph/stream resources. Dummy-capture
versus real-proposal query geometry, padding, and image causal positions must
also be checked on hardware before enabling this candidate. Do not claim
that separate keys alone complete the MTP accuracy gate.

## Cache-preserving cutover requirements

The current `ResidentClient.switch()` pauses with `clear_cache=true` and calls
`resident_reset()` before and after graph capture. Qwen reset also discards
prefix-Mamba tier metadata. In addition, whole-model dummy capture can zero
GDN/Mamba state rows. Merely changing the pause query to `clear_cache=false`
does not make this procedure cache-preserving.

For a future user-authorized cutover:

1. Drain with `mode=wait&clear_cache=false`; preserve thermal-pause ownership.
2. Install the eager QSA replacements and every imported alias that calls them.
   Their arithmetic, cache layout, and stored values are unchanged.
3. Retire and recapture **draft** graph entries for the MTP correction. Preserve
   target graphs, target KV pages, image caches, and prefix-Mamba checkpoints.
   Avoid the current whole-model capture/reset path. Lazy capture on a real
   partially occupied padded bucket is insufficient: capture must exercise the
   full bucket with correct per-step query geometry and stable buffer addresses.
4. Protect draft capture from writing scheduler-owned cached pages. Stage
   invalid slot mappings and isolated dummy page metadata, then restore staging
   buffers. Verify whether old draft-cache entries remain valid under corrected
   addressing; selectively rebuild affected draft state if required.
5. Verify all six acknowledgments, unchanged resident storage/checkpoint IDs,
   and clean graph state. Compare cached text and fresh/cached image results
   before resuming the owned pause.

That path is specified here but is not implemented or hardware-qualified.
Do not run the existing clearing switch as a substitute when the user requests
cache preservation. The current live session is not a maintenance window.

## Validation and next hardware gates

Run the offline regression suite with `--noconftest`; the ordinary UT conftest
requires NPU-only dependencies. Receipts accompanying this report record the
exact commands and results. The final run passed 188 tests. Two existing
resident-reset fixtures initially failed because they lacked the rank and
model now required by the transfer audit; those fixtures were updated without
changing reset behavior. The gather kernel is replaced by a CPU oracle,
so these tests do not qualify NZ hardware layout or asynchronous device races.

All manual pre-commit checks passed for the changed files. Required
`bash format.sh ci` was also run; repository-wide checks still fail on existing
unrelated lint/import/documentation issues. Incidental formatting edits to
unrelated files were restored and are not part of this delivery.

When a controlled NPU window is granted, qualify draft-only capture at both
configured buckets, changing request counts, rejection lengths, and image
contexts. Check per-step slot pointers and padding before testing acceptance.
Then compare MTP1/MTP2 on the same greedy and sampled prompts, and compare
23K–31K uncached prefill with unchanged prefix/image workloads. Measure wall
TTFT, decode gaps, acceptance by position, graph fallback, spills, and all-six
temperatures. Keep the 94°C hold and all-six-at-85°C resume.

The v3 residual schedule remains a separately component-qualified candidate;
its combined whole-model gate is pending. No new throughput, thermal, or 50%
improvement claim is made by this offline delivery.
