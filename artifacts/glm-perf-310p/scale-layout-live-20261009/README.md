# GLM resident scale-layout trials — October 9, 2026

## US English

This trial loads six append-only native candidates into the existing four-rank
GLM server. It keeps the loaded checkpoint, decode v984, context limit 131,072,
1280-token scheduling budget and MTP1 configuration. The live API remains PID
77825; worker PIDs are 79013, 79400, 79888 and 80521. No server restart is needed.

The new down-scale layout stages each dense gate/up tile once, then presents
contiguous scale columns to the down projection. Sparse expert batches retain
the qualified layout. It reuses dead input-scale scratch, adds no tensor storage,
and preserves the quantization and accumulation order. At partial dense tiles,
it writes 512 bytes instead of the earlier smaller tail stores; this is an
explicit traffic tradeoff rather than a guaranteed speedup.

Rank-zero worker logging now reports the full prompt length, reuse at admission,
each prompt chunk, scheduled position and remaining tokens using existing CPU
scheduler metadata. A permanent scheduler patch also logs admission and cached
tokens on the next startup. Progress means scheduled work, not confirmed device
completion; the existing iteration log reports completion time. Logging follows
the existing detailed-iteration setting and uses the configured serving logger.

The final comparison instruments the actual prefill bridge submissions, rather
than copying the candidate version into a status field. It records operator
namespaces, each kernel object's binary hash, and stage counts after preparation.
This audit does not read device tensors or time device instructions. It also
releases the retired baseline reference scratch before final graph capture.

The two 640-token final chunks are intentional: the live common cache boundary
is 640 tokens. Identical replay must leave the prompt's final token to recompute,
so a 6400-token prompt needs its reusable KDA checkpoint at 5760 as well as its
prompt-end state. Processing the final 1280 tokens in one forward would need an
intermediate recurrent-state checkpoint, not simply removal of the split.

## Measured serving result

| Prefill version | Candidate | Cold TTFT pair (s) | Mean (s) | Change |
| --- | --- | --- | --- | --- |
| 1001 | baseline | 51.94, 51.64 | 51.79 | +0.0% |
| 1020 | input-group | 51.34, 50.87 | 51.10 | -1.3% |
| 1021 | down-group | 50.88, 50.64 | 50.76 | -2.0% |
| 1022 | reduce-cache | 51.42, 51.18 | 51.30 | -0.9% |
| 1023 | compact-down | 51.44, 50.82 | 51.13 | -1.3% |
| 1024 | expert-ends | 52.15, 52.23 | 52.19 | +0.8% |
| 1025 | nz-full | 65.15, 64.55 | 64.85 | +25.2% |

The subsequent audited comparison measured 52.277 seconds for v1001 and
50.930 seconds for v1021: a 2.576% latency reduction. Every rank submitted 258
pack, routed-input and reduce stages, plus 258 gate/up and 258 down stages
split between W4 specialization and generic W2/W3. Actual operator namespaces
were `glm_reconstruction_v1001.launch` and `glm_reconstruction_v1021.launch`;
the gate/up and down binary hashes differ. No fallback dispatch was observed.

The strict text comparison initially restored v1001 because the first tokens
were `q` and `b`. The baseline itself produced `L` earlier for that same prompt,
so exact text was not a reproducible gate in this session. An additional
1920-token shadow request then compared both kernels on the identical actual
activations and routes at all 43 MoE banks, with both 1280- and 640-token chunks.
All 344 comparisons across four ranks were bitwise equal. The shadow path was
removed, and v1021 was retained for prefill with v984 decode and complete graphs.
This narrows the tested arithmetic question; it does not explain baseline
output variability or establish natural-language quality over arbitrary prompts.

`serving-summary.json` records the initial timing/text guard decision;
`shadow-summary.json` records the subsequent actual-input gate and promotion.
The service passed `/health`, was unpaused, and had maintenance disabled after
promotion. Worker PIDs and weight-storage identities remained unchanged.

The final baseline decode control measured 10.14 generated tokens/s after TTFT;
decode kernels were not changed by these prefill candidates. These are short,
repeated-token controls and are not production-throughput or quality guarantees.

A fresh four-rank profile of the retained v1021 path completed a 6400-token
cold request in 50.909 seconds. This is an attribution run with profiler
overhead, not another paired speed result. All 576 collectives matched across
ranks. Rank 0 attributes 15.258 seconds to gate/up, 7.013 to down, 1.442 to
preparation and 1.361 to route reduction. Its device stage has 11.652 seconds
of communication without overlap and 0.154 seconds of device-free time.
These counters are not independent speedups to add together. Communication
wait includes dependency and protocol effects, not only expert imbalance.

One representative W4 gate/up task is 78.727 ms, with vector 56.604 ms,
scalar 26.077 ms and Cube MAC 1.201 ms. Pipes overlap; the sample does not
establish the same proportions for every task. It motivates redesigning the
readback/scaling/control stream rather than relying on INT4 math alone.
The [complete streaming proposal](STREAMING.md) includes an architecture
diagram, current/proposed stages, memory boundaries and acceptance criteria.
Weight reuse across large expert row batches already exists in the live path.

The profile was exported offline after stopping the profiler and restoring
ordinary v1021. All ranks report 38 incomplete memory records for allocations
whose lifetimes precede the trace; allocation data are not a complete memory
proof. Raw tables remain in
`/srv/ai/artifacts/glm-scale-layout-20261009/critical-tables.tar.gz`, with
per-table provenance in `critical-table-hashes.json`. Condensed attribution
and representative counters are saved alongside this report.

An offline scheduling worksheet also identifies a mixed-batch cliff: one
two-token MTP decode leaves only 1278 of the 1280-token global budget, which
rounds the interior prefill chunk down to 640. A 1296-token total budget with
a 1280-token prefill cap leaves decode headroom. Known shared MoE scratch
grows by 1.914 MiB/rank, excluding other model/runtime allocations. This
candidate is not launched: `headroom-analysis.json` explicitly lists the
remaining whole-model memory and startup checks.

## Validation and limits

All six candidates compiled with CANN and passed synthetic plus real-checkpoint
W2/W3/W4, A4/A8, changed graph replay and all-peer gates. Resident swaps further
compare output bits against v1001 using the loaded 72-expert banks on every rank,
including dense-to-sparse route changes. Each serving comparison clears prefixes
and uses the same token-ID prompt; these repeated-token performance inputs are
not a natural-language accuracy evaluation. Generated text is retained for
comparison, not taken as evidence of general model quality.

The CPU GLM suite passed 1998 tests. Scoped manual hooks passed. The full
repository formatting check failed on existing files outside this change; see
`repository-format.log.gz`. Formatting ran in an isolated worktree.

Serving results are recorded after completion in `serving-summary.json`. Build
and standalone gate records are in `build-results.json` and `gate-results.json`.
The exact controller is preserved as `live-queue-r3.py.txt`; candidate helper
sources and binary hashes remain in the remote provenance directories.

Two initial controller attempts failed before candidate installation: an invalid
hyphenated control name, then replacement of the full v1001 resource dictionary
which removed its normalization entry. The corrected controller uses underscore
names and replaces only the MoE entry. Both attempts restored the baseline.
The first logging wrapper used an unconfigured logger namespace; the corrected
wrapper uses the serving logger and its prompt progress is visible in the log.

The first profiling idle check found an active user request. Its cleanup
unnecessarily reapplied the same candidate; this was acknowledged to the user.
The corrected controller restores only after it has acquired maintenance.
The completed profile restored v1021, stopped profiling, and reopened the
service. The architecture diagram is offline work and does not mutate the
running server.

Live server log:
`/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/packed-state-1280-server-v1006-r5.log`

Trial log:
`/srv/ai/artifacts/glm-scale-layout-20261009/live-queue-r3.log`

[Chinese companion](REPORT.zh.md).
