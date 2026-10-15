# W4 prefill resident trial results

The v3 queue-drained fused finalizer and local-count schedule are active;
the original v1 schedule and v1/v2 finalizers remain rejected. The persistent service remains
`qwen38-flash-next` at `http://192.168.53.187:8001/v1`, with corrected QSA,
compact top-k, draft partition, qLens capacity and the reset barrier.
Worker PIDs 579046–579049 and tracked weight storage remained unchanged.

## Component qualification

Both projections at 650 and 20480 allocated route rows passed exact FP16 bit
comparison against the installed operator on all four ranks. Three native
schedules covered all-peer, balanced EP4, all-local and expert-71 concentration.
These tests borrow the first actual resident expert bank on each rank; they do
not establish model-wide quality for every expert/input.

Rank-0 balanced 5120-local/20480-allocated rows:

| Component | Installed | Device-local-count |
| --- | ---: | ---: |
| Gate/up projection | 26.054 ms | 19.256 ms |
| Down projection | 12.510 ms | 10.727 ms |

The concentrated down-projection case regressed from 7.217 to 8.294 ms. Local
row count alone does not describe the expert distribution.

Fused finalization at 650 and 20480 rows passed exact weighted FP32 bits and
unchanged route-sum bits, including peer zeros, nonzero routes, signed zero,
subnormals and large finite values. At 20480 rows, per-rank median component
times were 6.09–6.16 ms for the existing sequence and 2.07–2.11 ms fused.
These are synchronized component measurements, not serving gains.

## Unprofiled model timings

Every request used 8192 prompt tokens, a unique early nonce, zero cached tokens,
and one output token. Each row is one sequential trial on a shared service;
these are not statistical throughput guarantees.

| Payload | HTTP seconds | Prompt tokens/s |
| --- | ---: | ---: |
| Compact-top-k control A | 23.032 | 355.68 |
| Device-local-count schedule only | 29.818 | 274.73 |
| Fused finalizer only | 31.698 | 258.44 |
| Restored compact-top-k control B | 23.163 | 353.67 |
| Shared-owner v2 finalizer | 31.371 | 261.13 |
| Queue-drained v3 finalizer A | 22.311 | 367.17 |
| Queue-drained v3 finalizer B | 22.218 | 368.71 |
| Queue-drained v3 finalizer C | 22.456 | 364.80 |
| Fresh compact-top-k control C | 23.158 | 353.74 |

The comparable before/after controls confirm that neither original component
win carried through the v1 integration. Those changes were removed using the resident hot
reload path. There was no server restart, model rename, port change, device
reset, or unsafe peer read.

Receipts are under `qsa-resident-v1/`: `w4-schedule-*.json`,
`w4-finalize-*.json`, `cold-benchmark-w4-*.json`, `install-w4*.json`, and
`restore-after-w4*.json`.

## Warning and failure handling fixes

The private timing setup previously inverted an INT64 permutation on NPU,
producing the ArgSort AiCPU warnings at 07:42:15 UTC. Setup now generates and
inverts permutations on CPU before transferring them. A regression rejects
integer NPU argsort; serving routing already sorts exact floating keys.

Validation now always includes all-peer, mixed and all-local cases even when
one repeat is requested. Native-action failures retain their tensor-bearing
tracebacks; warm-up outputs survive synchronization. Both model bindings now
fall back when an unseen shape would exceed descriptor capacity or allocate
during capture. Finalizer full-chunk and tail admission is checked together.

## Follow-up evidence

A bounded finalizer-only native profile recorded 6847 uncached prompt tokens
and 16 output tokens in 28.46 instrumented seconds. All four profiler sessions
started/stopped once and released ownership. The baseline serving payload was
restored afterward. Profile timing is not directly comparable to unprofiled
HTTP rates. Raw traces: `qsa-resident-v1/native-timeline-w4-finalize-v1`.

The trace establishes a concrete allocation/wait regression: physical allocation
calls rose from 92 to 1627, while device computation decreased by about 1.10
seconds. Peer transfer time stayed nearly unchanged, but peers spent much longer
waiting. See [the measured comparison](qsa-resident-v1/native-timeline-w4-finalize-v1/cold-prefill/ALLOCATOR-REGRESSION.md).

Copied native submission callbacks each retain their own tensor vector. Clearing
one callback does not clear sibling copies. Shared argument ownership was
qualified in a separately loaded v2 bridge; that change alone did not fix the
regression.
Descriptors are cached by shape/device and shared across layers; their size does
not explain gigabytes of reservation changes. These hypotheses require timeline
evidence before any bridge or memory-policy change. No new DMA allocation
experiment is active, and the failed empty private owner remains retained.

## Shared-owner v2 gate

The new bridge compiled in a separate namespace; both device binaries are
byte-identical to v1. Exact finalization gates at 650/20480 rows passed on every
rank, as did component timing with CPU permutation setup. The unprofiled model
trial remained slow at 31.371 seconds, so v2 was removed and compact-top-k
restored with the same PIDs. This does not support callback-copy retention as
the sole explanation. Legitimate queued submission ownership and host-ahead
allocation pressure remain to be investigated; no v2 serving gain is claimed.

## Queue-drained v3 retained

V3 keeps the same device binaries and shared argument ownership, and drains
pending host submissions on the producer thread after each native finalizer
launch. It preserves the original implementation during graph capture. This
is a host queue drain, not an explicit device synchronization; queued operations
can still wait internally. See the [runtime audit](qsa-resident-v1/native-timeline-w4-finalize-v1/cold-prefill/QUEUE-DRAIN-AUDIT.md).

Exact finalizer checks at 650 and 20480 rows passed on all four ranks. All 159
targeted CPU tests passed, including the integer-sort regression. Eight model smoke checks covered short prompts,
crossing the QSA boundary, 8K input, concurrent requests, and returning to short
input. They establish bounded integration behavior, not full-context quality.

Three uncached 8192-token requests measured 364.80–368.71 prompt tok/s,
median 367.17. A fresh baseline measured 353.74: a 3.8% throughput improvement
in this sequential experiment, not a decode improvement or statistical guarantee.
The v3 payload was retained through resident reload, with unchanged workers and
weight storage. A final chat smoke succeeded; inference was unpaused and all
NPUs healthy at 74–79°C. The endpoint and Kilo entry are unchanged.

Receipts: `qsa-resident-v1/w4-queue-*.json`, `validation-w4-queue.json`,
`cold-benchmark-w4-queue-*.json`, and `promote-w4-queue-finalize.json`.
The separate build receipt `w4-queue-route-compile-v3.json` describes build-time
state; runtime qualification is recorded separately. A bounded native trace
was collected under `qsa-resident-v1/native-timeline-w4-queue-v3`; all four
profiler owners released cleanly before the final payload was restored.

Rank-0 offline analysis confirms that the allocation/wait regression subsided:

| Metric | V1 finalizer | V3 queue-drained finalizer |
| --- | ---: | ---: |
| Physical allocation calls | 1627 | 98 |
| Physical free calls | 1482 | 0 recorded |
| Device-wide synchronization calls | 143 | 1 |
| Collective transit | 1.049 s | 1.038 s |
| Collective waiting | 7.555 s | 1.656 s |
| Device computation | 15.969 s | 16.062 s |
| Kernel-free gaps | 2.939 s | 0.673 s |

These bounded traces use 6847 versus 6852 prompt tokens and 16 generated tokens;
they diagnose the regression rather than establish equal-work throughput.
API durations overlap and cannot be summed. The v3 result supports host
submission backlog/operand lifetime as the integration problem, while v2 alone
did not isolate a useful improvement. Payload transit is effectively unchanged.
Hashed source evidence: [v3 allocation summary](qsa-resident-v1/native-timeline-w4-queue-v3/rank0-allocation-summary.json).

## V3 schedule subsequently retained

After qualifying both projections through the queue-draining bridge, the
local-count schedule was added alongside the finalizer. Three uncached 8K
requests reached median 393.55 prompt tok/s; all eight model smoke requests
passed. The model binding now enforces eager-only v3 execution even for cached
descriptors. See [complete schedule results](w4-queue-schedule-v3.md).
