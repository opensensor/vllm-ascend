# M3.1 larger context and prefill batching

This experiment retains the signed-W4 weights, TP2/PP3 partition, stage-local
expert parallelism, FP32 sparse scoring and 64-query attention tiles. It raises
prefill chunks from 128 to 512 tokens and grouped route capacity from 512 to
2048. A decoded expert weight tile can serve more tokens before being discarded.
The grouped kernel tiles its rows internally; per-core dequantization workspace
depends on the projection width, not the route count.

## Component evidence

`m31-native-v7-projections.json` covers 30 full-sized real expert projections,
including distributed and concentrated 2048-route calls. All passed independent
CPU references (maximum normalized RMS error below 1.83e-5); native and OPP
results were bit-identical. The versioned v7 build retains v6's pipelined wide-L1
schedule and extends its Python shape contract and load-time validation.
The standalone projection checks used chip 5 while public inference was drained.

Short coherence suites retain complete summary reports; long-context requests
and tool protocol checks retain their detailed receipts.

CPU tests cover EP sums, chunk boundaries, invalid capacities and lossless NZ
loading. `unit-tests.txt` records 163 passed and 24 conditional skips; the 11 new HCCL configuration tests
passed in the real NPU environment. The three warmup tests and existing
non-M3 explicit-budget regression also passed there. Direct collection of that
worker test hit an existing circular import; preloading `vllm_ascend.ops`, as
runtime startup does, resolved collection without changing its code.

## Profiling and baseline

The temporary profiler records nested component CPU durations and NPU events.
Whole-stage decode times were about 0.16 seconds across three pipeline stages.
Some short event intervals were negative on this CANN build; do not treat those
event sums as a precise attribution or add nested component durations together.
The probes indicated that expert projections, attention and collectives all
contribute to decode cost. Profiling was removed before the timed baseline.

The fresh 128-token baseline returned correct answers in 7.34 seconds for the
greeting, 13.64 seconds for France, and 61.01 seconds for 3443-token retrieval.
The previous cold 16043-token retrieval took 299.90 seconds; its original receipt
is in `../minimax-m31-tools-20261025/final-16k/`.

## Serving validation

The 48K/512-token service passed all nine coherence checks and all ten tool and
structured-output checks. Cold retrieval passed at 16043 tokens in 158.76 seconds
and 48603 tokens in 490.13 seconds. The 3443-token check took 37.90 seconds.
`run-gate.py` clears scheduler and physical caches before each long-context
request. All six worker PIDs and weight-storage fingerprints stayed unchanged.

The user requested more capacity, then explicitly delegated the full 160K
long-context test to their Kilo session. The subsequent 160K profile uses a
4,410,900,480-byte explicit per-rank cache budget, including spare cache blocks,
and 32 MiB ordinary HCCL buffers. Startup allocated 164,096 cache slots and
passed greeting, red-image, 3443-token retrieval and streamed-tool checks.
All six worker PIDs, weight fingerprints and native generation remained unchanged.
`short-160k/report.json` explicitly records that full-context testing was deferred.
Kilo now advertises 163,840 context/input tokens with 2048 output tokens reserved;
its total/stream-chunk timeouts are 90/60 minutes for slow cold prefills.
The earlier exact allocation was one reserved cache block short. A second
attempt allocated the cache but failed on first inference: lazy HCCL managed
workspace allocation could not obtain a 2 MiB region after the cache allocation.
The targeted M3 warmup profiles TP operations before allocating the explicit
cache; the existing custom pipeline transport already warms its own channels. Compressed raw failure logs
retain the original diagnostics, including a rejected direct P2P warmup that
used an unsupported HcclSend path instead of the existing pipeline transport.
Other models keep their explicit-budget path. The 128K × 16
baseline remains infeasible on these memory banks. No broad language benchmark
or original-FP4 parity is claimed.

The service remains eager, one sequence, one 448-pixel image, with native tool
parsing and the current-turn reasoning parser. ACLGraph, sequence-parallel MoE,
video and speculative decoding remain unsupported in this adapter.

## Resident scale-cache benchmark

The v8 candidate caches four quantizer scale rows across wide-L1 K tiles.
`m31-native-v8-resident-benchmark.json` records 15 projection/routing cases on
each of six real resident expert banks (90 comparisons). Every result was
bit-identical to v7. Five timed iterations followed two warmups per kernel.
Across-rank median speedups were approximately 1.004–1.021x for four-route
cases and 1.003–1.007x for balanced prefill; concentrated 2048-route cases were
approximately unchanged or slightly slower (0.993–0.999x).
These small differences do not establish a full-model improvement. V8 remains
an unpromoted microbenchmark candidate, and v7 remains the serving dispatch.

The benchmark drained active inference while queuing new requests, preserved
scheduler prefixes and physical KV storage, and restored the exact baseline
generation. All six worker PIDs and weight fingerprints stayed unchanged.
Reported device free memory ranged from 0.67 to 2.07 GiB after the bounded
benchmark. This is a tight experimental 160K allocation, not proof of peak
memory stability at full context. `v8-summary.json` contains the reduced timings.

## Operation

`serve.sh` and `activate-native.sh` are the direct launcher and systemd post-start
activation script. Context/scheduler allocation changes require one startup;
subsequent compatible Python/CANN experiments use resident transactions.
The old launcher and activation script are retained for rollback.

Repository-wide `format.sh ci` was run at the unchanged baseline in an isolated
worktree; its existing unrelated failures are summarized in
`format-ci-summary.txt`. Changed files receive their own hook checks.
