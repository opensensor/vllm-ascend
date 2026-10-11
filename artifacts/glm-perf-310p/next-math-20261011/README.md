# GLM math and rank ownership candidates

All four queued areas now have opt-in implementations and offline checks.
No NPU device was opened, no serving process was touched, and the two-card W2
and three-card W3 exports were preserved. The host's other 4 TB drive was used
for CPU compilation. These changes are **not hardware-qualified speedups**.

## Implementation and evidence

| Area | Implemented behavior | Offline evidence | Remaining hardware gate |
|---|---|---|---|
| Operand streaming | Two B slots in L1/L0B; prefetch next weights while Cube consumes the current slot. Replace activation and Cube operand fences with dependency events. | Production K/group-loop event-token test; six-kernel 310P-target compilation. | Actual overlap, numerical outputs, resource admission, and prefill/decode time. |
| Router ownership | One local TP owner computes both gate GEMM and top-k. Share byte-preserved INT64 IDs and FP32 weights. Optional prefill transfer overlaps local shared-expert work. | Exact mixed-dtype packets, pending lifetimes, peer bypass, local shared partial and single reduction; six/four-process Gloo. | HCCL ordering and asymmetric graphs; real routing/output parity; transfer cost. |
| Bulk prefill metadata | Cache inverse routes and weights in UB for at most 640 rows and top-k eight. Larger batches and other top-k values retain the original path. | Production C++ control test checks exact output, tails, fallback and zero GM scalar inverse-route reads in cached mode. | Absolute scalar/memory time and cold-prefill latency. |
| Token-owned prefill | Sort each 16-token window by global expert-route rank; project gathered rows and retain original route slots in UB. Stable combine skips invalid/peer slots. No GM routed-output slab. | Full Python ABI/reference gate and production C++ control test, including ignored NaN peer weights; target compilation. | Numerical/latency gate. Weight-reload cost makes it a capacity experiment, not the preferred speed candidate. |
| Head-sharded indexer | Partition Q and head-weight projection plus scoring into 16/8 heads for TP2/TP4. Reduce full FP32 scores after per-head ReLU and before selection. | Loaded projection/reference tests, full-score-versus-local-top-k counterexample, six/four-process score reduction. | Real pool/tail/cache and near-tie selection parity; communication and prefill time. |

The indexer candidate is an alternative to complete fixed-owner indexing.
Prefill head sharding keeps K/gate projections, pool/tail writes, and selection
replicated. Decode uses the original loaded indexer, including its native score
and cast hooks. Global 32-head score normalization is preserved. FP32 partition
reduction changes addition order; CPU score agreement does not establish exact
real-model selection parity. NPU admission requires the permanent AI-Core BF16
conversion bundle, zero indexer RoPE, and at most eight decode sequences.

Both rank experiments are composed into `glm_native_int4` and
`glm_channel_scale_candidate`, with admission before checkpoint/expert I/O.
They require TP2 or TP4 with replicated tokens, DP1, no context/sequence token
partition, no MTP, and no KV-transfer connector. They do not establish TP6
support. Existing PP stages and local attention/KDA heads remain unchanged.
Each rank retains its local routed experts and shared-expert partial, followed
by the existing combined reduction.

## Bytes and work, not latency predictions

The reproducible [memory/work ledger](memory-and-work.json) includes both TP
profiles and source reservations. Compiler/runtime overhead, HCCL, graph pools,
KV/recurrent state, original retained indexer weights/cache, and other live model
buffers require separate accounting.

- Weight streaming adds no UB buffer: L1 grows from 24,576 to 40,960 bytes and
  L0B from 16,384 to 32,768 bytes. L0A and accumulator storage stay single.
  The optional source-row cache adds another 256 UB bytes to the kernel totals.
- Bulk cached prefill requests 200,992 UB bytes, or 201,248 with source-row
  caching compiled. It removes GM scalar inverse-route reads, not all scalar
  instructions or the routed FP16 GM round-trip. The vector metadata load still
  moves bytes, and subsequent UB address reads remain scalar.
- Token down requests 195,328 UB bytes, or 195,584 with source-row caching.
  At 640 rows, new incoming-shape scratch drops from 25,231,360 to 14,827,522
  bytes, saving about 9.92 MiB. This excludes independently retained decode
  arena slots and short-lived sort/padding temporaries.
- Router packets are 60 KiB at 640 tokens. Send plus receive storage is
  180 KiB per rank for TP2 and 300 KiB for TP4, reused across sequential layers.
  There is one result transfer per MoE layer; small decode can regress.
- At 8192 context, 640 query rows, and a 32-million-element head-score budget,
  the conservative extra head-storage/scratch reservation is about 168/178 MiB
  per TP2 stage rank (three/four local indexers), or 156 MiB for TP4's 11
  indexers. A 192 MiB separate reservation admits these examples. A smaller
  score tile adds more collectives; a longer context needs a new admission plan.

The seeded uniform-routing example has 2573 local routes. Bulk down reads an
estimated 576 MiB of packed weights; 16-token ownership reads about 8.16 GiB,
**14.51 times as much**, while avoiding about 40.2 MiB of routed write/read
traffic. The skewed example still doubles weight reads. These are counts for
synthetic routing and exclude scales and hardware caching. They justify retaining
expert-batched down as the recommended speed path and keeping token ownership
for capacity studies. No timing estimate is claimed.

## Compiled bundles

All bundles are append-only, use 64 physical Cube rows and the same G256 channel
weight contract, and retain direct-division activation quantization.

Host root: `/srv/ai/storage/artifacts/glm-offline-next-math-20261011/`.
Local review copies: `/tmp/glm-channel-kernel-bundles/`.

| Bundle | Purpose | Metadata ABI |
|---|---|---|
| `v14-stream-weights` | Operand streaming alone | 3 |
| `v17-bulk-cache-stream` | Streaming plus cached bulk-prefill metadata; first speed trial | 3 |
| `v18-token-prefill-abi4` | Token-owned prefill capacity trial | 4 |
| `v19-token-stream-abi4` | Token ownership plus streaming | 4 |

Each contains six compiled kernels, its namespace-specific bridge, frozen Python
helpers, source snapshots, compiler flags and SHA provenance. Binary and helper
checks passed for all four. Token mode has a distinct metadata ABI: sorted window
rows/positions cannot be passed to a bulk-prefill kernel. Prototype v15/v16
receipts are [archived](archived-abi3-token-prototypes); the current loader rejects
their old token ABI. They are not deployment candidates.

Repository receipts list source paths and SHA values separately and include the
original provenance-file SHA. The runtime bundle's original manifest bytes are
unchanged. This avoids a secret-scanner false positive on a source filename
containing `token` followed by its SHA.

## Validation and next comparison

The 218-test combined CPU gate and two-test nightly CPU process gate are recorded in
[cpu-gates.txt](cpu-gates.txt) and
[nightly-cpu-transport.txt](nightly-cpu-transport.txt).
Direct process receipts are [TP2/PP3](gloo-tp2-pp3) and [TP4](gloo-tp4): nonzero
local owners, stage isolation, rows 1/4/37/640, two changed-input steps, exact
router bytes and full-score reductions. Synthetic argmax agreement is 100%;
FP32 partial-sum differences are recorded, not hidden.

All changed-file manual hooks pass. Full repository formatting was attempted;
existing unrelated lint/spelling/import failures remain. Its automatic edits to
144 unrelated files were restored in the isolated worktree. Source checksums,
results and limitations are recorded in [validation.json](validation.json).

Use [the runbook](RUNBOOK.md) for options and gates. First compare v14 with v17
at unchanged weights and launch geometry. Test router ownership and head sharding
independently before combinations. Retain the qualified service image until
actual operator dispatch, changed-input graph replay, real-model parity, peak
memory, cold time to first token and steady decode latency pass. No NPU tests
or serving benchmarks were run during this offline delivery.
