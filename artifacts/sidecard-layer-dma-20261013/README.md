# Complete Ascend block over direct DMA

The contiguous layers 18–32 implementation runs end to end with the existing
PCIe bridge on one Ascend Duo (physical chips 2,3) and one RTX PRO 6000.
It measured **9.80 decode tokens/s** on the 2K/64 fixture, compared with
**9.65** for the same block using sockets. All **192 complete vocabulary
vectors** match the socket reference exactly. The configuration remains
experimental: the current expert-only service measured **12.28 tokens/s**
on the same prompt through its API and was restored at **131,072 total tokens**.
The API and model-forward timings are different measurement boundaries;
they do not establish a precise architecture speed ratio.

![Measured contiguous-block design](design.png)

## Implemented execution

CUDA owns embedding, layers 0–17, layers 33–44, and the vocabulary head.
Ascend owns all attention, mHC, routing, shared MLP, and routed experts for
layers 18–32. Attention/shared MLP use TP2; experts use EP2 with 144 resident
experts per chip. Expert order and layer dependencies are unchanged.

Both Ascend chips receive the same complete deferred mHC state directly from
CUDA. Rank 0 returns the complete block result after its last layer. There are
two architecture boundaries, implemented as three DMA copies: input to each
chip and one return. HCCL reductions remain inside the Ascend block. Host
sockets carry authenticated metadata and acknowledgements; this implementation
still launches work from the host. No tensor payload stages through host RAM
in normal inference. Diagnostic fingerprint collection intentionally reads
payloads on CPU and is excluded from the throughput fixture.

CUDA owns dedicated 4 MiB input/result allocations; each worker owns private
HBM and an unused, poisoned ACL host lifetime guard. Each chip uses the existing
UAPI v5 blocking DMA bridge with persistent mappings. Registration checks
socket credentials, checkpoint identity, nonce, rank, and physical device.
Canonical state geometry, monotonic sequence, position, and request reservation
checks precede DMA. Shutdown requires both workers to release native mappings
before CUDA frees its allocations. Uncertain DMA preserves owners.

The active slot admits at most 32 prompt rows or one ordinary decode row.
The old socket implementation can overlap two prompt chunks. This first DMA
block uses one serialized slot, so its prefill measurements include that
scheduling difference. Neither this test nor a larger slot removes KDA's
sequential dependence within a prompt.

## Completed hardware measurements

Both block runs use the same recovered checkpoint, native expert build,
clipping, FP32 recurrent state, BF16-rounded residuals, native Sinkhorn,
native single-row BF16 rounding, and KDA-span replay. CUDA uses the current
DSA, Hadamard, and indexer projection optimizations. There is no prefix reuse.
Each decode fixture produces 64 outputs; the reported forward rate times the
63 subsequent model calls and excludes CPU vocabulary hashing.

| Prompt / repeat | Socket block tok/s | DMA block tok/s | Socket prefill s | DMA prefill s |
| --- | ---: | ---: | ---: | ---: |
| 2K / 0 | 9.5817 | 9.7606 | 27.0489 | 30.5647 |
| 2K / 1 | 9.7123 | 9.8434 | 26.8408 | 30.2690 |
| 8K / 0 | 9.5864 | 9.7522 | 100.3100 | 114.9414 |

The 2K median decode improvement from the transport change is **1.61%**.
The [socket report](socket-v1/stage-benchmark.json),
[DMA report](dma-v1/stage-benchmark.json), and [summary](summary.json)
retain complete output hashes, IDs, prompt identities, and timing samples.
The [current service API fixture](current-service-api.json) completed a cold
2K/64 request with zero cached tokens, 9.9501 seconds to first text, and
12.2823 decode tokens/s. The restored service completed another cold 2K/64 API request with identical
output text, zero reused tokens, 9.9292 seconds to first text, and **12.4416
decode tokens/s** ([receipt](restored-service-api.json)). The earlier
[projection qualification](../sidecard-projection-20261013/README.md)
contains three completed client measurements rather than this single sample.

Rank 0's median DMA decode components were **0.245 ms input transfer,
83.781 ms block execution/packing, and 0.366 ms return transfer**.
Rank 1 measured 0.284 ms input and 83.682 ms execution/packing.
Its final timing bucket covers its host acknowledgement, not a result DMA.
The [worker receipts](dma-v1/stage-dma-rank0.json) include every matched
sequence; these complete-block sequences can be paired across ranks.
Most of the delay remains inside Ascend computation and its internal
collectives, rather than the external PCIe copies.

## Native kernel findings

The installed operator library SHA256 is
`9b61b2069b44638193918de0ba76e546e1003dae535a7c27414a7f4ee0924d73`.
The [BF16 gate](sidecard-block-native-gate.json) tested both assigned chips
at 1, 4, and 32 rows, including residuals above FP16 range. All cases were
bitwise equal. Single-row large-state rounding took about 55 microseconds
versus 237–243 for eager conversion. At 32 rows the native operator was
slower, so the binding uses only large single-row inputs. Small mixer
coefficients and multi-row prefill retain the original conversion.

The existing native mHC post kernel rounds/saturates to FP16, whereas this
runtime preserves BF16 rounding and FP32 range. It was not substituted.
The installed recurrent KDA operator **does accept FP32 state**, despite stale
FP16-only comments in its model wrapper. The
[raw KDA gate](sidecard-block-kda-gate.json) tested one and four rows on both
chips. Inputs require FP16 normalized query/key and beta; the sidecard
reference keeps their preparation in FP32. Observed output differences were
at most 0.00003052 and carry differences at most 0.00038326 on these synthetic
cases. They are finite but not bitwise equal. First-call timings include setup
and are not steady-state kernel benchmarks. This kernel has **not** passed a
real-model accuracy or long-context gate for this sidecard and is unbound.
Adapting its input contract is a concrete next optimization target.

## Validation and limits

- **1,363 portable sidecard unit tests passed**, including state geometry,
  cache admission, replay rejection, range preservation, and mapping-release
  ordering. Every changed-file manual hook passed. Repository-wide
  `bash format.sh ci` retains seven unrelated failures, listed in
  [validation](validation.json) with the [full log](full-ci.log.gz).
- Both ranks passed all-layer native expert repeat checks and 24 real-weight
  attention/shared-MLP comparisons against complete CPU references.
- Three matched 64-vector fixtures passed exact socket/DMA vocabulary and ID
  comparisons. The 2K fixture additionally repeats within each architecture.
- [Final raw transport gate](raw-v2/raw-gate.json): six changing 1/4/32-row
  complete-state round trips matched exactly. Both ranks closed with unchanged
  poisoned guards. This gate uses identity state math, not dummy model serving.
- The live expert-only service was restored and completed its pipeline,
  transport, repeated decode, DSA, prefill, prefix, and real-generation gates.
  Its existing qualified 128K configuration and full tool profile are retained.
- Full-model source is archived per run. Later constructor/cleanup hardening
  was exercised by the final raw gate and unit tests; that hardening and the
  new complete-block server admission option were not replayed as a complete
  model server. Reporting-only changes after the archive are identified in
  the source comparison receipt. No blanket exact-source promotion is claimed.

The block tests completed 2K and 8K prompts. Although their specifications
reserve 128K cache capacity, they did not exercise that full boundary. The
benchmark does not establish general task accuracy or long-context recall.
The existing faster service's 128K qualification is a separate result.
The complete-layer candidate has no cross-request prefix reuse or MTP.

## Reproduction

On idle assigned accelerators, freeze fresh socket and DMA specifications with
the same checkpoint and kernel paths. Use `architecture`:
`cuda-prefix-ascend-middle-cuda-suffix`; select `stage_transport`: `socket`
or `direct-dma`. Optional `npu_native_bf16_round` defaults to false and requires
`npu_ops_library`; the tested specification enables it and `npu_decode_graphs`.
The complete-stage flag is independent of the expert-only DMA protocol.
Do not point an experimental block at the existing expert-only
`transport_reference`: the different backbone placements need separate gates.

```bash
python -m tools.sidecard.stage_benchmark --config /new/socket/runtime.json
python -m tools.sidecard.stage_benchmark --config /new/dma/runtime.json \
  --reference /new/socket/stage-benchmark.json
```

The explicit hardware system test runs the same real-weight pair and checks
native mapping release. It requires exclusive accelerator ownership:

```bash
python -m pytest --confcutdir=tests/e2e/nightly/sidecard \
  tests/e2e/nightly/sidecard --sidecard-stage-specs \
  /new/socket/runtime.json /new/dma/runtime.json
```

For a complete-block server, first run the socket specification through
`tools.sidecard.layer_server --gate-only` and retain its
`decode-repeat-gate.json`. The DMA server requires `stage_transport_reference`
and `stage_transport_reference_sha256` for that same-math socket receipt.
The unchanged startup gates compare all complete boundary/logit fingerprints
before exposing readiness. Disable `prompt_prefix_cache` for this architecture.
The benchmark above is the hardware run completed here; the nightly wrapper
was collected offline and skipped because no exclusive specifications were
supplied. Preserve the other Ascend card and restore the qualified service
after the experiment.
