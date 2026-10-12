# Faster GLM over direct CUDA ↔ Ascend DMA

The service is running at `http://192.168.53.187:8000/v1`, model
`glm-sidecard`, with one RTX PRO 6000 and physical Ascend chips 2 and 3 of
one 310P card. The other Ascend card was not accessed. Clocks, ECC, power,
topology, and vendor drivers were not changed.

The final configuration passes full-vocabulary bitwise parity against the
qualified socket reference and three repeated 64-step traces. Its median
decode rate is **10.70 tokens/s**, up **24.84%** from the contemporaneous
starting DMA median of **8.57 tokens/s**. It is approximately **11.03%** above
the saved socket result of 9.64 tokens/s; that socket result was not rerun
during this session.

## Completed service measurements

All decode comparisons use the same exact 2,048-token prompt, 64 output tokens,
greedy generation with EOS ignored, and zero reused prefix tokens. Decode
excludes time to first text. Each new row contains three completed requests;
these are a small repeated sample, not a statistical confidence study.

| Configuration | Decode runs, tokens/s | Median |
| --- | --- | --- |
| Starting direct DMA | 8.573, 8.300, 8.634 | 8.573 |
| Scalar metadata validation + CUDA packing replay | 8.958, 9.022, 9.151 | 9.022 |
| Also Ascend expert replay | 9.738, 9.817, 9.889 | 9.817 |
| Also imported-alias ordering, GPU mapping reuse, fewer receipt writes | 10.703, 10.692, 10.803 | 10.703 |

The final changes were measured together; this does not isolate the contribution
of each change. See [summary.json](summary.json) and the `before-*`, `after-*`,
`npu-graph-after-*`, and `final-after-*` receipts.

An 8,192-token prompt plus eight output tokens completed at 10.52 decode
tokens/s, with 96.30 seconds to first text. This is a context smoke test, not
a sustained long decode benchmark. The prior saved 8K run took 87.73 seconds
to first text, so this session does **not** establish an 8K prefill improvement.
The advertised limit remains 65,536; a complete prompt at that limit and
general long-context quality remain unqualified.

## Implemented changes

- CUDA graphs prepare one-row routes and both private GPU input slots. Commands
  still go only to active ranks. Native hot/shared work overlaps both workers.
- Ascend graphs replay the original one-row expert calculation, including its
  zero padding and deterministic fixed-slot reducer. Larger batches stay eager.
- Kernel UAPI version 5 retains up to two GPU DMA-BUF mappings per device fd.
  Identity uses the exporter object and physical chip, rather than numeric fd.
  Python retains original GPU allocation leases until clean fd close. HBM
  lifetime descriptors and address queries remain per copy.
- The coordinator checks and enables `SYNC_MEMOPS` on every imported CUDA
  alias before use. The original exports had the flag enabled, but all four
  imported aliases reported it disabled. A separate probe confirms setting
  and reading back the imported flag. NVIDIA documents CUDA/RDMA consistency
  in its [GPUDirect RDMA guide](https://docs.nvidia.com/cuda/gpudirect-rdma/).
- Routing validation uses a bounded scalar CPU loop. Activation rows and
  expert outputs still never bounce through host DRAM.
- Running receipts update every 1,024 calls instead of 64. Serializing an
  existing 4,096-record receipt measured approximately 9.8 ms on this host.
  Final close and quarantine receipts remain immediate.
- FastAPI lifespan cleanup releases worker IPC ownership before Uvicorn replays
  termination signals. The launcher preserves a virtual environment's Python
  symlink instead of resolving it to the base interpreter.

Source events, native completion checks, returned-write flushes, authenticated
ownership, and fixed CUDA → rank 0 → rank 1 reduction order are retained.
The [design PNG](design.png) shows the physical owners and data/control paths.
The [runbook](../../tools/sidecard/GLM_DIRECT_DMA.md) documents the configuration.

## Hardware and software validation

- Three real-weight expert gates each pass 64 cases spanning hot, cold, mixed,
  both ranks, and 1/4/17/32 rows. Repeated outputs are bitwise equal; CPU oracle
  comparisons meet the existing tolerance. Workers close with unchanged guards.
- Raw transient and cached diagnostics each pass 840 payload/red-zone copies.
  The final module passes another 240 copies and duplicated/reused-fd identity
  and capacity checks. Decode-sized cached transfer medians are roughly
  0.10–0.12 ms upload and 0.09–0.11 ms return, versus approximately 0.24–0.26
  and 0.21–0.26 ms in the corresponding transient diagnostic.
- Full 154,880-logit fingerprints match the socket reference at prefill and
  every step of the 64-token trace; three repeated traces match.
- All six final API cases pass: concurrency isolation, streaming equivalence,
  cancellation during prefill and decode followed by recovery, large output
  budget admission, and over-context rejection. The 8K request completes.
- **737 unit tests pass.** Scoped checks pass. A fresh isolated `format.sh ci`
  still fails the same seven pre-existing hooks as the committed baseline:
  Ruff check/format, codespell, typos, clang-format, markdownlint, and forbidden
  imports. Repository-wide CI is not reported as passing.

Worker timings are host wall spans and overlap CUDA and the other rank; they
must not be added into a disjoint model latency waterfall. Their bounded
sample medians and completed-transfer counters are in `summary.json`.
Serving checks completion identities and geometry, not a payload checksum on
every transfer. Payload correctness is established separately by the gates.

## Rejected candidates and shutdown recovery

Earlier cached candidates without imported-alias ordering failed the full-model
bitwise gate and stayed offline. One also attempted to reuse a Torch NPU
completion event; the delivered code creates a fresh event per invocation.
The final candidate changes ordering, graph execution, and receipt cadence
together, so these results do not assign every earlier mismatch to one cause.

Stopping the previous server exposed the Uvicorn signal replay bug: its workers
parked after the importer disappeared without a release acknowledgement.
Their receipts established completed DMA with no uncertain transfer, and the
importer was confirmed dead before a scoped administrative reap. No device
reset or forced module removal occurred. The patched V6 and V7 servers then
closed normally, preserved guard bytes, and left zero module references.

Automatic recovery from uncertain DMA remains unqualified. Such a failure
retains mappings, descriptors, and original owners. The ioctl remains blocking;
asynchronous completion queues and multiple in-flight slots are future work.
This qualification applies to this host and its recorded kernel/driver stack.

## Reproduction evidence

`runtime.json`, `process.json`, `source-sha256.json`, `kernel.json`, and
`executed-source.tar.gz` identify the actual execution. Private SDK environment
contents are excluded. All model and transport source files match delivery;
the diagram generator gained presentation labels after execution freeze, as
recorded in `execution-source-parity.json`.

The final runtime is `/srv/ai/storage/artifacts/sidecard-glm-direct-runtime-v8`.
All three flags in `direct_dma` are enabled. The worker receipts record exactly
two GPU mapping creations per rank and reuse on later transfers.
