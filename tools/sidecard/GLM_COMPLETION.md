# GLM CUDA completion contract

The optional GLM completion path registers coordinator-owned CUDA result
buffers with both Ascend workers. It queues the cold-result consumers before
receiving host completion replies. A separate, ordered Ascend DMA writes the
64-bit sequence that releases each CUDA stream wait.

The host still schedules Ascend computation and blocking DMA ioctls. Routing
metadata reaches the CPU for validation; activation and result tensors bypass
host staging. This does not implement an Ascend device dispatcher.

## Runtime selection

Use the existing authenticated JSON runtime configuration:

```json
{
  "direct_dma": {
    "cuda_completion_contract": true,
    "completion_input_record": true,
    "completion_library": "/absolute/path/completion_merge.so",
    "completion_library_sha256": "SHA256_OF_COMPILED_LIBRARY"
  }
}
```

Both switches default to false. The input-record switch requires the completion
contract. The existing `prefill_reference` and `prefill_reference_sha256` fields
must identify a previously qualified complete-model reference. Readiness requires
matching the external reference, including complete-vocabulary boundary hashes.
Keep the existing NPU prefill graph switch disabled on the qualified stack.

Compile and authenticate the native consumer before startup:

```bash
nvcc -shared -Xcompiler -fPIC --cudart shared --fmad=false -arch=sm_120 \
  -o completion_merge.so tools/sidecard/completion_merge.cu
sha256sum completion_merge.so
```

The supplied native library must expose all three consumer/preparation entry
points and the preload entry point. Startup authenticates its bytes and preloads
kernels before any future-value wait can block CUDA initialization.

The native capacity query `sidecard_glm_completion_max_rows(uint32_t separate)`
returns 64 for packed results and 128 for separate result/control buffers.
An older binary without this symbol remains limited to 64 rows. Startup checks
the admitted cold batch against the authenticated binary's advertised capacity
before allocating exports or submitting future waits. A 128-row cold window
requires `completion_input_record: true`: its FP32 result uses all 2 MiB of the
result slot, with the invocation header retained in the separate control slot.
CUDA model chunks remain 32 rows.

## Input-record path

1. CUDA generates a 64-byte invocation record after packing each input. It writes
   the record into the input packet and the coordinator's control allocation,
   with an unready sequence word. A source event completes before admission.
2. Ascend uploads the input and record with one CUDA-to-HBM DMA. Its existing
   routing metadata read also reads the record. The worker checks epoch,
   sequence, rank, slot, layer, rows, byte geometry, and reserved fields.
3. Ascend computes the cold partial and transfers its native FP32 output
   directly to the coordinator-owned CUDA result buffer.
4. Only after result DMA completes, Ascend transfers the sequence from the
   actual uploaded HBM record into the CUDA completion word with a second DMA.
5. The prequeued CUDA wait releases its consumer. The consumer validates the
   control record and adds rank 0, then rank 1 using explicit FP32 rounding.
   The scratch buffer may alias the preceding rank's output.

A main bridge fd retains the input and result mappings; a second fd retains the
control mapping. This respects the current two-mapping-per-fd bridge limit.
SCM_RIGHTS passes dedicated DMA-BUF fds once on authenticated private sockets.
Workers never import or consume the coordinator's GPU result address. Completion
allocations require native RDMA ordering in their owning CUDA context.

With `completion_input_record` false, the first contract version packs a
72-byte header before the FP32 result in HBM, then transfers that packet and its
sequence. It supports the same fixed rank addition order but adds an HBM copy
and host-to-HBM header upload. Retain separate evidence for the two versions.
The packed path retains its 64-row limit.

## Lifetime and error handling

Preallocate allocations, exports, scratch space, and completion events. Reuse
slots only after their sealed consumer events finish. The subsequent admission
flags CPU read already synchronizes the preceding stream; ticket retirement
adds no extra hot-path synchronization.

On shutdown, every worker closes its result/control bridge mappings and duplicate
fds before acknowledging closure. The coordinator then releases its owned CUDA
results. Worker input allocations remain alive until the coordinator closes
its IPC imports and sends the final release acknowledgement.

Uncertain native DMA, a pending CUDA consumer, or failed foreign-map closure
retains the original owners. A timeout cannot authorize freeing a buffer or
killing a worker. The CUDA runtime accepts an ASGI generation thread only when
it holds the existing dispatch lock and uses the original process/context.

## Qualification

Run the bounded producer/consumer test before GLM startup:

```bash
python -m tools.sidecard.completion_probe --bootstrap \
  --config /absolute/path/runtime.json --output /absolute/path/results.json
```

It uses the assigned physical chips 2 and 3, real DMA-BUF registration, direct
input-record upload, ordered result/sequence DMA, and prequeued CUDA consumers.
The cancellation fixture detects reordered FP32 additions. The first case
preflights sequence DMA before any future-value wait.

Then launch the normal `tools.sidecard.layer_server` from an immutable source
snapshot. Run teacher-forced decode, full prefill reference validation, completed
2K/64-output and 8K/8-output API benchmarks, and post-prefill arithmetic checks.
Compare matched uncached prompts and complete generated continuations.

See the [GLM integration measurements](../../artifacts/sidecard-glm-completion-20261013/README.md)
and [design PNG](../../artifacts/sidecard-glm-completion-20261013/design.png).
