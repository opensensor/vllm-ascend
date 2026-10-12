# CUDA backbone with Ascend cold experts

This runtime serves GLM-5.3-Flash using one RTX PRO 6000 Blackwell and one
Atlas 300I Duo. Its two assigned chips are physical devices **2 and 3**;
physical 0 and 1 remain available for other work.

The [asynchronous PCIe bus library](pcie/README.md) is separately hardware-tested.
It uses CUDA IPC and native per-slot DMA events; the endpoint described here
continues to use the qualified socket transport pending full-model integration.

CUDA owns all 45 backbone layers, attention and indexer caches, recurrent
state, mHC, routing, shared MLPs, embedding, head, and sampling. CUDA also
retains all 288 experts in layers 3–17 and 33–44. In layers 18–32, it retains
22 hot experts selected independently per layer from prior prompt routing.
The placement is static, sorted by global expert ID, and authenticated with
the checkpoint identity and its own SHA-256.

Ascend stores the remaining expert capacity in two independent resident
banks of 144 experts per layer. The initial implementation also retains the
GPU hot experts in those banks, but assigns them zero cold weight. Each
selected route executes once, with its original normalized weight. There
is no HCCL group, cross-vendor tensor parallelism, or weight paging.

For each middle-layer call, the coordinator copies FP16 activations,
INT32 global IDs, and FP32 cold weights into pinned host buffers. It sends
the same payload to each chip with positive work, then launches GPU hot
experts and the shared MLP before receiving either reply. Each chip returns
one weighted FP32 partial per row. The coordinator merges GPU, rank 0,
then rank 1, adds shared output, and converts to the layer dtype.
The private protocol authenticates readiness and checks the sequence,
layer, and row count on every response. A chip with no work receives no call.

CUDA captures KDA spans outside the middle block. In the middle block,
only KDA attention, mHC, and routing are captured; socket calls remain
outside capture. DSA length/indexer updates stay eager. Prefill uses bounded
32-row chunks, with overlap inside each expert call. Requests serialize and
reset all caches. Prefix caching, MTP, batching, vision, tool calling, and
nongreedy sampling remain unsupported.

## Reproduce

Use the isolated CUDA Python and the previously qualified clipped CUDA
and Ascend kernels. Preserve the private NPU compiler environment; do not
copy its contents into Git. The hardware report records executed snapshot
hashes, placement, validation, and measurements:
[expert runtime report](../../artifacts/sidecard-expert-runtime-20261012/README.md).

A specification inherits the existing authenticated layer-runtime fields
and adds:

```json
{
  "architecture": "cuda-backbone-ascend-experts",
  "expert_placement": "/absolute/path/to/placement.json",
  "expert_placement_sha256": "aaa70408a6046e4e7fafaf62173d95476617ae9d3a9ed3e01c31f760872caad1",
  "cuda_decode_graphs": true,
  "max_context": 65536
}
```

All other checkpoint/backend paths and identities must match the real
qualified deployment. This fragment alone is not a complete specification.
`layer_runtime` validates and freezes the placement alongside the source:

```bash
python3 -m tools.sidecard.layer_runtime \
  --output /srv/ai/storage/artifacts/your-new-expert-run \
  --specification /absolute/path/to/specification.json \
  --cuda-python /srv/ai/venvs/sidecard-cuda-20261012/bin/python \
  --source /absolute/path/to/clean-qualified-source \
  --port 8000
```

Use a fresh output directory. Checkpoint and native kernel authentication
must pass. Use only the four qualified `tools/glm_perf` helper modules
listed by the source snapshot, alongside the sidecard code. Readiness waits
for both resident workers, local native repeat gates, full-vocabulary
teacher-forced repeatability, and real-weight startup inference.

Run the API validation and fixed fixture:

```bash
python3 -m tools.sidecard.layer_validate \
  --base-url http://127.0.0.1:8000 \
  --output /absolute/path/to/api-gate.json --lengths 4096 16384
python3 -m tools.sidecard.decode_benchmark \
  --base-url http://127.0.0.1:8000 \
  --output /absolute/path/to/decode2048.json
```

The endpoint is `http://192.168.53.187:8000/v1`, model `glm-sidecard`.
Use `reasoning_effort: "low"` for direct arithmetic smoke tests.
The configured context limit and completed capacity tests are separate
claims; consult the report before relying on the largest context.
API details and scoped stop behavior remain in the
[layer runbook](LAYER_RUNTIME.md).

## Latency attribution

Use the [complete-model profile report](../../artifacts/sidecard-expert-profile-20261012/README.md)
for measured host/device delays, communication, and the bounded diagnostic
procedure. Its opt-in `expert_worker_profile` specification flag defaults
to false. Profiling preserves vocabulary bytes but adds observation cost;
the exact uninstrumented serving snapshot is restored after the diagnostic.
