# Complete-layer GLM sidecard runtime

This experimental text server runs all 45 main decoder layers with real,
authenticated native W2/W3/W4 expert weights and A4 activations. CUDA owns
embedding, layers 0–17, layers 33–44, final normalization/head, and greedy
sampling. One Ascend 300I Duo card owns layers 18–32: attention and shared MLP
use TP2; routed experts use EP2 with 144 experts per chip. All 288 experts
remain available in every routed layer. Physical NPU devices 2 and 3 are the
only assigned Ascend devices.

The server is independent of vLLM's scheduler. A single heterogeneous
`vllm serve` command cannot create these separate CUDA and NPU actors.
Use the custom entry point below. The [hardware report](../../artifacts/sidecard-layer-runtime-20261012/README.md)
records the original tested scope and timings. The
[optimization report](../../artifacts/sidecard-layer-optimization-20261012/README.md)
records projection layout and graph/stream experiments. The
[complete-block DMA experiment](../../artifacts/sidecard-layer-dma-20261013/README.md)
adds direct transfers and the current CUDA/native rounding optimizations.
The active, faster [expert-only service](../../artifacts/sidecard-projection-20261013/README.md)
uses the CUDA backbone and has a qualified 128K context. The complete-layer
implementation remains an experimental alternative.

The [cold-expert study](../../artifacts/sidecard-cold-expert-study-20261012/README.md)
retains synchronized component attribution and a real-weight expert-only probe.
An optional `stage_profile: true` specification field enables per-rank JSONL
traces in a private diagnostic runtime; NPU decode graphs must be disabled.
This deliberately synchronizes each component and records selected expert IDs
and original route weights. Use `tools.sidecard.layer_profile_run` against a
qualified numerical reference and arrange service restoration before starting.
Profiled timing is not production throughput. Use the current expert-only
service for the qualified deployment; use fresh isolated output directories
and exclusive accelerator ownership for complete-layer experiments.

## Execution and arithmetic

There are two cross-architecture boundaries per chunk/token. Each transports
hidden state FP16, four residual streams FP32, deferred post/comb mixers FP32,
and positions INT64: 73,816 bytes per token. Residual/mixer arithmetic rounds
to BF16 precision while retaining FP32 storage and range on the 310P.
Final residual contraction and RMSNorm stay FP32 until the normalized output
converts to FP16. Ascend formats owner-local linear weights as FRACTAL_NZ once at preparation,
avoiding repeated decode weight conversion. Absorbed DSA KV and indexer
operands retain their existing layout. Dense projections use FP16; routing and weighted expert
reduction use FP32. Both Torch backends enforce deterministic algorithms;
CUDA disables reduced-precision GEMM reduction and uses its configured
deterministic cuBLAS workspace. Expert codes and permanent FP16 scales are unchanged.
Both architectures apply the checkpoint's asymmetric SwiGLU clipping:
gate at most 10, up in [-10, 10].

With socket transport, known prompt chunks use a bounded two-slot pipeline: prepare CUDA prefix B
while Ascend processes A; send B before CUDA suffix A. Prefix and suffix
share one RTX and execute in order. Autoregressive decode is serial.
With `stage_transport: direct-dma`, both chips receive complete state directly
from CUDA and rank zero returns it using the existing PCIe bridge. This
serialized implementation uses coordinator-owned 4 MiB slots and admits
32 prefill rows. It does not overlap prompt chunks. Host sockets carry metadata
and acknowledgements; payloads stay on devices. Native mapping release must
complete on both workers before CUDA frees a slot.

KV, KDA recurrent/causal-convolution state, and sparse pool/tail state stay
with each layer. Sparse DSA uses learned causal pooled selection after 2,048
tokens. A new request clears all stage state; there is no cross-request
prefix reuse. Requests run one at a time.

Optional `cuda_decode_graphs` and `npu_decode_graphs` specification fields
capture contiguous KDA spans for single-token execution. Ascend DSA stays
eager. CUDA can also enable the qualified `direct_dma.cuda_dsa_graphs`,
`cuda_indexer_hadamard`, and `cuda_indexer_projection_graphs` options; cache
cursor updates remain outside captured projections. Multi-token prefill stays
eager. Optional `npu_native_bf16_round` uses the existing operator library only
for large single-row state conversion, preserving BF16 rounding and FP32 range.
Capture warms each span before recording, then resets recurrent/convolution
state. Every replay copies all deferred mHC inputs into owner-private buffers.
CUDA capture requires `cuda_backend.torch_stream_experts: true`: native
expert copies, projections and reduction use the current Torch stream without
per-layer device synchronization. That flag requires the matching authenticated
stream-aware CUDA library. A graph candidate must pass the full startup
repeat gate and a completed API benchmark before promotion.

HCCL remains an Ascend-only collective. Its raw Broadcast was a no-op on
this 310P stack; the runtime qualifies byte AllGather and FP32 SUM before
loading. Socket transport uses byte AllGather for boundary state; DMA sends
the input to each chip directly. Idle workers block on private CPU sockets.
Legacy socket children install parent-death signals. Privileged DMA workers
preserve owners after uncertain transfers or unacknowledged mapping release.
Shutdown interrupts prefill between chunks and closes only owned processes.

## Qualified environment

The validation host is `192.168.53.187`: RTX PRO 6000 Blackwell 96 GiB,
Ascend 310P3 dual-chip card, driver 26.0.rc1, CANN 9.1.0.

| Process | Isolated environment |
| --- | --- |
| CUDA coordinator | `/srv/ai/venvs/sidecard-cuda-20261012/bin/python`; Torch 2.14.1+cu130, Triton 3.8.0 |
| NPU actors | `/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python`; Torch/torch-npu 2.13, CANN 9.1.0 |
| Tokenizer | Transformers 5.17.0, matching the existing NPU environment; no upgrade |
| CUDA build/runtime | `/usr/local/cuda`, CUDA 13.4.92; SM120 |

CUDA uses the
separate `cuda_environment` JSON referenced by the runtime specification.
The [retained CUDA configuration](../../artifacts/sidecard-layer-runtime-20261012/cuda-environment.json)
contains only the deterministic cuBLAS workspace setting. The NPU loader
reads checkpoint bytes into private anonymous buffers and authenticates
again after loading, avoiding observed file-cache corruption from mapped
weights on this stack.

The NPU environment file is private host configuration. Reference its path;
do not copy its contents into repository artifacts. CUDA's source path is
isolated from NPU packages. The launcher snapshots source and records SHA256
for reproduction, preserving the saved NPU compiler paths inside its actors.

## Build the clipped routed kernels

The original native bundle's source is archived with its original provenance.
The builder authenticates those exact bytes, inserts only the clipping
operations, recompiles generic/W4 gate stages, and leaves pack/down/reduction
binaries unchanged. It also builds the clipped compact CUDA library.
Neither the checkpoint nor its frozen bundle is modified.

Run from a host checkout containing this implementation:

```bash
mkdir /tmp/sidecard-native-source
tar -xzf artifacts/sidecard-layer-runtime-20261012/frozen-native-sources.tar.gz \
  -C /tmp/sidecard-native-source

/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
  -m tools.sidecard.layer_kernel_build \
  --output /srv/ai/storage/artifacts/sidecard-clipped-kernels-new \
  --bundle /srv/ai/storage/models/GLM-5.3-Flash-one-card-sidecard-candidate/native-kernels \
  --native-source /tmp/sidecard-native-source \
  --environment-file /srv/ai/storage/artifacts/sidecard-glm-service-20261011-one-card-v19/runtime-environment.json \
  --cann /usr/local/Ascend/cann-9.1.0 \
  --cuda-source tools/sidecard
```

Use the qualified CANN environment when invoking the NPU Python. The exact
build receipt and commands are [retained](../../artifacts/sidecard-layer-runtime-20261012/clipped-kernel-build.json).
Run `layer_expert_gate` on CUDA and both assigned NPU devices before changing
kernel precision or clipping. The startup expert repeat gate tests every
middle-block layer and both all-routes-on-one-rank cases.

## Launch

The [runtime specification](../../artifacts/sidecard-layer-optimization-20261012/runtime-specification.json)
references the host's checkpoint, private NPU environment, two Python
interpreters, native ops, and clipped kernel receipt. For a new build, update
`native_gate_override`, CUDA `library` and `library_sha256` together using its
`build.json`. Keep `routed_swiglu_limit` at 10.0. Select a free loopback HCCL
rendezvous port. Output must be new and outside the checkout.

```bash
/srv/ai/venvs/sidecard-cuda-20261012/bin/python \
  -m tools.sidecard.layer_runtime \
  --specification artifacts/sidecard-layer-optimization-20261012/runtime-specification.json \
  --cuda-python /srv/ai/venvs/sidecard-cuda-20261012/bin/python \
  --output /srv/ai/storage/artifacts/sidecard-layer-new \
  --port 8000
```

The launcher replaces itself with the server and records its exact PID and
command. Redirect stdout/stderr to your chosen log for background operation.
Readiness is exposed only after native repeat/TP2 gates, three full
boundary/vocabulary repeat traces, a complete-model
serial/pipeline logit comparison, and a real generated request.

```bash
curl -fsS http://192.168.53.187:8000/health
curl -fsS http://192.168.53.187:8000/v1/models
curl -fsS http://192.168.53.187:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-sidecard","messages":[{"role":"user","content":"Calculate 7 * 8. Reply with only the number."}],"reasoning_effort":"low","temperature":0,"max_tokens":64}'
```

Stop only the recorded parent; its cleanup owns both child actors:

```bash
python -m tools.sidecard.layer_runtime \
  --output /srv/ai/storage/artifacts/sidecard-layer-new --stop
```

## API and validation limits

`/v1/chat/completions`, `/v1/completions`, `/tokenize`, `/v1/models`, and
`/health` are implemented. Completions accept text or one list of token IDs.
Chat accepts system/user/assistant text messages. Use `temperature: 0`, `n: 1`;
`max_tokens` and `max_completion_tokens` reserve output within the total
context budget. Streaming supports `[DONE]` and optional final usage.
Chat separates `reasoning_content` from `content`, including terminators
split across chunks. `reasoning_effort` accepts low/high/max. Benchmark-only
`ignore_eos: true` completes a fixed token count. The shared server also supports text function-tool messages and streaming
through its current tool adapter; these block experiments did not benchmark
tool use. Custom stop, logprobs, response formats, images, and non-greedy
sampling are rejected.

| Feature | Status in this runtime |
| --- | --- |
| Complete real-weight text model | Implemented and hardware gated |
| TP2 middle attention/shared MLP and EP2 experts | Implemented and hardware gated |
| Greedy repeated request | Bitwise native expert and complete output checks |
| Sparse DSA beyond 2K | Implemented; independent changed-input selection gates on both devices |
| Two-slot prefill overlap | Socket only; DMA complete-block slot is serialized |
| KDA span ACLGraph / CUDA graph capture | Optional one-token replay; CUDA DSA replay also available; Ascend DSA/prefill eager |
| flashcomm1 | Not implemented in this independent actor runtime |
| MTP | Disabled; main layers 0–44 only |
| Multimodal | Vision tower outside the initial text scope |
| Scheduler batching / prefix caching | Not implemented; one request owns state |
| General benchmark accuracy / long-context retrieval | Not established by capacity or arithmetic smoke |

The model config permits 1,048,576 positions. The deployed cap is selected
separately and must be capacity-tested with prompt plus requested output.
A 128K/16-sequence profile is outside the validated single-sequence runtime.
The report distinguishes configured context, completed capacity gates, and
quality coverage; no allocation estimate substitutes for a generation gate.

```bash
python -m tools.sidecard.layer_validate \
  --evaluation-config tests/e2e/models/configs/GLM5.3-Flash-Sidecard.yaml \
  --output /tmp/sidecard-api-gate.json
python -m tools.sidecard.layer_validate \
  --base-url http://192.168.53.187:8000 \
  --lengths 4096 8192 16384 65528 \
  --output /tmp/sidecard-context-gate.json
pytest --noconftest -q tests/ut/sidecard
```

The last length reserves eight output tokens and reaches a 65,536-token total.
Context fixtures measure capacity and latency, not recall accuracy.
`--noconftest` isolates these portable tools from the checkout's unavailable
upstream vLLM attention imports. NPU/CUDA gates require the actual hardware.

## CUDA stream rebuild receipts

The CUDA library can be rebuilt independently using the same clipping patch,
SM120 target, and `--fmad=false` compiler setting. Retain the clipped source
SHA256, exact compiler command, `library`, `library_sha256`, and
`swiglu_limit: 10.0` in its build JSON. Join it with the original clipped
Ascend build using the authenticated helper:

```bash
python -m tools.sidecard.layer_cuda_join \
  --output /srv/ai/storage/artifacts/sidecard-kernels-joined-new \
  --clipped-build /srv/ai/storage/artifacts/sidecard-clipped-kernels-20261012/build.json \
  --cuda-build /srv/ai/storage/artifacts/sidecard-cuda-stream-20261012/build.json
```

Use the joined receipt's CUDA path/hash and `native_gate_override` together.
The helper authenticates both architectures and copies unchanged Ascend
binaries; it does not claim hardware validation. Run the independent
`layer_expert_gate` with stream mode enabled before serving. It exercises
real W2/W3/W4 experts on a nondefault Torch stream against the CPU reference.
