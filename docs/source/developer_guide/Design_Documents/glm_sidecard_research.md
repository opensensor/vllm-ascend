# GLM sidecard research review

Reviewed October 11, 2026. This consolidates the prior sidecar discussion,
local GLM adaptation/performance work, hardware qualification and relevant
upstream architecture. The experimental `tools/sidecard` implementation now
serves joined GLM requests with CUDA routed experts and an Ascend backbone.
The intended CUDA attention/state ownership remains future work.

## The existing idea

The October 6 conversation settled on a capacity-first design: RTX owns
attention, persistent state, shared components and selected experts; Ascend
executes resident overflow experts. Compare against RTX plus CPU/weight
offload because the desired GLM checkpoint exceeds NVIDIA VRAM. Move compact
activations to the weights. Optimize token latency under capacity and fidelity
constraints, considering expert co-occurrence as well as frequency.

The retained conversation is session `01a10f7d-0354-72b2-8e33-ba475808d982`,
dated October 6 in the local session archive. Its initial illustrative speed
ranges were explicitly superseded by this capacity-first comparison. No
runtime implementation was delivered in that discussion. Drafting/MTP was
also explored; it is a later candidate because it needs accepted-token and
rollback accounting on top of correct target execution.

## Local evidence and reuse

| Source | Finding | Consequence for sidecard |
| --- | --- | --- |
| [Original GLM adaptation plan](../../../../glm53-flash-w2-310p-plan.md) | Real geometry: 45 layers, hidden 4096, 288 experts/top-8, 42 main routed layers, 34 KDA and 11 DSA layers | Read and hash the real config; seed the expert split for GLM-5.3-Flash |
| [Progress and corrections](../../../../artifacts/glm-progress-summary.md) | Initial incoherence involved projection/rounding/state and stale kernel defects; later real-weight execution became coherent | Read updates in order; early claims that bf16 emulation or quantization alone were the blocker are historical hypotheses |
| [October 6 decode profile](../../../../artifacts/glm-perf-310p/decode-trace-20261006/README.md) | Packed expert projections remain the largest identified task family; attribution is not additive recoverable latency | Measure the Ascend branch; moving attention alone does not guarantee fast generation |
| [Streaming design](../../../../artifacts/glm-perf-310p/scale-layout-live-20261009/STREAMING.md) | K32 factors, bounded tile lifetimes, compact gate/up-to-down handoff and FP32 addition order are numerical contracts | Reuse native expert projections; preserve scale, activation and output rounding markers |
| [Streaming experiment plan](../../../../glm-streaming-plan.md) | v1111 passed native/actual-input gates but regressed c4 serving by 6.8%; v1021 restored | Kernel parity alone does not justify serving promotion; pin build and live source provenance |
| [Two/three-card qualification](../../../../artifacts/glm-perf-310p/hardware-card-profiles-20261010/README.md) | Two-card real-weight TP4/MTP1 passed short arithmetic/tool/prefix gates; one 17-token tool run measured about 8.2 tok/s over decode gaps | Retain a matched Ascend-only control; this is short-request evidence, not broad quality or long-context proof |
| [Six-chip sharding analysis](../../../../artifacts/glm-perf-310p/six-chip-sharding-20261009/README.md) | Expert placement, attention head divisibility, pipeline mHC state and MTP have separate constraints | Separate stateless expert workers from CUDA-owned attention topology |
| `tools/glm_perf/glm_fused_moe.py`, `glm_int4.py`, `fused_weight_layout.py` | Native routing/projection/reduction and packed bank layout already exist | Extract an expert-only boundary; keep ongoing kernel experiments separate |
| `vllm_ascend/model_loader/glm_native_int4.py`, `tools/glm_perf/native_checkpoint.py` | Existing loaders/exporters validate native partition/layout identity | Add explicit arbitrary ownership exports; existing 72-expert rank exports cannot silently become 54-expert banks |
| `vllm_ascend/models/glm5next_w2/moe.py`, `model.py` | Current expert parallelism follows TP and includes shared/reduction responsibilities | The sidecard seam must remove those whole-model assumptions explicitly |

Historical GPU golden captures in `artifacts/golden_gpu_activations.npz` and
`golden_gpu_mhc_activations.npz` provide useful L0/L3/L7 diagnostics. They came
from a rented GPU with an NVFP4 source and are not current joined-hardware
results. The old 28 GB NVFP4 directory was an incomplete download, not a
complete resident model. GPU and NPU variants must pin source and precision
before claiming parity.

## Current hardware

Read-only SSH inspection of `matteius-threadripper` on October 11 found:

- RTX PRO 6000 Blackwell Workstation Edition, reported 97,887 MiB, PCIe
  `0000:01:00.0`.
- Two 310P3 cards at `0000:21:00.0` and `0000:49:00.0`, four logical devices
  0–3, all healthy. Chips report 47,302 / 46,717 MiB per pair.
- Approximately 247 GiB system RAM. GPU and the two cards appear beneath
  different PCIe root domains; this does not prove a direct transfer path.
- No NPU model processes were reported at inventory time. The historical
  six-chip/chips-2–5 launcher mapping is therefore not today's mapping.

The complete dated command receipts are in
[inventory.json](../../../../artifacts/sidecard-glm-20261011/inventory.json).
That initial inventory was hardware discovery only. After hardware
authorization, the [hardware report](../../../../artifacts/sidecard-glm-20261011/HARDWARE.md)
records successful host-staged CUDA/Ascend copies on all four chips and
single-expert native arithmetic on real W4/W3/W2 weights. A GLM TP4 server
was resident during those later probes and remained running afterward.
Subsequent joined requests passed seven operational API cases and 126
real-input expert shadow checks across all 42 routed layers. Greedy repeat
variation remains unresolved. The 8 GiB-per-rank capacity profile admitted
709,120 tokens and completed a short chat with a 32,000-token output budget;
full-capacity prefill and long-context quality were not evaluated. Its
128-token prefill iterations took about 25 seconds, requiring a separate
performance correction. See the [service runbook](../../../../tools/sidecard/SERVICE.md)
for current profiles and measured results.

## Upstream designs checked

The [vLLM AFD plugin](https://github.com/vllm-project/afd-plugin) separates
attention and FFN roles with workers, wrappers and connectors. Its current
README identifies a vLLM 0.30.0 target and lists CUDA/NPU connectors in
separate backend packages. The listed NCCL and HCCL/CAM connectors do not
establish a CUDA-to-310P path. Current model support and hardware validation
also do not establish GLM-5.3-Flash hybrid KDA/DSA support. Reuse its narrow
role/connector idea; audit version and model support before importing it.

[KTransformers](https://github.com/kvcache-ai/ktransformers) is a useful
comparison for heterogeneous resident/offloaded MoE execution. CPU/GPU
support or an Ascend path is not evidence that CUDA and this 310P kernel
stack can execute jointly under one process or collective.

[CUDA transfer guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
describes pinned memory and stream requirements for asynchronous host copies.
Initial cross-vendor transport should use explicitly owned host buffers and
completion fences. Whether a buffer can be shared or registered efficiently
with both runtime stacks remains a hardware gate. Same-host installation and
the PCIe tree do not establish peer DMA or overlapping execution.

## First deliverable and next experiment

The [sidecard tooling](../../../../tools/sidecard/README.md) provides explicit
per-layer ownership, config hashing, logical storage lower bounds, read-only
inventory and a CPU dispatch/combine oracle. Offline implementation now also
includes bounded binary packets, five persistent CPU worker processes, a
complete-batch backend seam, canonical safetensors ownership planning,
unchanged-byte owner exports, verified partition loading and startup coverage
acknowledgements. The complete export-to-child-execution path is exercised
with small synthetic packed-weight fixtures; it is not a real GLM or native
Cube export qualification.
The initial manifest assigns
72 experts to CUDA and 54 to each of four NPUs; it makes no admission or
throughput claim. MTP and graphs are deferred in this candidate.

The initial hardware gates passed 24 synchronous transfer cases and 12
single-expert arithmetic cases. The latter used synthetic activations and
matched the CPU native reference exactly; they do not establish captured GLM
or CUDA parity. Top-k=1/unit-weight execution provides a candidate raw-output
seam without a model-runner change. Its route gathering and complete owner-bank
behavior still need qualification. The diagnostic 80 MiB prefill return took
roughly half a second, including integrity checks and extra verification copies.

The selected-expert packet gate now executes actual global experts 0–7 on
CUDA and all four Ascend chips, with explicit noncontiguous local maps. All
18 W4/W3/W2 cases matched CPU raw rows and final output bitwise. Final
combination also runs on CUDA and passed six explicit FP32 ordering/rounding
cases. The CUDA kernel is a correctness implementation using unpacked INT8
weights. Packet handling/validation remains expensive, and this fixture does
not establish whole-model bank capacity, attention behavior or serving speed.

## First joined service

The [initial service](../../../../tools/sidecard/SERVICE.md) now runs as
`glm-sidecard` on `192.168.53.187:8001`. To establish a serving boundary with
the installed environments, this profile retains the qualified Ascend
backbone and moves native EP rank-zero experts 0–71 to CUDA in every routed
layer. The other three native expert partitions remain on Ascend; all four
chips still perform backbone work. This is a bootstrap profile distinct from
the target CUDA-attention/54-experts-per-chip design above.

CUDA admits and authenticates 3,024 resident experts, using 71.02 GiB of actual
allocations. One CPU layer is staged at a time. The opt-in worker extension
composes the existing native loader and expert-method seams; model-runner code
and source checkpoint files remain unchanged. Real requests use bounded expert
packets, then the existing GLM scaling/shared/all-reduce composition. Eager
execution is required and original NPU shadow banks remain resident.

Seven operational cases passed: completion, chat, repeated completion,
243-token chunked prefill, two concurrent requests and streaming. The repeat
request completed but its text changed, so full validation remains false.
Every routed layer accumulated additional CUDA routes during API validation.
The 126 real-input expert shadow comparisons had maximum absolute error
1.1921e-7 and maximum NRMSE 6.8954e-8. These operator checks do not qualify
whole-model quality or explain the observed greedy repetition variation.
The initial chat arithmetic smoke returned `4` with a completed response.

The next numerical gate is an isolated comparison against matched Ascend-only
eager execution to locate repeat variation before claiming model fidelity.
CUDA attention/state, packed CUDA MoE, the arbitrary 54-expert Ascend banks,
graph capture, MTP and matched performance comparisons remain future work.
