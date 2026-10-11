# GLM on one Ascend card and the RTX

The previous TP4 service was stopped at the user's request. The replacement
is running at `http://192.168.53.187:8001/v1`, model `glm-sidecard`, using this
allocation. See the [hardware receipts](../../artifacts/sidecard-glm-20261011/one-card/README.md).

| Hardware | Assignment |
| --- | --- |
| Ascend card 16409, PCIe 49:00.0, physical devices 2 and 3 | Sidecard TP2 backbone and native experts |
| RTX PRO 6000 Blackwell | Packed CUDA experts |
| Ascend card 8217, PCIe 21:00.0, physical devices 0 and 1 | Existing Qwen service preserved |

## Explicit ownership

Every main MoE layer, 3–44, assigns experts 0–143 to CUDA, 144–215 to NPU rank
0 and 216–287 to NPU rank 1. Rank 0 composes its CUDA and native contributions;
rank 1 contributes its native range. The existing GLM composition retains
routed scaling, tensor-parallel shared experts and the final HCCL reduction.
Attention and recurrent state remain on Ascend.
NPU owner names identify TP ranks 0 and 1, independently of the physical device
pair. Physical selection accepts exactly `(0, 1)` or `(2, 3)`; cross-card pairs
and four-device allocations are rejected. The current deployment uses `(2, 3)`.

The main thread copies immutable input packets before native expert execution.
One session-owned thread exchanges those host packets with CUDA while the
main thread runs its NPU experts. Every NPU operation stays on the caller's
stream. The caller joins the CUDA reply before combining results. Native or
transport errors close the session; no fallback response is returned.

The native checkpoint has four 72-expert file partitions. The one-card profile
uses two NPU workers and two partitions on CUDA. `stage_checkpoint` hard-links
the original tensor files and replaces copied metadata only. It preserves the
source manifest hash, scale receipts, precision schedule and native kernel
bundles. Changing a TP4 launch flag alone does not establish this ownership.
The loader hook sets the NPU bank ranges before any expert allocation; the
streamed loader follows those admitted ranges.

## Packed CUDA weights

CUDA stores four signed codes in each 16-bit word, in K32 word planes. W2,
W3 and W4 integer codes are preserved exactly; the existing FP16 scales are
copied unchanged. This halves code storage relative to the INT8 CUDA banks,
allowing 144 experts per layer in approximately 71.2 GiB of bank storage.
Admission also reserves projection workspace, temporary Cube conversion
storage and 4 GiB of free device memory.

The direct CUDA preparer translates authenticated Cube bytes into this layout
without widening whole banks on CPU. CPU shard handles are reused within each
layer. Temporary device conversion storage is freed after preparation.

The scalar/DP4A reference uses wide weight addresses: a 144-expert gate bank
crosses the signed 32-bit byte-address limit. The reference and packed kernels
retain ordered FP32 block scaling and accumulation. Partial CUDA outputs
combine weighted expert rows in original router slot order.

## Qualification and launch

Stage a fresh checkpoint using the source tree and qualified interpreter:

```bash
python -m tools.sidecard.one_card \
  --source /srv/ai/storage/models/GLM-5.3-Flash-two-card-native-candidate \
  --output /srv/ai/storage/models/GLM-5.3-Flash-one-card-sidecard-candidate
```

`packed_weight_gate` checks direct Cube conversion against independent CPU
packing for all three precisions and multiple tile geometries. The expanded
`batch_projection_gate --one-card --weight-layout k32_nibbles_n_v1
--partial-reduce` compares actual checkpoint outputs from 144 experts in W4,
W3 and W2 layers, across 128/512/640 tokens and 1/2/8 routes per token.
The 27 cases must be bitwise equal, and their receipt must pin the compiled
library and original checkpoint manifest.

Prepare a fresh service with `service_prepare --one-card --tensor-transport
--packet-tokens 512 --weight-layout k32_nibbles_n_v1 --partial-reduce
--cuda-gate PATH --npu-devices 2,3`. This sets `ASCEND_RT_VISIBLE_DEVICES=2,3`, TP2,
and router owner rank 1. It refuses an unpacked INT8 bank cache. Wait for
`cuda-ready.json` before launching the API.

Start with conservative KV allocation and `--max-model-len -1`; measure
admitted capacity and memory before increasing the cache budget. The previous
645,120-token limit is historical and does not apply to this allocation.
Use the qualified virtual environment for preparation and launch; the system
interpreter does not have the required project dependencies.

The smaller placement omits the resident CUDA-expert NPU shadow copies. The
validator identifies the pinned GPU correctness gate separately from the
serialized-versus-overlapped checks. Three short calls and one bulk prefill
per layer run both schedules on identical real inputs. All 168 comparisons
passed with zero error, and all seven operational API cases passed. Greedy
repeat variation remains unresolved; whole-model quality is unqualified.

The running prefix-enabled profile uses 3.5 GiB KV per NPU and advertises
260,352 context tokens. Kilo reserves 32,000 output tokens and 228,352 input
tokens. Near-capacity admission was checked by disconnecting after response
headers; this is not a completed 260K prefill or a long-context quality result.
The earlier 4 GiB candidate left only 28 MiB free on the smaller NPU after
8K and mixed-size requests, so its 300,672-token limit was retired.

Stop only saved, command-verified process IDs and wait for exit before another
CUDA admission query. The resident daemon handles SIGTERM/SIGINT by releasing
its banks and socket, including when its launcher inherited SIGINT ignored.
