---
library_name: vllm
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - multimodal
  - vision
  - ascend
  - ascend-310p
  - atlas-300i-duo
  - conversational
  - int4
  - modelslim
  - moe
  - qwen4_exp
  - sparse-attention
  - w4a16
  - w4a8
---

# Qwen3.8-Flash-Next W4 G128 for Ascend 300I

> [!WARNING]
> This is an experimental text-and-image checkpoint for Ascend 310P. It requires
> the pinned OpenSensor vLLM fork, the OpenSensor vLLM Ascend development
> runtime, and compiled custom operators. It is not a drop-in model for stock
> vLLM Ascend.

This is a complete four-bit checkpoint derived from
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. It targets two
Atlas 300I Duo cards, exposed as four Ascend 310P3 devices.

The checkpoint stores routed-expert weights as packed signed INT4 with
asymmetric groups of 128. Runtime policy determines the activation path:

| Model policy | Selected expert computation |
| --- | --- |
| W4 with activation quantization permitted | Native W4A8: per-group INT8 activation quantization and packing, then INT4 x INT4 to INT32 |
| W4 requiring FP16 activations | Existing W4A16 backend |
| [Companion W8A8 Dynamic](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i) | Existing W8A8 backend, unchanged |

Selection uses checkpoint quantization metadata, hardware support, and
supported shapes before graph capture. Native W4A8 is explicit; the runtime
does not silently change activation precision.

## Quantization and checkpoint format

Ascend ModelSlim IR converted 73,728 routed-expert projections from the
original BF16 checkpoint using asymmetric per-group min/max RTN. Each group
contains 128 weights. Two signed four-bit values are packed into each byte,
with the low nibble first; scales are FP16 and zero points are INT8. The
checkpoint advertises `qwen4exp_w4a16_group_v1` in
`text_config.ascend_expert_quantization`.

Router, shared experts, attention, Gated DeltaNet, embeddings, the PLE table,
vision components, and MTP weights remain floating-point where present.
Floating BF16 source tensors were exported as FP16 for the 310P serving path;
FP32 source tensors remain FP32.

The complete export contains 1,610 Safetensors shards and 222,746 tensors.
Tensor payload is 181,637,185,528 bytes (169.16 GiB); the model directory with
metadata is 181,713,896,545 bytes. The approximately 102.40 GB PLE table is
included in that total. See [provenance](PROVENANCE.json) for source,
conversion, byte accounting, and validation details.

## Native W4A8 result

The retained native backend keeps expert weights packed as INT4 and fuses
activation quantization and INT4 packing on device. The c1 decode shape uses a
compile-time streamed-weight schedule; c2-c4 instantiate the resident-weight
schedule. This split avoids the aggregate-throughput regression seen when the
streamed pipeline was compiled into every shape.

Measurements used the real checkpoint, two Atlas 300I Duo cards, TP4 with
expert parallelism, MTP2, `FULL_DECODE_ONLY` graphs, a 262,144-token serving
limit, four scheduler slots, 512 batched tokens, and fixed worker CPU affinity.
Warmup runs are excluded from the medians.

| Workload | Resident baseline | Retained native W4A8 | Change |
| --- | ---: | ---: | ---: |
| c1 decode, median | 28.1735 tok/s | **29.8610 tok/s** | **+5.99%** |
| c1 decode, peak | - | **30.3191 tok/s** | - |
| c4 aggregate decode, median | 59.6759 tok/s | **59.5233 tok/s** | -0.26% |

Candidate c1 runs were 29.9512, 30.3191, 29.8610, 29.7040, and 29.6911
tok/s. Candidate c4 aggregate runs were 62.1914, 59.5233, and 58.2283 tok/s.
The c4 difference is within observed run-to-run variance.

The fixed 228-question zero-shot quality gate scored **206/228 with zero
invalid answers**. This is a fixed sampled gate, not an official full-dataset
MMLU score. Across the validation service run, interval counters reported
7,412 accepted speculative tokens from 9,078 drafted tokens (81.65%). Five
real-NPU changing-input graph tests passed, including c1 gate/down, 120-row c4
fallback for both projection widths, and fused down with changing inputs,
routes, and weights.

Machine-readable results are in
[`native-w4a8-results-20260928.json`](native-w4a8-results-20260928.json).
Implementation and full test notes are in
[`opensensor/vllm-ascend@d4d2c42f1`](https://github.com/opensensor/vllm-ascend/commit/d4d2c42f1)
and its
[`WEIGHT-PIPELINE-C1-RESULTS.md`](https://github.com/opensensor/vllm-ascend/blob/0637d3baa6bc2000ba6433b301f233c6516847b4/artifacts/native-int4-20260927-pass3/WEIGHT-PIPELINE-C1-RESULTS.md).

### Grouped MTP update

Revision
[`opensensor/vllm-ascend@17954180d`](https://github.com/opensensor/vllm-ascend/commit/17954180d003ed1be4c799c305ec38cc33b55c1c)
adds an explicit `mtp_expert_execution=w8a8_grouped` mode. It quantizes the
MTP checkpoint's routed experts once at load time, stores the local expert
banks in CANN FRACTAL_NZ format, and runs both draft expert projections through
the 310P grouped W8A8 path. The established `w8a16_routed` draft path remains
the default, so this does not silently change MTP activation precision.

On the same four-device service configuration, five c1 runs measured 29.7374
tok/s median, 30.8504 tok/s mean, and 34.3382 tok/s peak. Three c4 trials
measured **61.3700, 60.8789, and 59.4403 aggregate tok/s**, for a **60.8789
tok/s median**. This is a 2.28% gain over the retained native-W4A8 c4 median
while c1 changed by -0.41%, within run-to-run variation. All four c4 requests
ran concurrently with none queued.

The candidate passed model loading, memory profiling, graph warmup, a coherent
generation smoke test, and the measured runs. Focused host coverage passed 71
tests with one skip. The earlier 206/228 quality gate was not rerun for this
isolated draft-path change. Raw results and test notes are in
[`qwen-next-gains-20260930`](https://github.com/opensensor/vllm-ascend/tree/17954180d/artifacts/qwen-next-gains-20260930).

## September 30 PLE frontend update

Revision
[`opensensor/vllm-ascend@d4d2c42f1`](https://github.com/opensensor/vllm-ascend/commit/d4d2c42f1)
removes the decode-time NPU-to-host synchronization from PLE n-gram hashing,
reuses a persistent parallel host gather pool, and adds the fused 310P PLE
gate/short-convolution epilogue. Hashing reads the runner's existing CPU token
mirrors; no additional token transfer is introduced.

On real checkpoint tensors, the exact one-token PLE frontend fell from 1.423 ms
to 0.470 ms. Its components measured 0.024 ms for host hashing, 0.312 ms for
hot host gather, 0.032 ms for pageable host-to-device copy, and 0.356 ms for
the 2560-to-12800 FP16 projection. The transfer is already small enough that
pinning it is not justified by these measurements.

The same revision adds an opt-in dynamic-W8A8 PLE projection. It must be
selected explicitly with
`ascend_expert_quantization.ple_projection_execution=w8a8_dynamic`; FP16 remains
the default, so the runtime never silently changes PLE activation precision.
Including activation quantization, the projection measured 0.201 ms versus
0.356 ms for FP16. Relative L2 error was 0.58% and cosine similarity was
0.999983 at the measured one-token shape.

Paired whole-model c1 trials used five 512-token completions. Dynamic W8A8
measured 27.16 tok/s median versus 26.99 tok/s for the exact projection. MTP
acceptance changed between outputs, so target-step time is the clearer kernel
measure: 85.18 ms versus 89.48 ms, a 4.81% reduction. The W8A8 trials ranged
from 26.24 to 32.57 tok/s. Six c4 runs measured 57.96 tok/s aggregate median;
the last three warm runs measured 58.57 tok/s median and peaked at 59.39 tok/s.
A changing-input graph replay returned the correct `17 x 19 = 323` result.

Machine-readable stage and whole-model evidence is in the commit's
[`qwen-ple-frontend-20260930`](https://github.com/opensensor/vllm-ascend/tree/5bc6a28ff/artifacts/qwen-ple-frontend-20260930)
artifact directory. These short-prompt measurements do not validate full-window
accuracy or throughput.

The earlier W4A16 runtime study measured 14.71 tok/s at short context and
14.60 tok/s near 23.4K context. Those historical results remain in
[`benchmark-summary-20260927.json`](benchmark-summary-20260927.json); they are
not directly comparable to the later native-W4A8 service because the runtime
and serving configuration changed.

## October 2 GPQA Diamond end-to-end result

The native W4A8 service completed all 198 GPQA Diamond questions on two Atlas
300I Duo cards (four Ascend 310P3 chips, TP4/EP4). The matched reference was
one RTX 6000 Pro running the same base model as an `UD-IQ4_XS` GGUF in
`llama.cpp`. Both used the same serialized AISBench zero-shot chain-of-thought
prompts (dataset version `b1ed2c`), temperature 0, seed 1024, top-k 20,
top-p 1, and an 8,192-token output ceiling. The servers' chat templates and
quantization formats differ, so this is a serving-stack comparison rather than
an isolated hardware or quantization test.

| Measure | Ascend W4 | RTX IQ4_XS |
| --- | ---: | ---: |
| AISBench-style `Answer: X` score | **140/198 (70.71%)** | **140/198 (70.71%)** |
| Responses with a final answer | 117/198 | 124/198 |
| Correct answers in final response text | 114/198 (57.58%) | 120/198 (60.61%) |
| Hit the 8,192-token output limit | 81/198 | 74/198 |
| Total output tokens | 949,014 | 957,402 |

The benchmark extractor scans the combined response for the last `Answer: X`.
It therefore credited 26 Ascend and 20 RTX cases where a correct letter
appeared in unfinished reasoning but generation stopped before any final
answer. The final-response row above is a separate completion diagnostic,
not the configured benchmark score. When both services did finish an answer
for a question, all 112 letters agreed (109 correct, three wrong).
Chemistry caused most output-limit hits: 62/93 on Ascend and 57/93 on RTX.

The final 106 Ascend cases used MTP2 with `FULL_DECODE_ONLY` C3/C4 graphs
captured at `[9, 12]` and four concurrent requests. On those same 106 cases,
median client-observed output rate was 12.50 tokens/s on Ascend versus 61.72
tokens/s on RTX with three slots. Ascend delivered 48.93 aggregate output
tokens/s over that 2 h 52 m 46 s phase. The RTX full run delivered 183.11
aggregate output tokens/s over 1 h 27 m 9 s; those wall rates cover different
case windows. The Ascend phase completed without an eager decode fallback,
zero-acceptance interval, or server error. Earlier Ascend cases were resumed
across server changes, so no single uninterrupted whole-run Ascend wall rate
is claimed.

The measured development runtime also synchronizes pending recurrent-state
NPU writes before a prefix-Mamba state slot is evicted or reused. The GPQA
client saves each completed case and resumes missing IDs; two prior timeout
rows were successfully replayed. The healthy final phase does not isolate
which reliability change prevented the earlier degraded server behavior.

This run does not evaluate the companion W8A8 checkpoint, full-context
accuracy, multimodal input, or MTP-versus-target-only quality parity.

## October 5 prefill and profiling update

The native-W4 development service now has an explicitly selected 2,560-token
expert chunk, built-in FP16 SwiGLU, and the CANN route finalizer. The native
pack and matmul operators support 25,600 top-10 routed rows, and the projection
schedule retains the smaller row tile through this batch size. The source
chunk default remains 1,536 tokens; selecting 2,560 requires a matching rebuilt
OPP package. The CANN finalizer remains opt-in because its rounding differs
from the torch finalizer. Model weights and checkpoint format are unchanged.

Three matched 23,410-token cold prompts on TP4/EP4, MTP2 and decode graphs
`[3, 6]` produced the following preliminary service results:

| Configuration | Mean cold TTFT | Effective prompt rate |
| --- | ---: | ---: |
| 1,536-token expert chunks, built-in SwiGLU, CANN finalizer | 64.292 s | 364.1 tok/s |
| 2,560-token expert chunks, revised projection schedule, same activation/finalizer | 62.032 s | 377.4 tok/s |

This is a **3.52% reduction in mean TTFT** across three cases. Every request
reported zero cached tokens. Two 32-token outputs matched exactly; one opening
phrase changed. The earlier run used a different service instance and day,
so run order and device conditions were not fully controlled. A one-layer
23,410-token partial improved from 457.74 to 420.97 ms with exact output
parity; it excludes attention, shared experts, TP reduction and scheduling.

An October 5 resident-server replay of the saved case-1 prompt measured
62.392 s TTFT versus its previous 61.840 s, with zero cached tokens and
identical output. A short-context 512-token decode observation measured
30.91 tok/s; replaying that request measured 31.87 tok/s, but later output
wording differed. These are individual observations, not a demonstrated
speedup in decode or a guarantee at long context. Separate conversational
requests around 22K–26K context measured approximately 26 tok/s decode.

The service reported 1,068,936 cache tokens and 4.08x planner concurrency at
262,144 tokens per request, and captured both decode graphs. This is startup
capacity accounting, not a measured four-request full-window workload.
Sustained thermal behavior, concurrent-request latency and broader quality
remain unqualified for the latest prefill configuration. The October 2 GPQA
result above was obtained with the earlier serving configuration and was not
rerun for this prefill update.

Four-rank named profiling measured approximately 95.7% device occupancy.
Native W4 projections, QSA K/V gathers and large hyperconnection casts remain
major targets. Long host copy calls overlapped active device execution almost
entirely, so their host duration cannot be counted as freely removable latency.
The published trace PNGs show both the task timeline and measured occupancy.

Two further candidates remain experimental and are **not serving defaults**:

- A separately named sparse QSA gather avoids fallback cache reads for masked
  groups. It passed 32 NPU correctness cases and produced exact full-attention
  outputs. Selecting it only for sparse tiles reduced first-chunk synthetic
  parallel-gather attention from 192.566 to 188.835 ms (1.94%); dense tiles
  retain the existing operator. Real multi-request/prefix dispatch and model
  TTFT remain unqualified.
- A resident Python candidate reuses the FP16 normalized hyperconnection
  projection operand, removing a repeated cast while retaining FP32 mixing
  math. Its host tests passed, but NPU performance and serving recapture are
  pending. It is inactive in the measured service.

The implementation and evidence are published at
[`opensensor/vllm-ascend@ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032`](https://github.com/opensensor/vllm-ascend/commit/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032).
See the [batching results and trace figures](https://github.com/opensensor/vllm-ascend/tree/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032/artifacts/qwen38-prefill-batch256-20261005),
[runtime runbook](https://github.com/opensensor/vllm-ascend/blob/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032/docs/source/developer_guide/performance_and_debug/qwen38_310p_runtime_runbook.md),
and [resident experiment status](https://github.com/opensensor/vllm-ascend/blob/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032/artifacts/qwen38-prefill-batch256-20261005/resident-20261005/README.md).

## Image input support (October 8)

**Text and image input work with the OpenSensor Qwen serving implementations
on Ascend 310P.** Both quantized checkpoints retain the vision encoder and
multimodal processor; expert quantization does not remove image support.
Use the matching [OpenSensor vLLM fork](https://github.com/opensensor/vllm)
and [vLLM Ascend fork](https://github.com/opensensor/vllm-ascend).

The live W4 service processed fresh image requests on October 8, 2026, as
confirmed by the operator. Its vision weights loaded and multimodal cache
activity appeared in the service logs. This is functional confirmation, not
an image accuracy benchmark. The W8 implementation uses the same image
frontend; its historical text performance measurements below have not been
rerun as multimodal benchmarks.

Enable the vision path by omitting `--language-model-only` and using:

```bash
--limit-mm-per-prompt '{"image":1,"video":0}' \
--mm-processor-kwargs '{"min_pixels":65536,"max_pixels":1048576}'
```

This profile accepts one image per prompt and bounds processor resizing to
1,048,576 pixels. Video is disabled in this profile. Image processing consumes
additional memory and prefill time; text benchmark rates below describe their
original text workloads. The client must advertise image input and send an
OpenAI-compatible `image_url` content part. For example, using only the Python
standard library, with a local PNG saved as `screenshot.png`:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

image = base64.b64encode(Path("screenshot.png").read_bytes()).decode("ascii")
payload = {
    "model": "YOUR_SERVED_MODEL_NAME",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this screenshot."},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + image,
            }},
        ],
    }],
    "max_tokens": 256,
}
request = Request(
    "http://localhost:8000/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=300) as response:
    print(response.read().decode("utf-8"))
```

Replace `YOUR_SERVED_MODEL_NAME` with the ID returned by `/v1/models`; the W4
recovery service uses `qwen38-flashnext-latest-gate`. Image support does not
establish video support, image quality parity, or full-context accuracy.

On the live W4 recovery service, long-running multi-session traffic still
exhausted the prefix-Mamba device checkpoint tiers and spilled state to CPU,
even with low KV cache usage. Three scheduler slots and the startup KV planner
capacity do not guarantee spill-free retained sessions or sustained throughput.
The [serving report](https://github.com/opensensor/vllm-ascend/tree/main/artifacts/qwen38-serving-gate-20261007)
records this limitation; it is separate from functional image input support.

## Image-enabled serving example

The October 5 text measurements used these software revisions:

- [`opensensor/vllm@3ab5dda29`](https://github.com/opensensor/vllm/commit/3ab5dda29acabea01f6a63d0806bdbbb4a27bde5)
- [`opensensor/vllm-ascend@ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032`](https://github.com/opensensor/vllm-ascend/commit/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032)

Build and install the matching 310P custom operators and activate the CANN
environment as described in the [runtime runbook](https://github.com/opensensor/vllm-ascend/blob/main/docs/source/developer_guide/performance_and_debug/qwen38_310p_runtime_runbook.md).
The October 8 image-enabled recovery profile also includes the existing
[qLens capacity and mixed-prefill fixes](https://github.com/opensensor/vllm-ascend/commit/483b57819).
Its launcher and configuration evidence are in the
[October 7–8 serving report](https://github.com/opensensor/vllm-ascend/tree/main/artifacts/qwen38-serving-gate-20261007).
Use a complete compatible runtime and OPP package; assembling unrelated model
files from different snapshots does not reproduce this deployment.

The example below uses three scheduler slots, MTP2, graphs `[3,9]`, a
1,024-token scheduler batch and image processing. The matching four-device
recovery profile reported 997,292 cache tokens (3.80x planner capacity at
262,144 tokens per request), with a 4 GiB workspace reserve. This is startup
accounting, not a measured three-request full-window workload. It deliberately
profiles memory again instead of reusing the earlier fixed text-only KV budget.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_ASCEND_KV_CACHE_FRACTION=0.70
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1

vllm serve matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name qwen38-flashnext-latest-gate \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --no-async-scheduling \
  --disable-custom-all-reduce \
  --max-model-len 262144 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 3 \
  --gpu-memory-utilization 0.965 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --mamba-cache-mode align \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,9]}' \
  --hf-overrides '{"text_config":{"ascend_expert_quantization":{"backend":"cube_310_int4_a8","activation_quantization":"int8_per_group","grouped_activation":"cann_builtin_fp16","grouped_finalize":"cann_v2","grouped_prefill_chunk_tokens":2560,"bits":4,"format":"qwen4exp_w4a16_group_v1","group_size":128,"lm_head_execution":"w8a8_dynamic","ple_projection_execution":"w8a8_dynamic","offset_dtype":"int8","packing":"signed_int4_low_nibble_first_in_axis","scale_dtype":"float16","shared_expert_execution":"tp_sharded","symmetric":false}}}' \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":1048576}'
```

Do not add `--quantization ascend`: model-specific metadata selects the W4
loader, and the W4 loader rejects that generic override. Removing the native
backend override keeps the checkpoint's FP16-activation W4A16 path.

## Validation scope and limitations

- The public source revision and custom operator implementation are pinned,
  but users must build the 310P operators locally; stock vLLM Ascend does not
  contain this backend.
- The 262,144-token value is a configured serving limit. Full-window accuracy
  and throughput have not been validated.
- Four scheduler slots were active during aggregate testing; four simultaneous
  full 262K windows were not filled or measured.
- The 228-question gate is useful regression evidence, not a broad substitute
  for GPQA, coding, or official full-dataset MMLU evaluation.
- Image input is functional in the October 8 profile; image quality and video
  have not been benchmarked. FlashComm1 remains unvalidated for this profile.
- Cold prefill remains workload-sensitive.

The model weights are distributed under the
[Qwen Community License 1.0](LICENSE), inherited from the base model. Review
its terms before use. The vLLM Ascend code is separate from these weights and
retains its own license.

Please cite the original Qwen model and technical report when using this
checkpoint.
