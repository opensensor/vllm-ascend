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
  - modelslim
  - moe
  - qwen4_exp
  - sparse-attention
  - w8a8
---

# Qwen3.8-Flash-Next W8A8 Dynamic for Ascend 300I

> [!WARNING]
> This is an experimental, locally validated checkpoint, not a final release.
> It requires a custom development runtime based on the public OpenSensor
> vLLM and vLLM Ascend forks; it is not a drop-in model for stock vLLM Ascend.
> A controlled W8 decode study is available below. The currently documented
> public runtime revision is pinned, but broader W8 quality and long-context
> evaluation remain in progress.

This repository contains an Ascend ModelSlim `W8A8_DYNAMIC` conversion of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
pinned at revision
[`de4b8e4d43b917e7706784d8bb445c9af86a3540`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540).
It targets two Atlas 300I Duo cards, exposed as four Ascend 310P3 devices, with
tensor parallelism across all four devices.

## Quantization variants

| Checkpoint | Routed-expert weights | Expert activation path on 310P | Status |
| --- | --- | --- | --- |
| This repository, W8A8 Dynamic | INT8, per-output-channel | FP16 input; dynamic per-token INT8 inside the grouped matmul | Text and image support; September 26 text measurements below |
| [W4 G128 for Ascend 300I](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i) | Packed INT4, asymmetric groups of 128 | Explicit W4A16 or native W4A8 backend | Separate experimental checkpoint; see its card for hardware results and limits |

Each repository contains a complete checkpoint with its own loader metadata.
The W8 checkpoint stores INT8 expert weights, not INT8 activations. Its selected
310P target path accepts FP16 activation tensors and dynamically quantizes them
per token inside the grouped matmul, so `W8A8_DYNAMIC` describes the expert
arithmetic. The September 26 MTP draft instead used W8A16 weight-only expert
matmuls with FP16 activations; that does not change the target model's W8A8 path.
The W4 checkpoint now has a validated, explicitly selected native W4A8 backend
in addition to its FP16-activation W4A16 path. Results from different serving
configurations are not a controlled quantization quality or speed comparison.

## September 30 runtime status

The September 30 public integration revision was
[`opensensor/vllm-ascend@d4d2c42f1`](https://github.com/opensensor/vllm-ascend/commit/d4d2c42f1),
paired with
[`opensensor/vllm@3ab5dda29`](https://github.com/opensensor/vllm/commit/3ab5dda29acabea01f6a63d0806bdbbb4a27bde5).
The latest work keeps this checkpoint on the existing W8A8 backend and adds an
independent native W4A8 backend selected from W4 metadata and explicit runtime
policy. It does not globally change activation precision or redirect W8 models
through the W4 kernels.

The integrated revision also moves Qwen4Exp PLE n-gram hashing to existing
CPU token mirrors, reuses the host gather pool, and adds a fused 310P PLE
epilogue. These model-independent PLE changes are available to this checkpoint,
but the W8 checkpoint has not been rerun on that revision. The new dynamic-W8A8
2560-to-12800 PLE projection remains an explicit W4 metadata option and is not
silently enabled for this W8 checkpoint.

The W8 checkpoint was not rerun during the latest W4 kernel optimization, so
the measured W8 numbers below remain the September 26 results. The companion
W4 card records the two-card native-target measurements and the latest
explicit grouped-MTP result: 29.74 tok/s median c1 and **60.88 tok/s aggregate
c4**, with the earlier fixed 206/228 zero-shot quality gate and zero invalid
answers. That quality gate was not rerun for the isolated grouped draft change.

**September 26 update: approximately 19 tokens/s in the decode study and a
160,000-token configured context limit on the same hardware.** The restored
server measured 19.22 tokens/s for a 512-token completion. These speed results
use shorter prompts; full-160K correctness and decode speed are not yet validated.

The checkpoint preserves the base model's Qwen4 experimental architecture:
48 decoder layers, 512 routed experts, hybrid Gated DeltaNet/Qwen Sparse
Attention, PLE n-gram embeddings, a vision encoder, and MTP weights. Text and
image input are supported by the OpenSensor serving implementation. MTP quality
parity remains provisional pending the quality comparison.

### October 2 companion W4 validation

The [separate W4 G128 checkpoint](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i)
completed a 198-question GPQA Diamond replay on four Ascend 310P3 chips. Its
AISBench-style score was **140/198 (70.71%)**, equal to a matched RTX 6000 Pro
IQ4_XS run. A final response actually contained the correct answer on 114/198
Ascend questions versus 120/198 RTX questions; many other responses reached
the 8,192-token output limit while still reasoning. The W4 card documents the
full protocol, scoring caveat, and throughput. **These are W4 results; this
W8A8 checkpoint was not used in that run.** The W8 full 198-question quality
comparison remains outstanding.

### October 5 companion W4 runtime update

The [separate W4 G128 checkpoint](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i)
now records three matched 23,410-token cold-prefill observations: increasing
actual expert chunks from 1,536 to 2,560 tokens with a revised native INT4
projection schedule reduced mean TTFT from 64.292 to 62.032 seconds (3.52%).
Those runs used built-in FP16 SwiGLU, an opt-in CANN route finalizer, TP4/EP4,
MTP2 and two decode graphs. They came from separate service instances and days,
so this remains preliminary performance evidence. Latest short-context decode
observations were approximately 31 tok/s, without a demonstrated decode gain.

The [published Qwen runtime work](https://github.com/opensensor/vllm-ascend/commit/ffa980c1d0ac9f31cd1dbbd06cd4bd3286571032)
includes trace PNGs, operator evidence and experimental resident controls.
The new sparse gather and cast-reuse candidates are not serving defaults.
**These performance measurements belong to W4; this W8A8 checkpoint was not
rerun, and its measured results below remain unchanged.** Native INT4 chunk
limits, finalizer selection and W4 command overrides must not be applied to
this W8 checkpoint.

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

## What is quantized

The conversion used Ascend ModelSlim's `W8A8_DYNAMIC` route:

- 73,728 routed-expert linear projections use INT8 weights with FP32
  per-output-channel scales and offsets.
- In the selected 310P target path, FP16 routed-expert activations are
  quantized dynamically to INT8 per token inside the grouped matmul. The
  checkpoint does not contain stored INT8 activation tensors.
- Embeddings, PLE tables, attention, Gated DeltaNet, shared experts, vision
  components, and the language-model head remain FP16 where present.
- The export contains 254 Safetensors shards and is approximately 240.06 GB
  (223.57 GiB). It is not a uniformly INT8 checkpoint.

## September 26 provisional device milestone

The selected `affinity` build binds each tensor-parallel worker to six physical
CPU cores on the same Threadripper 3970X host with two Atlas 300I Duo cards
(four Ascend 310P3 devices). The live API confirms the 160,000-token limit.
The configuration uses one-token MTP speculative decoding, prefix caching,
2,048-token chunked prefill, and `FULL_DECODE_ONLY` graphs.

| Measurement | Observed value |
| --- | ---: |
| Serving limit (`--max-model-len`, prompt plus completion) | 160,000 tokens |
| TP ranks / active sequences | 4 / 1 |
| MTP draft depth | 1 token |
| Restored-server check: 32-token prompt, 512-token completion | 19.22 tokens/s |
| MTP acceptance in that check | 89.26% |

The controlled comparison isolates CPU affinity against the contemporary
serving baseline. Each row is the median of three different prompts, with
temperature 0, seed 42, thinking disabled, and `ignore_eos=true`:

| Workload | Control tokens/s | CPU affinity tokens/s | Median paired gain |
| --- | ---: | ---: | ---: |
| Short prompt, 512 output tokens | 17.90 | 19.01 | +4.60% |
| Short prompt, 2,048 output tokens | 18.40 | 19.52 | +5.49% |
| Approximately 23K prompt, 512 output tokens | 17.88 | 18.67 | +5.05% |

Decode rate is calculated from server-side token-gap timings and excludes
prefill. These are single-request measurements, not aggregate throughput.
The gain column is the median of paired request ratios, not the ratio of the
two displayed medians. The long prompts contain 23,407–23,424 tokens, including
23,168 cached prefix tokens; decoding still attends to the full prompt.
Cold-prefix warmup time to first token was 45.65 seconds for control and
45.35 seconds with affinity, showing no material prefill improvement from
CPU binding. The approximately 5% paired gain should not be confused with the
entire improvement over earlier development builds.

Later live traffic also demonstrates workload sensitivity: 15 completed
requests with 17,880–50,233 prompt tokens and 201–3,506 output tokens measured
14.0–17.7 tokens/s between 23:20 and 23:36 UTC. This is an uncontrolled log
sample, not an additional A/B test. Approximately 19 tokens/s is therefore a
measured benchmark result, not a guarantee for every prompt length or workload.

**160,000 tokens is the configured serving limit, not a completed full-context
correctness test or the model's architectural maximum.** A controlled
100K/160K prompt test with retained outputs remains a release gate. Two
simultaneous 160K contexts have not been implemented or validated.

The seven-item semantic/tool smoke check matched control at 6/7, including a
passing strict-argument tool call. Both builds failed the same string-reversal
question. All nine paired benchmark outputs had different hashes, so these
results do not establish bit-exact output or broad quality equivalence.
The experimental grouped-MTP draft candidate was not promoted after a
tool-call stall and device-memory errors; its root cause remains unresolved.
The selected build retains the baseline W8A16 draft expert path.

The [benchmark evidence summary](benchmark-summary-20260926.json) includes
per-request measurements, MTP acceptance, the restored-server check, runtime
settings, and source-artifact checksums. The vLLM build is
[`opensensor/vllm@3ab5dda29`](https://github.com/opensensor/vllm/commit/3ab5dda29acabea01f6a63d0806bdbbb4a27bde5).
The Ascend runtime is the experimental
`qwen38-decode-study-20260926/affinity` snapshot, with torch `2.13.0+cpu`,
torch-npu `2.13.0rc1`, transformers `5.17.0`, and CANN / `npu-smi` `26.0.rc1`.
The later public integration revision is pinned above; these measurements still
refer to the September 26 `affinity` snapshot.

### Previous September 25 milestone

The earlier `qwen38-mtp-grouped-qsa-r21` build used a 131,072-token serving cap
and reported 14.8–16.0 tokens/s, including 14.8 tokens/s for a 1,359-token
completion. Uncached approximately 24K prompts took 61.5–83.3 seconds.
Those provisional timings are historical observations from a different
development build and workload, not the control arm of the September 26 study.

## Initial smoke baseline

The following smoke test was observed on September 23, 2026:

- Hardware: 2 x Atlas 300I Duo, 4 x Ascend 310P3 chips
- Tensor parallel size: 4
- vLLM Ascend revision:
  [`e3091d23015e3fa4a75471fc9d57f45e8b6c7a6b`](https://github.com/opensensor/vllm-ascend/commit/e3091d23015e3fa4a75471fc9d57f45e8b6c7a6b)
  on public branch `main`
- vLLM fork revision:
  [`3756b28a3c63d7d5c5503912cdc5e9dfbbe3c7f6`](https://github.com/opensensor/vllm/commit/3756b28a3c63d7d5c5503912cdc5e9dfbbe3c7f6)
  on public branch `main`
- CANN / `npu-smi`: `26.0.rc1`
- Mode: eager execution, FP16 non-quantized compute, Ascend quantization
- Tested context limit: 2,304 tokens
- Weight loading: 33.363 GB reported per TP rank
- API readiness: `/v1/models` returned HTTP 200
- Generation smoke test: `/v1/chat/completions` returned HTTP 200

This is functional validation, not a performance or quality benchmark. The
observed smoke request showed approximately 7 prompt tokens/s and generation
throughput varying from 0.2 to 1.9 tokens/s. Those figures are not controlled,
steady-state measurements and should not be used for comparisons.
They are retained as historical bring-up evidence and have been superseded by
the September 26 provisional milestone above.

The Qwen4Exp MTP draft head and 310P runner integration were subsequently
merged in vLLM Ascend
[`839b6657f`](https://github.com/opensensor/vllm-ascend/commit/839b6657f2e08b60fcd2d3fa124341003465104f).
The draft's FP16 expert loading, four-stream hidden-state width, runner
selection, and PLE history staging passed 17 focused host-side tests. The
September 23 smoke results above used the earlier revisions and did not use
MTP; real-checkpoint MTP generation was subsequently exercised in the
September 25 provisional milestone.

The conversion-time quality check evaluated 28,245 next-token predictions on a
paired text calibration set:

| Checkpoint | Mean NLL | Perplexity |
| --- | ---: | ---: |
| BF16 reference | 2.5476 | 12.7768 |
| W8A8 QDQ reference | 2.5046 | 12.2393 |

The lower value for the quantized reference on this small calibration set is not
evidence that quantization improves general model quality. Independent held-out
evaluation remains required.

## Planned comparative quality evaluation

### Preliminary five-item pipeline check

On September 25, 2026, the first five GPQA Diamond items were replayed serially
against the Ascend milestone using AISBench revision `19018e9c`, dataset version
`b1ed2c`, temperature 0, seed 1024, and the same 8,192-token output ceiling as
the RTX reference:

| Measurement | Ascend W8A8 | RTX UD-IQ4_XS |
| --- | ---: | ---: |
| Correct / attempted | 5 / 5 | 5 / 5 |
| Accuracy | 100% | 100% |
| Extractable final choices | 5 / 5 | 5 / 5 |
| Scored output tokens | 8,190 | 7,746 |

The Ascend run used client batch size 1 to match the server's single active
sequence and completed its inference phase in 538 seconds, including one
warmup. This deterministic first-five subset is a pipeline and qualitative
smoke check only. It is too small and non-random to estimate model quality or
claim an advantage over the RTX result. The full 198-item W8 replay remains
pending.

The planned W8 quality comparison will replay the exact September 23 RTX
6000 PRO `llama.cpp` GPQA Diamond configuration against this Ascend checkpoint:

| Setting | Frozen value |
| --- | --- |
| Harness | AISBench revision `19018e9c` |
| Dataset | GPQA Diamond version `b1ed2c`, 198 questions |
| Prompt | Zero-shot chain of thought, identical shuffled choices |
| Sampling | Temperature 0, seed 1024, top-p 1.0, top-k 20 |
| Output ceiling | 8,192 tokens |
| RTX UD-IQ4_XS reference | 138/198, 69.70% |
| Ascend W8A8 result | Pending |

The reference run produced 54 responses that exhausted the output ceiling
without an extractable final choice. Consequently, the comparison will report
raw accuracy, answered-only accuracy, cap-without-answer count, output-token
distribution, domain results, and item-level answer flips—not only the single
headline score. The frozen command and reporting gates are documented in the
[Ascend GPQA replay runbook](https://github.com/opensensor/vllm-ascend/blob/main/artifacts/qwen38-1m/ASCEND-GPQA-BENCHMARK.md).

## Provisional serving example

The command below records the September 26 text-only W8 benchmark arguments.
For image input, omit `--language-model-only` and use the image-enabled flags
in the image support section above. Keep the W8 loader and quantization policy. It requires
the selected development snapshot, its compiled 310P operators, and the
configured Ascend environment. A stock installation or the command alone does
not reproduce the measured performance. The measured launcher also uses
`VLLM_ASCEND_KV_CACHE_FRACTION=0.88`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`,
`TASK_QUEUE_ENABLE=1`, and `OMP_NUM_THREADS=1`. Worker CPU pools on this host
are `0-5`, `8-13`, `16-21`, and `24-29`; these core IDs are specific to its CPU
topology. The public integration revision is pinned above; the cold-prefill
path remains under study.

```bash
vllm serve /path/to/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i \
  --served-model-name qwen38-flash-next-w8a8 \
  --dtype float16 \
  --quantization ascend \
  --tensor-parallel-size 4 \
  --max-model-len 160000 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.965 \
  --disable-custom-all-reduce \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --mamba-cache-mode align \
  --enable-chunked-prefill \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --no-async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

## Known limitations

- Requires four Ascend 310P3 devices and the custom Qwen4Exp runtime described
  above; the measured snapshot is not a stock vLLM Ascend release.
- This four-chip deployment has a 160,000-token serving cap. The controlled
  speed study reached approximately 23K prompt tokens, and the later live
  log sample reached 50,233. Full 100K/160K correctness and two concurrent
  160K contexts remain unvalidated.
- Longer contexts supported by the underlying model, including 262,144-token
  and extended 1M-token operation, are not validated on this configuration.
- Image input is supported by the shared OpenSensor frontend. Independent
  image quality evaluation and video validation remain outstanding.
- MTP generation is functional in the provisional development build, but its
  item-level quality parity against target-only decoding is not yet measured.
- Accuracy has not yet been evaluated on an independent benchmark suite; the
  exact GPQA Diamond replay is prepared but pending.
- Performance evidence covers nine paired requests and one restored-server
  decode check. Broader workload, concurrency, and full-context benchmarks
  remain pending; approximately 19 tokens/s is not a universal decode rate.
- Cold prompt KV computation remains slower and more variable than desired.
- The public vLLM Ascend integration is pinned at `5bc6a28ff`; the performance
  values still describe the earlier September 26 `affinity` runtime snapshot.
- The current export is unusually large for a W8A8 model because substantial
  non-expert state, including the PLE n-gram table, remains FP16.

## Release checklist

- [x] Record base-model revision, derivative license, and conversion provenance.
- [x] Merge the 310P Qwen4Exp serving code and host-tested MTP integration.
- [x] Validate MTP loading and generation with the real checkpoint on four
  310P devices, including an ACL graph run.
- [ ] Run the no-MTP comparison and measure item-level quality parity.
- [x] Measure MTP acceptance and controlled single-request decode performance
  in the September 26 CPU-affinity study.
- [ ] Complete broader workload, full-context, and concurrency benchmarks.
- [x] Complete a serial five-item GPQA pipeline check with a valid prediction
  ledger and evaluation summary.
- [ ] Run the frozen GPQA Diamond replay and publish item-level diagnostics.
- [ ] Validate a full 100K/160K prompt and resolve the cold-prefill issue.
- [x] Complete the metadata scan, shard inventory, and checksum verification.
- [ ] Complete independent license review.
- [x] Upload and verify the weight shards on the Hub.

The detailed release gates are tracked separately. This public checkpoint
remains experimental until the outstanding validation is complete.

## Provenance

- Base model: `Qwen/Qwen3.8-Flash-Next`
- Base revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Base architecture: `Qwen4ExpForConditionalGeneration`
- Conversion method: Ascend ModelSlim `W8A8_DYNAMIC`, `ascend_v1` export
- Target: Atlas 300I Duo / Ascend 310P3
- Conversion-time source tensor bytes: `239392751608`
- Conversion-time exported tensor bytes: `239958982648`
- Export tensors: `222746`

The source revision was recovered from the conversion project record and
verified directly against the official Qwen repository. The checkpoint's model
geometry matches that revision.

## License

The model weights and derivative checkpoint are distributed under the
[Qwen Community License 1.0](LICENSE), inherited from the base model. This is
not GPL. In particular, review the license's attribution requirement and its
separate-license condition for certain commercial Model-as-a-Service and AI work
assistant uses.

The vLLM Ascend implementation code is separate from these weights and remains
under the license of its source repository.

## Citation

Please cite the original Qwen model and technical report. A release-specific
citation may be added after the repository name and maintainers are finalized.
