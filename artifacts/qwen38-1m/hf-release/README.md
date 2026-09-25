---
library_name: vllm
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
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
> This is a draft model card for a locally validated, experimental checkpoint.
> The checkpoint currently requires the public OpenSensor vLLM fork and the
> OpenSensor vLLM Ascend `main` branch; it is not yet a drop-in model for stock
> vLLM Ascend. Do not make this repository public until the release checklist
> is complete.

This repository contains an Ascend ModelSlim `W8A8_DYNAMIC` conversion of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
pinned at revision
[`de4b8e4d43b917e7706784d8bb445c9af86a3540`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540).
It targets two Atlas 300I Duo cards, exposed as four Ascend 310P3 devices, with
tensor parallelism across all four devices.

The checkpoint preserves the base model's Qwen4 experimental architecture:
48 decoder layers, 512 routed experts, hybrid Gated DeltaNet/Qwen Sparse
Attention, PLE n-gram embeddings, a vision encoder, and MTP weights. The current
Ascend serving path has only been exercised for text generation. Multimodal
execution is not claimed as supported, and MTP remains provisional pending the
quality comparison and final public runtime freeze.

## What is quantized

The conversion used Ascend ModelSlim's `W8A8_DYNAMIC` route:

- 73,728 routed-expert linear projections use INT8 weights with FP32
  per-output-channel scales and offsets.
- Activations are quantized dynamically by the Ascend execution path.
- Embeddings, PLE tables, attention, Gated DeltaNet, shared experts, vision
  components, and the language-model head remain FP16 where present.
- The export contains 254 Safetensors shards and is approximately 240.06 GB
  (223.57 GiB). It is not a uniformly INT8 checkpoint.

## September 25 provisional device milestone

Development build `qwen38-mtp-grouped-qsa-r21` moved the checkpoint well beyond
the initial smoke test. The following values were captured from server and
per-request logs on September 25, 2026. They are provisional engineering
measurements, not a controlled public benchmark:

| Measurement | Observed value |
| --- | ---: |
| Milestone serving limit (`--max-model-len`) | 131,072 tokens |
| Allocated NPU KV capacity | 142,237 tokens |
| TP ranks / active sequences | 4 / 1 |
| Weight memory per rank | 33.93 GiB |
| Peak activation memory per rank | 1.56 GiB |
| Decode graph memory per rank | 0.32 GiB |
| MTP draft depth | 1 token |
| Representative per-request decode | 14.8–16.0 tokens/s |
| 1,359-token sustained decode sample | 14.8 tokens/s |

This build used MTP speculative decoding, prefix caching, chunked prefill, and
full-decode-only graphs. Representative 201–328-token completions measured
15.3–16.0 tokens/s; a longer 1,359-token completion measured 14.8 tokens/s.
These are batch-one request timings rather than a multi-request throughput
benchmark.

The 131,072-token value is the serving limit selected for this four-chip
milestone (`--max-model-len`), **not the model's architectural maximum context**.
The configured limit and KV allocation show that a 131K-token request fits the
runtime's calculated memory budget. They do **not** by themselves establish
end-to-end 131K correctness. The longest request in the reviewed server-log
sample was 43,603 prompt tokens, but real KiloCode sessions have exceeded that
length; their exact token counts have not yet been recovered into the benchmark
evidence bundle. A controlled full 100K/131K prompt test remains a release gate.

Initial prompt processing also remains under active optimization. In this
build, uncached 24K-token prompts varied from 61.5 to 83.3 seconds
(approximately 291–394 effective prompt tokens/s), while cached continuations
avoided most of that work. Until the cold-prefill path is frozen, do not treat
these prompt timings as final performance claims.

The vLLM build was
[`opensensor/vllm@3ab5dda29`](https://github.com/opensensor/vllm/commit/3ab5dda29acabea01f6a63d0806bdbbb4a27bde5).
The corresponding vLLM Ascend performance work was still an experimental
staged snapshot; the final public `main` revision must be recorded after the
cold-prefill fix is merged.

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
the September 25 provisional milestone above.

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

The first public quality comparison will replay the exact September 23 RTX
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

The command below records the September 25 milestone configuration. It is not
the final release command because the Ascend performance snapshot and cold
prefill path have not yet been frozen on public `main`. It will not work with a
stock vLLM Ascend installation.

```bash
vllm serve /path/to/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i \
  --served-model-name qwen38-flash-next-w8a8 \
  --dtype float16 \
  --quantization ascend \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.965 \
  --disable-custom-all-reduce \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-chunked-prefill \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --no-async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

## Known limitations

- Requires four Ascend 310P3 devices and the Qwen4Exp support in the associated
  public vLLM fork and vLLM Ascend `main` revisions.
- This four-chip deployment has a 131,072-token serving cap; that cap is not
  the model's maximum context. The reviewed log sample reached 43,603 prompt
  tokens, while longer real KiloCode sessions have run successfully. A
  controlled 100K/131K request with retained artifacts is not yet validated.
- Longer contexts supported by the underlying model, including 262,144-token
  and extended 1M-token operation, are not validated on this configuration.
- Multimodal inputs are not validated.
- MTP generation is functional in the provisional development build, but its
  item-level quality parity against target-only decoding is not yet measured.
- Accuracy has not yet been evaluated on an independent benchmark suite; the
  exact GPQA Diamond replay is prepared but pending.
- Performance has not been measured with a controlled benchmark protocol.
- Cold prompt KV computation remains slower and more variable than desired.
- The final vLLM Ascend code revision corresponding to the performance
  milestone is not yet frozen on public `main`.
- The current export is unusually large for a W8A8 model because substantial
  non-expert state, including the PLE n-gram table, remains FP16.

## Release checklist

- [x] Record base-model revision, derivative license, and conversion provenance.
- [x] Merge the 310P Qwen4Exp serving code and host-tested MTP integration.
- [ ] Validate MTP loading and generation with the real checkpoint on four
  310P devices, including an ACL graph run and a no-MTP comparison.
- [ ] Measure MTP acceptance rate and controlled latency/throughput.
- [ ] Run the frozen GPQA Diamond replay and publish item-level diagnostics.
- [ ] Validate a full 100K/131K prompt and resolve the cold-prefill issue.
- [ ] Finish license review, metadata scan, shard inventory, and checksums.
- [ ] Upload and verify the weight shards. This private repository currently
  contains metadata files only.

The detailed publication gates are tracked separately. Checked code items do
not imply that the model is ready for public release.

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
