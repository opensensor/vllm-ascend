# M3.1 quantization and reasoning-loop evidence

Research date: October 25, 2026. No external code was executed, no model was
loaded, and no service configuration was changed.

## Does the VQ warning contain a demonstrated loop reproduction?

### Takeaway

The warning is relevant but weaker evidence than a source-matched failure
reproduction. The published artifacts quantify conversion loss and runtime
agreement; they do not establish a repeated-tool-loop cause or fix.

### Cited Findings

- The approximately 2-bit VQ release warns about reasoning loops, damaged text
  and weaker answers. Its limitations explicitly say long free generation and
  tool use have not been measured. Thus its warning must not be represented as
  a controlled agent-loop experiment. [Model card](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF)
- The published source-reference generation samples are four short prompts:
  capitals, a prime-number function and a sky explanation. The function sample
  contains only initial reasoning; there is no repeated loop in these samples.
  The generation script limits its intended context below 2,048 tokens, where
  every visible block fits in attention. [Samples](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF/blob/main/quality/reference_generation_samples.txt),
  [generation script](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF/blob/main/tools/gen_mm31.py)
- The HF discussions API returned zero discussions. PR 110 comments contained
  automated review exchanges, with no user-supplied looping reproduction.
  The PR was merged October 25; the model card still describing it as a draft is
  stale. [Community API](https://huggingface.co/api/models/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF/discussions),
  [PR 110](https://github.com/Baekpica/ds4-dfm-rs/pull/110)

### Inferences

- The VQ warning supports caution about aggressive conversion, but it cannot
  tell us whether our signed W4 conversion causes the observed tool loop.

### Gaps

- No paired original/VQ multi-turn free-generation trace or quantified loop
  rate was found in the reviewed artifacts. No demonstrated sampling or
  quantization change that fixes such a loop was found there.

## What do the numerical receipts actually establish?

### Takeaway

Two different comparisons must be kept separate: converted bytes versus the
original, and a CUDA runtime versus a reference using those same converted
bytes. The latter cannot clear conversion loss.

### Cited Findings

- The original-versus-VQ receipt covers 60,431 positions, of which 12,191 are
  assistant positions. Assistant KL is 0.150019 and top-1 agreement 88.229%; all
  positions give KL 1.98225 and top-1 58.568%. These are teacher-forced fidelity
  metrics, not an agent completion score. [Raw receipt](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF/blob/main/quality/release-final/result.json)
- Original versus itself with different batching has KL 0.564790 overall and
  0.002902 on assistant positions. This is why the author separates assistant
  positions from prompt/schema/user text. [Noise floor](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF/blob/main/quality/noise-floor.json)
- The runtime quality code explicitly compares the engine with a reference
  over the same VQ bytes and simulated NVFP4 caches. It gates assistant top-1
  at 95% and relative NLL difference at 2%. Therefore PR 110's 96.15% runtime
  agreement is not 96.15% agreement with the original checkpoint.
  [Quality harness](https://github.com/Baekpica/ds4-dfm-rs/blob/main/crates/ds4-cli/examples/minimax_quality.rs),
  [PR results](https://github.com/Baekpica/ds4-dfm-rs/pull/110)
- The recipe uses calibrated eight-dimensional codebooks and residual codes
  on selected layers; it is not uniform signed INT4. [Recipe](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF/blob/main/RECIPE.md)

### Inferences

- Our single-token CPU KL 0.532 and four-expert output errors cannot be ranked
  against this held-out assistant-token evaluation. The meaningful transferable
  lesson is to measure conversion loss separately from runtime error, using
  assistant continuations and a same-source batching control.

### Gaps

- These metrics do not identify a loop threshold. The quantized model even has
  a lower assistant NLL than its teacher in the receipt; that does not establish
  superior task quality or absence of free-generation failures.

## Is there a source-preserving compact alternative?

### Takeaway

Yes, a reviewed external converter preserves the original routed-expert FP4
codes and fine-grained scales in compact storage. This is an actionable format
reference, not proof that its implementation fits or runs efficiently on 310P.

### Cited Findings

- Unsloth llama.cpp PR 320 offers native NVFP4 expert packing and exact
  dequantization modes. The author reports byte-exact expert codes/block scales
  on fixtures and 31/31 full-checkpoint spot checks. The PR remains open.
  [PR 320](https://github.com/unslothai/llama.cpp/pull/320)
- Inspected head `822be707b23345b291f3042defa7508aa813cc4b`,
  `conversion/minimax.py`: `_dequant_nvfp4_eager` uses signed E2M1 levels
  `0, 0.5, 1, 1.5, 2, 3, 4, 6`, multiplying by an E4M3 scale per 16 values
  and dividing by `weight_global_scale`. `_nvfp4_pack_np` rearranges original
  nibbles/scales, while the expert stack stores `1 / global_scale` separately.
  It does not fit a shared 32-by-32 scale or remap weights onto uniform INT4.
  [Pinned converter](https://github.com/unslothai/llama.cpp/blob/822be707b23345b291f3042defa7508aa813cc4b/conversion/minimax.py#L205)
- Important qualification: the converter discards trained `input_global_scale`
  because llama.cpp computes activation scaling itself. Exact stored weights
  therefore do not imply identical original W4A4 arithmetic.
  [Pinned activation-scale handling](https://github.com/unslothai/llama.cpp/blob/822be707b23345b291f3042defa7508aa813cc4b/conversion/minimax.py#L313)
- TensorSharp's benchmark describes UD-Q4_K_XL as NVFP4 routed experts, Q8_0
  dense/shared/attention, F32 indexer/router and BF16 embeddings/head. Its small
  file-reading/calculator task completed successfully on both engines in all
  three reported repeats. This is a useful positive smoke test, not long agent
  coherence validation. [Benchmark and configuration](https://github.com/zhongkaifu/TensorSharp/blob/main/docs/perf/minimax-m3-vs-llama-cpp.md)
- The Unsloth community's current loading report concerns missing tensor
  support in stock llama.cpp; no loop is reported in that thread.
  [Loading issue](https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF/discussions/3)

### Inferences

- Source FP4 uses a nonlinear codebook. Our conversion can avoid its current
  requantization loss by retaining those codes and per-16 scales, then decoding
  them in an appropriate operator. This requires a fresh memory/performance
  assessment and exact tensor comparisons.
- Mathematical observation, not a tested kernel: doubling the E2M1 values gives
  signed integers `0, 1, 2, 3, 4, 6, 8, 12`, all representable in INT8. A nibble
  lookup can therefore expand the source codes to INT8 without changing their
  represented values if the scale is halved. Fine-grained per-16 scaling must
  still be applied correctly; ordinary single-scale INT8 GEMM is not equivalent.
  This removes a weight-representation obstacle, not activation-quantization
  error, accumulation differences or the need for a new kernel.

### Gaps

- No 310P memory budget, throughput result or multi-turn coherence result exists
  for this proposed source-preserving path. No independent conventional W4
  quantization report proving the same loop and its fix was located.
