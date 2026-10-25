# M3.1 independent history and activation investigation

The reasoning/tool loop is **not fixed**. Source-preserving experts, local
operator agreement, and improved logit agreement have not yet established
successful multi-turn behavior. These are diagnostics, not a promoted serving
profile or a performance improvement.

## Independent history

`tools/minimax_m3/history_reference.py` independently reads the original mixed
NVFP4/MXFP8 checkpoint, propagates every prompt token through all 60 layers,
and computes sparse attention against its own keys and values. It imports no
Ascend model or operator code. Its attention calculation masks chronological
keys rather than using the serving implementation's gather path. Original QAT
activation/cache conventions are not implied by its default FP16 convention.

The first complete 5,549-token pass took 1,112 seconds on CUDA with streamed
source weights. Its final logits differed from the captured Ascend logits by
7.60% normalized RMS error, but both selected the opening reasoning token.
Their probabilities for that token were approximately 0.775 and 0.769;
KL divergence was only 0.0000858. This is not evidence that the observed logit
error caused the loop.

That first pass constructed rotary frequencies with negative powers on CUDA.
The source/HF reciprocal-power construction on CPU produces slightly different
FP16 cache entries (2,597 of 355,200 entries, maximum difference 0.0004883).
The second full pass uses the canonical CPU construction and completed in
1,093 seconds: final-logit normalized RMS error is 6.56%, with the same greedy
first token. The precision candidate is 5.32% from this second reference and
matches its top-10 ranking. Neither result establishes behavioral correctness. The first pass's
selection differences must not be presented as proven device faults.

The tool supports validated checkpoint resume and token-suffix extension of
its own complete prefix cache. Tests compare resumed and uninterrupted passes,
check cached-prefix preservation, reject mismatched reference identities, and
compare incremental attention with full-history attention. Query batching can
change floating-point accumulation at FP16 rounding boundaries.

## Independent continuation comparison

A batched suffix pass consumed the 118 preceding generated tokens from the
precision-candidate response, using only independently computed prefix and suffix
states. **118 of 119 greedy next-token choices match** the captured response.
The first mismatch is offset 58: the independent reference prefers "header"
(probability 0.4155), while the device chose "head" (0.3965), in a tool description.
All preceding repeated reasoning and the complete `head -60 README.md` command
match. The 60-layer suffix calculation took 919.6 seconds before the final head.
See [conditional comparisons](teacher-comparison.json).

This is teacher forcing: after the first mismatch, later rows are conditional
comparisons, not a freely generated trajectory. Even the matching prefix uses
batched query arithmetic, which can round differently from incremental decode.
A separate incremental reference generated only the opening thinking token and
"I" before being stopped in favor of the more efficient suffix comparison.
Neither experiment proves a fresh multi-turn agent will finish. The results
weaken a device-specific kernel explanation for this continuation, while leaving
shared FP16 conventions, source attention/cache arithmetic, prompt/history
interaction and onset of the loop unresolved.

## Resident candidates

The normalization/rotary candidate keeps Gemma's `1 + weight` in FP32 until
output conversion and uses explicit FP16 rotary multiply/add steps. It reduced
final-logit error against the first reference from 7.60% to 5.59% and matched
its top-10 ranking. This comparison uses the first reference's rotary convention;
it is not proof of source equivalence or a behavioral fix.

The activation candidate applies NVFP4 rounding before expert projections and
MXFP8 rounding before dense projections. Its portable transforms exactly matched
native NVIDIA quantizers on 1,695,744 real/synthetic values per format. After
startup recovery, all six NPU ranks also matched those expected values exactly.
All 21,888 source input-global scales in the 342 expert banks were 1.0, as
required by the current diagnostic NVFP4 transform. KV/index-cache quantization
and BF16 output conventions remain outside this trial.

The activation-only late-history replay completed with HTTP 200, 119 generated
tokens, and two additional discovery calls. It again promised to summarize,
then asked for README and Git information. This does **not** pass the loop gate.
The diagnostic took 460.6 seconds, including 258.6 seconds to first output;
its slow reference quantizers are not production performance candidates.

The precision-only replay likewise returned 119 tokens and more README/Git
inspection. All 119 raw generated token IDs were captured: the decoded model
text contains the repeated reasoning and both tool commands, and JSON-decoding
the API arguments preserves those commands. This particular continuation was
not created by a tool parser or the UI.

Two additional attention candidates passed the six-rank synthetic device gate: fused FP32 normalization/rotary with identity-scaled FP8
K/V and index Q/K, and a full-attention ablation with eight-query tiles. The
first still differs from a complete BF16/Hopper execution. The second changes
the trained sparse attention pattern and is only an isolation experiment.
The [NVIDIA fused implementation](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu)
and [Hopper recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash)
provide the comparison target; native M3 QK operators were unavailable in the
local CUDA wheel, so no native fused-QK equivalence claim is made.

As a bounded counterfactual, FP8 rounding of the observed index queries and
history keys changed 57 of 240 selected block sets (62 of 3,840 selected blocks;
at most two replacements in one set). The original FP16 selections were first
revalidated exactly. This uses observed histories, not independently propagated
FP8 histories, and does not establish a cause for the loop.

## Fused residual normalization

A subsequent audit found another convention shared by the earlier device and
independent reference: both rounded the residual sum to FP16 before normalizing.
The [native FlashInfer kernel](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/norm.cuh)
normalizes its FP32 sum while separately storing a rounded residual. The new
`fused_gemma_forward` candidate preserves that distinction; the earlier candidate
and its recorded results are unchanged.

On 238 real recorded input/residual pairs (1,462,272 values), the new CUDA tensor
candidate differs from the native CUDA operation in 82 values, versus 378,918
for the earlier convention. Peak row normalized RMS error is 0.00233% versus
0.05937%; all stored residuals match exactly. See the
[conditional native comparison](fused-residual-cuda-gate.json). This does not
establish a loop fix or complete native all-reduce/BF16 equivalence. The candidate
is included in the FP8-attention replay candidate. Its held-history replay still
promised a summary and requested README/Git information: 119 generated tokens,
468.8 seconds, no final answer. See [the replay receipt](fp8-attention-replay.json).

The full-attention ablation also continued that pattern, returning 126 tokens
and two README/Git calls in 507.4 seconds
([receipt](full-attention-replay.json)). Both experiments preserved worker PIDs
and weight storage and restored baseline dispatch afterward. These negative
continuations do not establish whether the changes prevent a loop from starting
in a fresh conversation, nor does full attention reproduce the trained model.

## Fresh Kilo behavior

The combined source-weight, activation-rounding and earlier normalization/rotary
candidate completed eight responses and **16 successful tool calls without a
project summary**. It reached the external eight-response guard after 4,066
seconds. The tool calls were not exact duplicates, but later reasoning repeatedly
returned to inspecting the fork, including "Actually, I have enough. But let me
check a couple more things". This is continued over-investigation and does not
qualify as a demonstrated loop fix. It does not prove that the model would never
finish with a larger step budget. See [the complete bounded receipt](combined-fresh-kilo.json).

Every completed Bash tool had exit status zero; read tools returned their files.
The seven completed assistant/tool groups in the final served request retained
both tool-call IDs in their original order. No forced final answer was injected.
This was one seed (71), temperature 1, top-p 0.95, the original pre-mitigation
agent prompt, and the 32K diagnostic service. The first request reused an earlier
payload; only the global config path and user environment timestamp changed on
request two. This limits strict cross-run comparisons and reduced prefix reuse.

Three preceding attempts stalled in response delivery despite completed engine
work. A captured SSH connection had outstanding retransmissions after a long
idle period. The successful diagnostic transport used SSH keepalives and an
on-host SSE client. The responsible network component was not identified. Those
three attempts are excluded from model-quality results; the user's normal Kilo
configuration was not changed by this transport workaround.

## Attention and fused normalization device gate

All six ranks produced exact FP8 K/V and index Q/K results on a synthetic
four-token projection fixture. Main Q differed in one of 16,384 values, with
maximum absolute error 0.00000191. On 1,462,272 real recorded normalization
values per rank, 74 differed from native CUDA; normalized RMS error was
0.000384%, stored residuals matched exactly, and no nonfinite values appeared.
The full-attention ablation also produced the expected visible-token mean.
See [device receipts](attention-npu-gate.json). These are conditional operator
tests, not a complete native attention implementation or coherence validation.

The first gate attempt caught an old cached Python helper signature. Corrected
resident exports embed the quantizer source rather than importing that stale
worker module. All six error acknowledgments were collected, the baseline was
restored, and the corrected gate passed without a shard reload. Use the frozen
self-contained hooks for reproducing these resident experiments.

## Operator-resource failure and recovery

Introducing a previously unused CANN power operation after loading source banks
failed allocation on rank 3. Subsequent diagnostic casts on that rank returned
incorrect values; restoring Python dispatch did not recover the device context.
This failure occurred during this experiment, after the earlier loop reproductions.
It does not explain the original incident.

One cold restart reserved the activation operators before expanding the source
banks. The loader now offers an explicit, default-off
`activation_reference_warmup` option, allowed only with source NVFP4 banks. It
checks both transforms and a device cast before loading weights and fails startup
on disagreement. The fresh six-rank native-fixture gate passed with no nonfinite
values. Later candidate switches use resident reload and preserve worker PIDs
and weight storage; they do not reload shards.

## Reproduction and limits

Run the independent reference against a private tokenized conversation:

```bash
python3 -m tools.minimax_m3.history_reference \
  --model /path/to/original/MiniMax-M3.1-Flash \
  --tokens /private/tokenize.json --output /private/reference \
  --device cuda:0 --save-kv
```

`--observed` optionally compares the six rank traces from the preceding
[runtime investigation](../minimax-m31-runtime-trace-20261025/README.md).
`--prefix-cache` extends a completed independent prefix by a nonempty token suffix; the input
must contain that exact prefix followed by the new tokens. `--tail-logits`
selects how many trailing prediction rows to save. `--activation-format
nvfp4-mxfp8` enables the portable activation convention, not original quantized KV.
Private conversations, tokens, model weights, and activation tensors are excluded
from this artifact. Numeric receipts and diagnostic controller sources are kept.

The service remains a source-weight, text-only, 32K eager diagnostic on port
8000, with TP2/PP3/EP2. The standard optimized 160K W4 launcher is unchanged.
Graphs, DSpark, multimodal inference, 128K/batch-16 capacity and performance
qualification are outside this correctness experiment. Tokenizer, chat template,
and generation settings match the original release byte-for-byte; the model
configuration changes only custom-loader/quantization metadata and BF16 to FP16
([configuration audit](serving-config-audit.json)). No model quality
qualification is claimed from HTTP success alone.

## Independent serving reference in preparation

An unmodified [Unsloth llama.cpp revision](https://github.com/unslothai/llama.cpp/tree/822be707b23345b291f3042defa7508aa813cc4b)
was built on the Threadripper host with CPU support. Its converter's
`--outtype auto` path preserves the released NVFP4 expert codes/scales, writes
MXFP8 dense weights as BF16, and forms normalization weights in FP32. Conversion
to 14 GGUF shards is running directly onto the host's NVMe. This avoids another
signed INT4 expert quantization. No native-backend behavior result is available
at this checkpoint; llama.cpp is an independent implementation, not a claim of
equivalence to the hosted BF16/QAT execution.

The focused suite passes 86 tests, including a regression requiring failed
activation warmup to stop before source-bank loading. The whole-tree formatter
still reports unrelated existing failures; the scoped manual hooks pass.
See [validation](validation.json) and [whole-tree results](whole-tree-format.json).
