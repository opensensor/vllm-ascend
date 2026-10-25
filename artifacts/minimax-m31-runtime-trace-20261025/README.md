# M3.1 production runtime trace — October 25, 2026

The reasoning loop is **not fixed**. This experiment checks arithmetic and
transport in the running source-preserving diagnostic service; it is not a
successful coherence test or a validation of the complete original QAT path.

## Results

[Numeric summary](summary.json), [all arithmetic comparisons](comparison.json),
[transport/cache checks](transport-comparison.json),
[embedding checks](embedding-comparison.json), and
[capture receipt](capture-receipt.json) contain the measured evidence.

| Check | Result |
| --- | --- |
| Layer/phase/rank records | 240, covering all 60 layers |
| Sparse block sets | 480/480 match |
| Routed expert IDs | 228/228 match |
| Largest local operation normalized RMS error | 0.04294%, layer 57 decode index-Q projection/norm/rotary |
| Largest complete MoE normalized RMS error | 0.02774% |
| Largest complete dense MLP normalized RMS error | 0.00588% |
| Largest attention projection/TP sum normalized RMS error | 0.00377% |
| Final-logit top-10 ranking | Exact for both phases and both final-stage TP ranks |
| Final-logit normalized RMS error | At most 0.00195% |
| Cache append/prefix, layer transport, tensor finiteness | All recorded checks pass |
| Source embedding and greedy next-token embedding | Exact |

These results argue against a gross arithmetic, cache-append, or transport fault
at the captured positions in the source-preserving trial. They do **not** clear
the optimized W4 service or establish correct full-history inference. The next
discriminating test is an independently propagated history/reference execution,
including the source activation/cache conventions, followed by the actual
multi-turn loop regression. Numerical agreement alone is not a loop fix.

## Experiment

Replay the private 5,549-token failing conversation with temperature 0, seed 71,
and two completion tokens. Capture the final prompt token (position 5,548) and
first decode token (position 5,549) across all 60 layers, both tensor-parallel
ranks, and all three pipeline stages. The final prefill batch has 429 tokens.
This exercises actual production operators with real checkpoint weights.

The service uses the source-preserving expert banks from
[the preceding investigation](../minimax-m31-source-nvfp4-20261025/README.md),
FP16 dense compute and activations, eager execution, and 32K context. It retains
the existing workers and weight storage through resident hot reload. Observer
hooks copy selected outputs to CPU but leave operator math unchanged. The
controller clears caches before replay and restores the original dispatch after
capture. Capture overhead makes this unsuitable as a performance benchmark.

The CPU comparison independently reads the original source checkpoint and
calculates projection, normalization, rotary, routing, expert and dense MLP
outputs. It also compares sparse index scores, selected block sets, attention
outputs, tensor-parallel reductions, layer/pipeline transfers, cache appends,
and final normalization/logits where captured.

## Limits

- Comparisons use FP16 activation/rounding conventions, not the original
  activation and KV quantization. Passing them cannot establish source-model
  equivalence.
- Historical cached keys/values come from the running server. The experiment
  does not independently recompute every historical token. Prefix-preservation
  checks establish only that the measured decode append preserves prior values.
- Each operation consumes observed production inputs to isolate local errors.
  Small per-operation errors can accumulate; this is not a full independently
  propagated reference or a quality benchmark.
- The trace covers two token positions in one conversation. It does not rule
  out batch-, position-, request-lifecycle-, or generation-dependent faults.
- Private conversations and activation tensors are excluded from Git. The
  numeric comparison and frozen scripts are the reproducibility record.

## Source comparison

The supported Hopper recipe uses FP8 KV/index caches and Marlin experts,
without DSpark enabled. Therefore absence of DSpark alone does not establish
why our implementation loops. The supported Blackwell recipe uses native
NVFP4 experts and quantized KV/index paths. These are meaningful differences
from our FP16 diagnostic runtime, not evidence that either difference causes
the loop. See the [official vLLM recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash).

The pinned [Unsloth implementation](https://github.com/unslothai/llama.cpp/blob/822be707b23345b291f3042defa7508aa813cc4b/src/models/minimax-m3.cpp)
was also inspected: per-head Gemma normalization, partial rotary, max-reduced
block scoring, local-block forcing, sigmoid routing, and routed/shared expert
composition are consistent at the algorithm level. This code review is not an
execution comparison against that implementation.

## Reproduction

`capture.py.txt`, `controller.py.txt`, and `compare.py.txt` preserve the exact
experiment scripts. Copy them to the paths named by the controller and run
with the serving environment. The controller requires exclusive ownership of
the resident lock and the private replay fixture; it is not a general serving
command. Copy rank traces to the comparison script's local trace directory,
then run the comparison with CPU PyTorch and safetensors. The source checkpoint
and paths are explicit in the scripts. See the preceding investigation for
startup, source identity, memory requirements, and feature limitations.

## Validation scope

The live trace completed with HTTP 200 and 240 records; all six worker PIDs
and weight-storage identities remained unchanged. The service is unpaused and
out of maintenance after restoring baseline dispatch. The transport checks
passed for all 240 records, including exact cross-stage transfers and
preservation of cached prefixes. Source embedding rows match both captured
positions, including the greedy token selected from the prefill logits.

The all-layer CPU arithmetic comparison was resumed after 48 completed records
with per-layer weight caching to avoid repeated source decoding. The reference
formulas did not change. `resume.json` is optional input containing previously
completed result rows; omit it for a complete fresh comparison. Runtime service
code was not changed by this investigation.

Scoped manual pre-commit checks pass. `bash format.sh ci` was also run in an
isolated checkout; whole-tree checks fail on existing Ruff, spelling, formatting,
Markdown, and forbidden-import issues outside this artifact. Automatic edits in
that disposable checkout were not copied back. No runtime functionality or
performance fix is claimed by this diagnostic-only change.
