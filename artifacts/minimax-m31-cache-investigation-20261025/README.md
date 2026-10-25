# M3.1 cache and numerical repeatability investigation

No stale-KV defect was reproduced in these bounded tests. We did reproduce
operator nondeterminism, and retained deterministic operator mode after it made
cold/warm cache checks agree. **The original reasoning/tool loop still occurs.**
This is a repeatability improvement, not a demonstrated coherence fix.

## Controlled cache checks

Identical token IDs, greedy sampling, seed 71, fixed output caps and unchanged
weights were used throughout. All runs used the same six workers; there was no
restart or shard reload. Public admission was held during each bounded test and
restored afterward. Proposed commands from the loop replay were never executed.

The first prompt contains 1025 tokens. Its last token follows two full 512-token
prefill chunks, and a warm request reuses exactly 1024 cached tokens. Both cases
therefore evaluate that final prompt token through a single-token path. Nine
generated token IDs and their returned top-20 log probabilities were compared.
This does not compare the complete vocabulary distribution.

| Mode | Cold versus warm | After unrelated request | Prefix reset, old storage retained | Prefix and physical KV reset |
| --- | --- | --- | --- | --- |
| Retained partial graphs, default operators | Scores differ | Scores differ | Scores differ | Scores differ |
| Eager, default operators | Scores differ | Scores differ | Scores differ | Scores differ |
| Eager, deterministic router only | Scores differ | Scores differ | Scores differ | Scores differ |
| Eager, deterministic operators throughout | Exact | Exact | Exact | Exact |

All nine generated token IDs matched in these synthetic cases even when scores
changed. Cache hits were measured using per-request deltas of the server's
prefix-cache counters, not inferred from timing. Warm requests reused 1024
tokens; cold/reset requests reused zero. The repetitive raw completion is a
numerical probe, not a chat-quality assessment.

A second prompt contains 4097 tokens, spanning 33 pages with only 16 blocks
selected by sparse attention. With deterministic operators and partial graphs,
all five returned top-20 distributions matched exactly across cold, warm,
post-unrelated-request and zeroed-cache requests. Warm requests reused exactly
4096 tokens. Cold requests took about 33 seconds; warm requests about 1.41
seconds. No full-160K or original 29K-session cache test was run.

## Locating the first difference

The live storage audit found all 360 main-K, main-V and index-cache tensors
covered by `resident_reset` across six workers. No attention/index cache was
missing from the runner-owned storage traversed by reset.

Two zeroed-cache eager requests were traced on TP ranks 0 and 1 through layers
0–3, including both 512-token chunks and the final single token. Different
physical pages were allocated, but canonical main/index histories matched
exactly. Hidden inputs/outputs in the first three dense layers also matched.
The first hidden-state difference appeared after the first MoE, layer 3.

A finer trace found differences first in its FP32 router logits. Routing IDs
matched in that traced layer, but combination weights differed slightly. The
three expert projections and clipped activation matched on valid local rows;
uncomputed peer rows were excluded from the comparison. Tracing copies tensors
to CPU and can perturb scheduling; it is diagnostic evidence, not a performance
measurement or a complete trace of all 60 layers.

An isolated real-weight router probe then reproduced the issue on every chip:
identical inputs gave FP32 logit differences up to 1.90735e-6 across five calls.
With `torch.use_deterministic_algorithms(True)`, all five router outputs were
bitwise equal on all six chips, at both one and 512 input rows. A scoped router
setting also stabilized that isolated projection, but did not stabilize all
full-model output scores. Keeping the setting enabled across execution did.
The other contributing operators have not been individually isolated.

## What remains unresolved

Incremental decoding and cold recomputation of the same token sequence still
produce different scores, even with deterministic operators. At continuation
offsets 1, 2 and 7, maximum differences among common returned log-probability
entries were 0.21715, 0.51103 and 1.03775, respectively. The selected tokens
matched. Those tests change prefill/decode batch shapes and kernel paths; they
do not establish a cache-storage defect. Deterministic execution is not the same
as batch-invariant arithmetic. Prefill/decode parity is the next targeted check,
including expert activation quantization and other shape-dependent operations.

The 5549-token continuation from the original looping conversation still says
it will summarize, then issues the same discovery commands. It completed with
193 output tokens and `tool_calls`. Existing loop history can perpetuate the
pattern; this is not a fresh reproduction of its onset. Custom W4 conversion,
expert INT8 compute and broader model quality remain unresolved. No claim is
made that deterministic mode restores native source-model accuracy.

## Retained configuration and operational change

The service retains its six chips, selected dense INT8, expert INT8 v15,
activation v2, all 240 partial graph pieces and 163840-token context setting.
Operator determinism is now enabled with `warn_only=True`, matching the setting
used by Ascend's existing batch-invariance support. Unsupported operators may
warn; universal determinism is not promised. This change does not enable the
separate Triton/batch-invariant operator overrides or alter weight layout.

Three 64-token counting requests measured **3.87234, 3.86872 and 3.87177 tokens/s**
after the first output. The median is **3.87177**, about **2.18% below** the prior
3.95815 median. This modest measured cost is retained for repeatability. No
performance improvement is claimed.

`tools.minimax_m3.resident_determinism` verifies the exact serving source, drains
requests, releases graph captures, changes the flags on all workers and restores
the saved generation before resuming. It checks worker and weight-storage
identity. Failed or uncertain mutations leave the service paused. Flags persist
through subsequent ordinary resident reloads; the controller's `disable` action
restores the default false/false settings.

```bash
python -m tools.minimax_m3.resident_determinism enable \
  --serving-source /workspace/m31-startup-serving-source.py \
  --output /workspace/m31-determinism-enable.json
```

The installed post-start script now runs this after dense conversion. The live
transition was hardware-tested and shell syntax checked; the newly appended
startup step was not tested through another cold restart. The paired gate/up
candidate remains undeployed. Synthetic responses, compact traces and prototype
controllers are saved here; large CPU tensor traces remain on the serving host.
Prototype `.py.txt` files document this session's fixed paths and are not general
production entry points.

## Validation

The M3 CPU suite passed **266 tests**, with 24 dependency/hardware skips. New
controller tests cover graph-release ordering, source rejection, worker failures
and fail-closed transitions. Hardware evidence includes the six-chip flag
transition, unchanged worker/storage identity, cache comparisons, loop replay,
graph recapture and decode measurements.

Changed-file hooks passed. Required whole-repository `bash format.sh ci` ran in
an isolated worktree and failed on preexisting unrelated violations; see
`whole-tree-format.json`. Its automatic edits were not included.
