# M3.1 reasoning/tool-loop investigation

The reported loop is reproduced, but its cause is not established. Neither
sampling changes nor completion guidance is a demonstrated fix. Do not treat
this investigation as a quality qualification of custom W4 or the INT8 kernels.

## Session and protocol evidence

The original Kilo 7.8.17 session completed 19 assistant tool turns and 38 tool
results, growing from 7727 to 29075 prompt tokens. Completed turns ended with
`tool_calls`, not an output-token limit. The twentieth turn was interrupted.
The model repeatedly promised to summarize while asking similar repository
questions. Private session contents and full request payloads are not committed.

A synthetic two-request wire test through the actual Kilo CLI confirmed that
assistant `reasoning_content`, tool-call IDs and tool results survive the next
request. The server's pinned protocol normalization and template also preserve
reasoning history. No missing-history defect was identified by these checks.

The MiniMax-only local Kilo settings now use temperature 1.0 and top-p 0.95,
matching the [model's published guidance](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/README.md).
The code agent's stale 16K text is corrected to 163840 tokens, with instructions
to finish an overview once evidence is sufficient. The wire probe confirms the
new sampling values. Other providers, permissions and tools are unchanged; no
arbitrary step cap was added. Restart Kilo/open a fresh session to load settings.
`kilo-minimax.json` is a sanitized config fragment, not a replacement for the
user's full configuration. `kilo config check` passes.

## Bounded behavior checks

The frozen replay uses the last three completed tool turns from the session,
5549 input tokens, seed 71 and at most 256 output tokens. Proposed tool calls
are recorded, never executed. This is a bounded continuation, not a complete
replay of the original 29K-token trajectory.

| Case | Result |
| --- | --- |
| Retained graphs, dense INT8, temperature 0 | Repeated the reported reasoning and discovery commands; `tool_calls` |
| Same history, temperature 1/top-p 0.95 | Different discovery requests; still `tool_calls` |
| Completion guidance plus cached README evidence | Still requested repository discovery; `tool_calls` |
| Eager, dense INT8, temperature 0 | Similar repeated intention to summarize followed by discovery; `tool_calls` |
| Eager, attention tile 8, dense INT8 | Exact reported repeated reasoning and discovery commands; `tool_calls` |
| Eager, attention tile 8, dense FP16 | Again promised to summarize, then requested README/remotes/branches; `tool_calls` |

The matched tile-8 runs used the same 5549-token prompt, seed, temperature and
output cap, changing only the selected dense projections. INT8 produced 193
completion tokens; FP16 produced 119. Both completed successfully. This checks
one continuation from existing loop history, not a fresh full agent trajectory.

The loop-like continuation occurs without graphs and with dense FP16. Neither
optimization is necessary for this continuation, but the experiment does not
prove numerical equivalence or exclude a contribution to its onset. The proposed paired gate/up kernel was never loaded
and cannot explain the original incident.

## FP16 diagnostic memory failure and recovery

Dense FP16 restoration initially failed on rank 3 while five ranks succeeded.
The controller kept serving paused throughout the mixed state. Synchronizing
between projections alone did not resolve the allocation failure. Clearing
rebuildable INT8 kernel scale metadata then allowed every rank to restore its
exact CPU-backed dense weights without changing worker PIDs or expert shards.

The subsequent 5549-token request nevertheless failed allocating 66 MiB in
sparse-attention scratch. That run has no valid output comparison. The distributed
request stalled and required a service restart; resident reload cannot recover
workers blocked in an unfinished collective. The restart uses the existing
qualified startup profile, including selected dense INT8 and partial graphs.

The conversion controller now clears rebuildable W4A8 metadata after draining
requests and releasing graphs. Restoration retires old INT8 allocations and
format-cast scratch between projections. Tests cover the cleanup ordering and
resuming restoration after a partial allocation failure. This addresses
transition headroom; it does not promise that dense FP16 fits every workload.

After restart, the matched A/B used eight-query attention tiles instead of 64
in both modes to reduce temporary storage. Both requests completed. The updated
controller restored FP16 and re-enabled all 348 dense INT8 projections on the
same six workers, then reinstated the normal tile-64 partial-graph generation.
The smaller tiles are diagnostic only and are not persisted for serving.
`dense-round-trip.json` records the successful hardware transition. Custom W4
expert storage and v15 expert INT8 compute were unchanged in both runs.

## Custom W4 remains an open suspect

Recomputing metrics from historical October 24 single-token CPU traces gives
logit cosine 0.85849, normalized RMS error 0.51366 and different selected expert
sets in 20 of 57 MoE layers between decoded source NVFP4 weights and custom W4.
Trace hashes and the changed layer IDs are in `historical-w4-drift.json`.

This was an untemplated `Hello` token, with FP16 activations; it excluded sparse
attention/QK/RoPE and the source model's activation quantization. It is neither
native source-model inference nor evidence that quantization caused this agent
loop. It does show why small isolated kernel errors and simple coherence checks
cannot clear the weight conversion of responsibility. A matched source-format
reference remains necessary to attribute a quality regression to custom W4.

## Retained service

The server is active and unpaused, with maintenance disabled, all 240 graph
pieces captured, no disabled layers and the original 163840-token context
setting. A fresh synthetic tool-result continuation answered `FR-PAR` and
finished without another tool call. This confirms that tool completion works;
it is not a fix for the original long agent trajectory.

One post-recovery 64-token counting request measured **3.927 tokens/s** after the
first token, near the prior 3.958 median. This is a single recovery measurement,
not a performance improvement claim. No full-160K test was run. The offline
paired gate/up performance candidate remains uncompiled and undeployed.

## Validation

The M3 CPU suite passed 260 tests, with 24 dependency/hardware skips. The required
whole-repository `bash format.sh ci` ran in an isolated worktree and failed on
existing unrelated formatting, spelling and import issues. Those changes were
not included. See `whole-tree-format.json` for failed hooks.

All changed-file hooks passed. Full traces and private replay payloads remain
on the working hosts; compact responses and transition receipts are included.
