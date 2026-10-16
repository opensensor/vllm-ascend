# TP6 mixed scheduling and private HC graphs

This experiment targets mixed prefill/decode on the six-device 310P resident-v8
server. The original scheduling policy remains active; the 256-token candidate is not
retained because quality qualification is incomplete. The existing shared-RAM
prefill PLE path stays enabled. Existing target
decode uses `FULL_DECODE_ONLY` graphs; mixed batches use eager execution. The
frozen runtime has synchronous scheduling and a one-entry batch queue. Do not
infer its configuration from newer scheduler code in the checkout.

## Scheduling experiment

The original policy targets 200 ms, starts at 128 tokens, allows 128–640 tokens,
aligns to 128, smooths with 0.25, and inserts eight decode-only steps between
mixed prefill chunks. Measured mixed chunks remained at the 128-token floor.
The candidate fixes the mixed chunk at 256 tokens and retains eight decode-only
steps. This trades longer individual mixed steps for fewer interruptions.
It does not add concurrent device execution or change pure-decode graphs.

A ten-trial screening sequence tested decode cadence 8, 4, 2, and 0, and the
256-token candidate, in forward/reverse order. Reducing the decode-only cadence
helped prefill but substantially reduced decoder progress during prefill.
The retained cadence is eight. A longer ABBA confirmation used fresh uncached
9715–9720-token prefills while a 2048-token synthetic decoder remained active:

| Measurement | Original policy | Fixed 256, cadence 8 |
| --- | ---: | ---: |
| Prefill TTFT, trial 1 | 88.435 s | 51.922 s |
| Prefill TTFT, trial 2 | 89.325 s | 52.115 s |
| Largest decoder content gap, trial 1 | 1.164 s | 1.163 s |
| Largest decoder content gap, trial 2 | 1.133 s | 1.176 s |

Median prefill TTFT decreased 41.5%. These are streaming content-update gaps,
not individual token latencies: speculative updates can contain several tokens.
Forced-length synthetic responses sometimes degenerated and acceptance varied,
so decoder throughput from this sequence is not a reliable optimization claim.
The user reported coherent recent Kilo sessions. A separate normal-stop smoke
also produced one repetitive response with the original scheduling and no new
graph binding; its cause is unresolved. Six subsequent normal-stop 512-token
responses were coherent across graph/eager/graph phases. This does not establish
a general serving-quality regression or prove comprehensive model accuracy.

## Normal-stop code-review confirmation

A second ABBA sequence used a 14,444-token code-review decoder, including
14,336 cached prompt tokens, and a fresh 4339–4342-token scheduler review.
No minimum output length or EOS suppression was used. The decoder could produce
up to 768 tokens; the new review could produce 64. Inputs were repository source,
not repeated filler or user conversations.

| Measurement | Original policy | Fixed 256, cadence 8 |
| --- | ---: | ---: |
| Prefill TTFT, trial 1 | 39.959 s | 24.392 s |
| Prefill TTFT, trial 2 | 40.130 s | 24.314 s |
| Largest decoder content gap, trial 1 | 0.979 s | 0.999 s |
| Largest decoder content gap, trial 2 | 0.649 s | 0.975 s |

The first three decoders degenerated; the final baseline decoder was coherent,
but its new prefill response degenerated. That last decoder finished before
prefill produced its first content, so it does not have complete overlap.
All four new prefills reported zero cached prompt tokens. These timings support
further investigation of 256-token chunks but do not establish a qualified
end-to-end improvement. The baseline is restored. This reproducer is distinct
from the user's recent coherent Kilo sessions, and its root cause is unresolved.

A final isolated check used the same source context with a shorter review
question, fresh leading nonce per execution mode, and 128 generated tokens.
All four responses were coherent: uncached/cached with existing target graphs,
then uncached/cached with target eager execution. Uncached prompts contained
14,450/14,454 tokens; cached requests reused 14,336 tokens. This did not reproduce
the failure, but the shorter outputs and changed question prevent ruling out
context length, prefix state, or execution mode as contributing factors.

## Private mixed HC graph candidates

Both experiments use real loaded weights, six ranks, modules 0, 49, and 97 out
of 98 HC modules, and three input scales. Shapes 131/134 cover a 128-token
prefill plus one/two three-token decoders. Shapes 259/262 additionally exercise
the first module for 256-token prefills. Each experiment checks 144 cases.

The first captures HC mixing and clones its outputs for independent lifetime.
All intermediate outputs were exact, but two complete residual-combine cases
were not. The second also captures the existing separate injection projection
and sigmoid, returning the small injection tensor instead of the large
normalized tensor. All its intermediate outputs were exact, but three complete
combine cases differed, with maximum absolute errors 0.001953125–0.00390625.
Two of those also showed differences between repeated unchanged eager calls.
The cause is unresolved; neither candidate is qualified or bound into serving.

The early-injection candidate's complete component timing, including input
staging and output copies, had these medians across available passing cases:

| Rows | Eager | Replay |
| --- | ---: | ---: |
| 131 | 0.805 ms | 0.575 ms |
| 134 | 0.819 ms | 0.581 ms |
| 259 | 1.111 ms | 1.099 ms |
| 262 | 1.135 ms | 1.121 ms |

These selected-component measurements are not end-to-end gains. At the proposed
256-token chunk sizes, the benefit is only about 1%. The separate `mix_timing`
field for early injection compares unequal work and must not be used as a
performance claim. Only `complete_timing` compares complete equivalent work.

Both sets of private graphs were retired explicitly after completed diagnostics.
Live graph owners, twelve entries per worker, remain intact. Private allocation
was approximately 80 MB/rank for mixing and 47 MB/rank for early injection.
Capture released unused allocator reservations; that reduction is not new
usable context capacity and did not clear prefix caches.

## Existing decode graphs and W4

A temporary target-only eager diagnostic preserves graph owners and restores
the exact runner attribute state. Six normal-stop requests in graph/eager/graph
order produced coherent responses. Graph request totals were 13.28–13.80 seconds;
eager totals were 48.01–48.97 seconds. These short diagnostic samples are not an
accuracy or broad throughput benchmark. The original graph path is restored;
drafter graphs were not disabled by this diagnostic.

W4 applies to TP6, but the local expert counts are 85/86 rather than TP4's 128.
Qualification must use actual loaded shard shapes. Earlier TP6 full-MoE fusion
trials passed local correctness checks but regressed complete calls by 2–5%;
see [the fusion report](qfn_tp6_offline_fusions.md). Those candidates remain off.

## Reproduction and ownership

`tools/qwen4exp/tp6_pacing_control.py:engine_source(8, 256)` generates a fixed
256-token candidate. `engine_source()` restores the exact original policy
object. Both require a paused, drained TP6 paced scheduler and check ownership
again before changing the policy. No scheduler methods or model weights change.
This is a resident experiment; cold-start defaults are unchanged.

`tp6_hc_graph_gate.py:engine_source(early_injection=True)` generates the private
second gate. Use `engine_source(retire_graphs=True)` to retire the latest
completed private capture. The gate cannot install a serving candidate. Failed
or uncertain owners stay retained for inspection. `tp6_eager_diagnostic.py`
generates temporary eager enable/disable controls; it is diagnostic only.

Evidence and the authenticated controller are under
`artifacts/qfn-tp6-graphs-pacing-20261016/`. Raw receipts and event streams remain
in the matching directory under `/srv/ai/storage/artifacts` on the host. The
controller reads the resident-v8 token locally and never exports it. Review
hard-coded deployment paths before reuse. Operations use
`pause?mode=wait&clear_cache=false`; no worker restart or prefix flush occurs.

Run `PYTHONPATH=. python artifacts/qfn-tp6-graphs-pacing-20261016/prepare_inputs.py`
from the checkout to generate code-review prompts and pacing controls, then copy them
with `normal_workload.py`, `realistic.py`, and `control.py` to the evidence
folder on the host. `python3 realistic.py` runs the normal-stop ABBA experiment
and restores the baseline afterward. It requires exclusive test use; output
receipt paths deliberately refuse overwrite. Use the recorded source revision
for exact prompt reproduction. Prompt text is generated from repository files,
not private Kilo conversations.

## Validation

60 focused host tests pass across scheduler policy, pacing ownership, graph
replay guards/lifetimes, private retirement, and exact eager override restoration.
Hardware diagnostics and API trials are separate evidence. Whole-repository
`bash format.sh ci` reports existing Ruff, spelling, Clang format, Markdown,
and forbidden-import failures outside this change; unrelated formatter edits
were reverted. No 256K accuracy, six-session stress, or cold-restart persistence
claim follows from these tests.

Final audit confirms HTTP health 200, resumed serving, all six original worker
PIDs, the exact original pacing policy object, shared PLE enabled without
failure, target eager override off, and both private graph owners retired on
every worker. NPU health is OK on all devices. The test server is available
again with its original decode graphs and scheduling policy.
