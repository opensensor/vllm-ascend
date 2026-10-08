# Gate A baseline: first request blocked by KDA metadata

## Current profiling evidence, 6 October 2026

The startup failures below are historical. Fresh **c1 and c4 captures on all
four ranks** are now recorded in
[the resident decode study](../decode-trace-20261006/README.md).
They used direct CANN profiling, disabled launch blocking, retained the
loaded weights, and recorded device iteration boundaries alongside scheduler
token counts. Collection was finalized before restoring serving graphs.

The study includes task exports, numerical summaries, and PNG charts. Packed
expert projections remain the largest identified task family; AI CPU casts
are also visible. Replay communication labels and the complete dependency
graph remain incomplete, so this is not a claim that full critical-path
attribution has closed. The validated column-cache prompt improvement is in
[the prompt study](../prompt-profile-20261005/README.md).

## Historical startup record

The user released the four Ascend 310P devices on 2026-10-01. The baseline
source is an isolated archive of committed HEAD
`5fdfb93908e796856e5040251c955ab2eb1a0cd9`; unrelated working-tree
edits are excluded. The exact launch, checkpoint metadata, runtime versions,
and loaded custom OPP files are recorded in `serve-baseline.sh`,
`identity.json`, and `opp-files.sha256`.

The first server attempt loaded 33 shards in 315.24 seconds but failed on its
first dummy forward because a git archive does not include the GLM Torch
extension. The `mhc_sinkhorn_310` Torch binding was therefore unregistered.
The previously built GLM extension was linked into the isolated archive, its
SHA-256 was recorded in `identity.json`, and a CPU import confirmed registration.
The second startup read all 33 shards in 269.79 seconds and reported 274.11–274.31
seconds total `load_model` across ranks. Its dummy forward then selected the
generic attention backend for GLM kpool sparse MLA and failed because that
backend has a different `forward` signature. The failure is in the 310P
compatibility selector: `(use_mla=True, use_sparse=True)` was absent from its
map, while GLM's indexer sets `use_sparse=True`. The logs are
`server-attempt1-missing-binding.log` and
`server-attempt2-sparse-mla-routing.log`.

The third startup included the targeted kpool routing fix. It loaded weights
in 286.70 seconds (290.91–291.56 seconds total `load_model`) and opened the
API. The first 32-token request failed on all ranks in the W2 KDA path:
`metadata.non_spec_decode_metadata` was `None` when `kda_310.py` read its
`causal_conv1d` field. The log is `server-routingfix-fault.log`, and the
request record is `fault-routingfix.jsonl`. No parser-aware quality or
throughput result has been recorded.

The fourth startup included the W2 KDA metadata fix. It loaded weights in
299.33 seconds (303.28–303.58 seconds total `load_model`) and opened the API.
The 32-token fault request completed its 36-token prefill, then failed on the
first decode token: the shared MLA `forward()` called the 310P override with
three arguments, but that override still expected six. The evidence is
`server-kda-fixed-mla-fault.log`, `fault-kda-fixed.jsonl`, and the per-worker
`io-kda-fixed.csv`. The four workers recorded 39.06–43.89 billion physical
read bytes and 21.23–21.46 GiB peak sampled RSS; these are startup run
measurements, not a filtered-loader comparison.

A GLM kpool-only selector fix is staged in `vllm_ascend/platform.py` with four
passing CPU regression cases. A W2-specific 310P GDN builder is also staged:
the shared 310P builder omits nested prefill and decode metadata, while W2 KDA
consumes it. The W2 builder supplies those fields without changing the shared
310P path. Four builder and five KDA CPU tests pass on the target host. The
KDA metadata was exercised by the fourth startup and passed its prefill step.
The 310P MLA decode override now accepts the shared MLA result object and
passes its query and cache tensors to the fused 310P path. The focused CPU
suite has 24 passing tests, including two decode-interface regressions. The
user deferred further NPU use before this MLA fix could be validated in
serving.

The saved 30 September traces at
`/srv/ai/src/glm-round-20260930/glm-profile-codex-20260930/npu` on Threadripper
have eight raw rank directories across one- and four-stream captures. Their
parsed `kernel_details.csv` exports exist only for ranks 1 and 3 of the first
capture, and ranks 2 and 3 of the second. Rank 0 has no parsed kernel CSV in
either capture. These files establish the CSV/communication JSON schema and
historical bottlenecks; they cannot satisfy all-rank gate-A attribution. The
historical short runs also generated 32 tokens without the parser-enabled
complete-answer requirement.

`python3 -m tools.glm_perf.analyze_trace` now reads four rank exports, accepts
explicit phase or step windows, separates overlapping task sums from elapsed
task envelope, and reports matched collective arrival spread and wait/transit
attribution. It deliberately rejects incomplete captures. Its envelope is
only an elapsed bound: the exports do not contain a complete cross-stream
dependency chain.

The launcher preserves the historical parser, tool parser, eager mode, TP4,
checkpoint, and OPP configuration with an isolated source path. The remaining
gate-A sequence is the 32-token fault smoke, complete-answer suite, one/four
stream 256-token workloads, 8K and 16K resident checks, all-rank profiling,
per-rank HBM ledger, host RSS, and fault-log review. If startup or the first
generation faults, diagnose that transition before timing candidates.
