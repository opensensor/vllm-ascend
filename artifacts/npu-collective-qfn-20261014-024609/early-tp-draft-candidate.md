# Early TP draft candidate

Status: resident trial completed, then removed from the serving payload.
The narrow CPU ordering tests pass; the model trial did not establish a speed
improvement. This is an experimental candidate, not a promoted optimization.

`tools/qwen4exp/resident_candidates/qfn_early_tp_draft.py` changes one condition
in the qualified `NPUModelRunner.sample_tokens`: the existing early pipeline
parallel draft path also admits a narrow tensor parallel QFN decode case.
The complete qualified function must match SHA256
`25dc0ac0921d87f03c8bb1974972b73b91c5acf868b1b014185410b6eb837dbc`.
The candidate compiles the runtime's own function, closes over its admission
helper, and retains the original module globals and inference decorator.
It does not copy the newer local model runner into the resident process.

Admission requires QFN's exact target class and stable MTP hidden buffer,
MTP with two scheduled draft tokens, TP4/PP1/DP1, synchronous scheduling,
padded drafting, uniform decode with speculative metadata, no discarded
requests, no pending prompt logprobs, no DCP, and greedy draft sampling.
All other TP requests keep bookkeeping before drafting. The existing PP
condition and behavior remain unchanged.

## Dependency audit

The qualified target PLE callbacks already execute during target forward.
QFN MTP explicitly rejects PLE layers in its constructor, so draft execution
does not need the sampled IDs appended to host PLE history by bookkeeping.
Its padded input preparation consumes sampled IDs and accepted/rejected
counts from device tensors. The backup host token is selected only when a
row has no valid sampled token; discarded rows are excluded from this first
candidate. Rows with no valid tokens are not extended by bookkeeping.

MTP reads the target's model-init hidden buffer and copies into its own stable
input buffer. Draft attention metadata and slot mappings derive from target
step metadata, independent of newly appended host output history. Greedy draft
sampling is argmax and does not consume per-request generators or output
history. Bookkeeping still extends history exactly once, and the final
accepted-token/Mamba state update remains in its original position.

Prompt logprobs are an explicit exclusion: bookkeeping can consume target
hidden states that are weak graph outputs. Draft replay through a shared graph
pool could overwrite those states before prompt logprobs are calculated.
The candidate must not be broadened without addressing that lifetime.

The optimization only moves submission ahead of a synchronous host read.
Draft expert dispatch still synchronizes routed IDs, so CPU time currently
attributed to rejection parsing can move elsewhere rather than disappear.
Only unprofiled end-to-end measurements can establish a speed improvement.

## Validation and hardware gate

CPU regressions execute the exact pinned function fixture, verifying one
proposal before bookkeeping in the admitted case, original ordering in
excluded cases, unchanged PP behavior, preservation of live globals, and
failure on source drift. A prompt-logprob test models draft overwrite of
target hidden state and verifies that the guarded path avoids that hazard.
The fixture is the unmodified qualified function, not a runnable model.

After resident installation, require matching deterministic baseline tokens
and completion lengths for C1 and C2 decode; include rejected speculative
tokens, short/long requests, request turnover, and prompt-logprob fallback.
Check unchanged worker/weight identities and native collective ownership.
Then compare multiple unprofiled C1 trials and a bounded CPU trace. Keep
cold-prefill and unsupported modes on their original order.

## Resident trial

The final wrapper-aware candidate hot-loaded with unchanged worker PIDs and
weight storage. Eight short, boundary, 8192-token, concurrent and subsequent
short requests completed with coherent responses and healthy ownership. Some
token sequences differed; repeated baseline runs already show variation, so
this is a semantic regression gate, not exact model-wide parity.

Three unprofiled 512-output C1 trials measured 30.24, 32.96 and 31.40 tok/s,
median **31.40**, versus earlier QSA-only median **31.66**. The ranges overlap;
there is no demonstrated gain. The serving payload returned to QSA-only
ordering before the compact-top-k trial. Admission was guarded at runtime;
no hot-path admission counters were installed, and the baseline native trace
does not measure this candidate's realized overlap.

Receipts: [validation](qsa-resident-v1/validation-early-tp.json),
[benchmark](qsa-resident-v1/benchmark-early-tp.json).
