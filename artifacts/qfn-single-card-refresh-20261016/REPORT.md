# One-Duo Qwen3.8 refresh, October 16, 2026

The corrected FP16 baseline reproduces Josh's approximate 23–26 decode tok/s
band. Early draft submission, complete-prefix QSA selection and direct pinned
position copies gave small deltas that overlap baseline drift and changing MTP
acceptance. Dynamic W8 output heads did not provide a dependable improvement.
These are optional experiments, not mandatory startup additions. The shared
PLE RAM path is retained: it improved 512-token decode from 22.71 to 23.04
tok/s against its following baseline (~1.5%), with exact row checks passing
on both workers. It requires sufficient host RAM.

The important correctness change is the draft-KV addressing fix from
`4ee1d73d5`. The frozen launcher requests v2, but Qwen MTP selects the v1
runner; therefore the old single-card bundle also needs this fix. The new
`single_card_kv_fix` tool verifies all 5,274 original manifest entries, admits
exactly this proposer change and preserves the original manifest. The initial
corrected timing copy and canonical shipping patch have identical ASTs; their
comment-only difference and hashes are recorded in `patch-equivalence.json`.

## Matched configuration

One physical 300I Duo / two 310P3 devices, 0/1, on the qualification host
`192.168.53.187`. The other two cards were idle. Host RAM is 247 GiB; these are
not measurements on Josh's lower-memory host. Model weights are the original
W4 G128 300i checkpoint, with TP2/EP2, MTP2, 131,072 total tokens per request,
one sequence, 512-token prefill chunks, seven FP32 prefix-state slots,
`.965` memory utilization, `.80` KV fraction and decode graph `[3]`.
Automatic allocation profiling remains enabled. No firmware, driver, CANN,
transformers, operator package or expert-weight replacement was performed.

The baseline uses the frozen October 11 bundle with the mandatory KV fix.
Target/draft output heads and PLE projection remain FP16 unless explicitly
identified as the W8-head arm. `native.sh` was not installed. Large raw logs,
controller receipts and isolated bundles remain under
`/srv/ai/storage/artifacts/qfn-single-card-refresh-20261016/` on that host.

## Measurements

Each median covers three fixed prompts, temperature zero, MTP2 and forced
output lengths. The table excludes warmups, CPU profiling and exact shadow
checks; the evidence JSON also preserves those separately labeled runs.
`benchmark-summary.json` retains per-request rates, acceptance and text hashes.

| Configuration | 512-token decode | 2,048-token decode | 512-token decode after ~8K prefix |
| --- | ---: | ---: | ---: |
| FP16 baseline, first | 22.30 | 24.16 | 21.67 |
| FP16 baseline, return | 23.07 | 24.64 | Not repeated |
| Early draft | 23.67 | Not measured | Not measured |
| Complete-prefix QSA | 23.13 | Not measured | Not measured |
| Early draft + QSA | 23.13 | 25.28 | Not measured |
| Direct pinned positions | 23.03 | Not measured | Not measured |
| All three | 23.38 | 25.73 | 22.44 |
| W8 output heads only | 22.14 | 24.79 | 22.66 |
| Shared PLE RAM only | 23.04 | 24.94 | 21.94 |
| Return after shared PLE RAM | 22.71 | Not measured | Not measured |
| Custom collective only | 21.88 | 23.60 | 22.21 |
| Custom collective + shared RAM | 22.22 | 24.60 | 22.09 |
| Custom collective return, RAM removed | 22.43 | Not measured | Not measured |

Rates are client streaming decode tokens/s. The ~8K timed prompts actually have
8,228/8,239/8,245 tokens and reuse 8,192 cached tokens. They are different from
the exact 8,192-token cold/warm recall probe. That probe measured 275 prompt
tok/s and 0.815-second replay TTFT on baseline, 277 and 0.861 seconds for the
early/QSA pair, and 275 and 0.801 seconds for W8. Each replay reused 8,064 tokens;
all recalled the passcode. See `prefill-summary.json`.

The baseline returned faster even without candidates. Three sequential prompts
per arm do not establish statistical significance or portable gains. The final
FP16 restart is another return check, not an independent optimization.

## Correctness and profiling

Every completed smoke arm retained 6/7, with `ASCEND` reversed as `DNESCA`
instead of `DNECSA`. Failures are preserved in `smoke-summary.json`. QSA shadows
compared all four selection tensors exactly on both workers, including the
fallback beyond complete-prefix coverage. Source guards reject incompatible
frozen model methods rather than transplanting a new model implementation.

The W8 trial changed only `lm_head_execution` in the model command. Runtime
audits observed INT8 target head weights and FP16 PLE projections on both
workers; startup verified identical target/draft head weights and scales before
sharing them. Audit wrappers were removed before timing. The fixed held-out
228-question MMLU sample scored 204/228 for FP16 versus 207/228 for W8, with
zero invalid answers and no lost correct answers. Dataset identity and all
paired prompt hashes/predictions are recorded in `quality-comparison.json`.
This small fixed sample is not an official full MMLU score or broad accuracy
certification. W8 full-window accuracy was not tested.

The separate CPU sample pointed to output bookkeeping and QSA selection.
Sampling at 100 Hz across processes fell 2.83 seconds behind; percentages in
`cpu-profile-summary.json` describe sampled stacks, not precise wall-time shares.
Profiled throughput is not used in the comparisons above.

## Retained shared PLE RAM

The frozen baseline reads PLE rows from a file mapping backed by the OS page
cache. The candidate gathers the same FP16 rows from a sealed shared memfd,
then preserves the original hash, compact-table staging, transfer and FP16
projection. This is a host RAM optimization; it does not enable direct
NPU-initiated access or native PLE DMA. HCCL remained enabled in this arm.

Each worker passed eight exact comparisons against its original row gather.
The 95.37 GiB table was reused from the existing owner, not allocated twice.
The measured host has 247 GiB RAM; this option is not qualified on Josh's
128 GB host. The preparation tool requires table size plus 48 GiB currently
available RAM. Avoid pushing this table into swap.

The retained result is a modest measured improvement: 23.04 tok/s for short
512-token output, 24.94 for 2,048 output tokens, and 21.94 after an ~8K prefix.
The cold 8K recall probe measured 281.12 prompt tok/s; its first replay reused
8,064 tokens with 0.802-second TTFT. Both recalled the passcode. These are
three-prompt medians on this host, not a promise of the same gain elsewhere.

## Custom collective integration and startup diagnostics

The isolated collective bundle backports four focused plugin changes: the
communicator, two graph lifecycle files, and admission of the communicator's
configuration. Model, cache, drafter, operators and weights stay frozen except
for the mandatory draft-KV fix. Tooling is copied separately and its package
resolution is pinned ahead of inherited SDK/controller paths. The original
corrected bundle is unchanged.

The TP2 binomial private graph gate passed all 16 FP16/FP32 geometries with
changed inputs, different capture counts, eager interleaving, reverse replay
orders, graph retirement and explicit ownership release on both ranks. The
compact evidence is in `collective-private-gate-summary.json`. Its small
synthetic collective timings are not whole-model throughput measurements.
The first gate attempt lacked the SDK compiler's Python import path and failed after
its timed cases, before final checks. Both workers were inspected at the host
error, stopped, and confirmed gone. That attempt is not counted as a passed
or protocol-clean gate; the fresh complete gate passed and closed correctly.

API startup subsequently exposed an independently reproducible pandas import
failure. Its installed `indexers` binary did not match its wheel RECORD; all
other recorded pandas files matched. The damaged file was preserved and only
that binary was restored from the cached **same-version 3.0.6** wheel after
checking its bytes against RECORD. Standalone pandas import then passed.
No dependency version changed. `pandas-integrity-before.json` and
`pandas-repair.json` record this repair; the cause of the corruption is unknown.
The loader error occurred before model workers were created.

The collective-only serving arm retained 6/7 smoke and correct 8K cold/replay
recall, but did not improve the short-output or 2,048-token medians over the
nearby HCCL baseline. Its cold probe measured 273.22 prompt tok/s and the first
replay TTFT was 0.832 seconds with 8,064 cached tokens. The private gate's
microbenchmark uses an out-of-place HCCL reference with a baseline clone;
serving's original HCCL reduction is in place. The custom operation returns a
new tensor. That difference limits extrapolation from synthetic timings; this
experiment does not isolate the cause of the whole-model slowdown.

The combined collective-plus-RAM arm also retained 6/7 smoke, passed eight
exact row comparisons per worker, and recalled both cold and repeated 8K
prompts. Cold prefill measured 279.05 prompt tok/s and replay TTFT was
0.843 seconds. It reached 22.22/24.60 tok/s for short/long output. Returning to
the collective-only reader gave 22.43 tok/s for short output, so the combined
arm's small gain over the first collective baseline did not survive that return
comparison. HCCL plus RAM remains the retained option for this one-card host;
the custom collective stays optional and is not a dependable one-card speedup.

Both collective workers completed explicit graph retirement and resource
release before stopping. The TP6 service restoration is recorded separately
in `restoration-summary.json`; no one-card experimental bundle is installed
into that service. `collective-serving-summary.json` preserves compact worker
checks and shutdown evidence, with hashes identifying the full host receipts.

## Scope of context testing

The final FP16 restart reproduced short decode. A new 130,560-token capacity
probe was cancelled at the user's request before completion; it is excluded
from performance and correctness claims. Active KV usage returned to zero.
The October 11 full-window qualification remains historical evidence, not a
new validation of these candidates. Subsequent comparisons use short prompts
and 8K cold/replay pairs only.

## Restored six-device service

The original TP6 configuration is serving again at
`http://192.168.53.187:8001/v1`, model `qwen38-flash-next`, API PID 1751329,
under `resident-singlecardrestore`. All six ranks report active custom SUM,
clean graph state, native PLE enabled with verification complete and no failure,
and shared-prefill PLE installed with four exact checks complete and none
remaining. The table owner survived the full experiment.

The retained scheduler, early draft and complete-prefix selection were restored.
Each rank passed 862 exact selection comparisons before fast mode resumed.
The 5,704-token prefix probe passed, and both final public arithmetic requests
returned `391`. The first 2,426-token request exercised only three of the four
shared-prefill checks; a second small independent prompt completed the fourth.
This was an incomplete verification count, not a row mismatch. No full-window
request was needed for restoration. The service retains its 262,144-token
per-request limit and six-sequence TP6 profile; this is not new full-window
accuracy qualification.

## Repository validation

The original refresh passed 152 focused tests and 64 draft-slot regression
cases. The PLE/collective follow-up added 20 passing adapter/shared-gather tests
and 141 passing collective graph/lifecycle tests. The authenticated benchmark
metrics regression test also passed. Hardware evidence is separate
from these CPU tests.

The required `bash format.sh ci` was run in an isolated checkout. It failed on
existing repository-wide Ruff, formatting, spelling, clang-format, Markdown and
forbidden-import findings. Its unrelated automatic changes were reverted.
Full pytest collection hit 13 existing dependency/import errors, including
`modelscope`, `vllm.third_party.flash_linear_attention` and NPU/native imports.
Scoped checks cover the delivered files; this is not a full-suite pass.
