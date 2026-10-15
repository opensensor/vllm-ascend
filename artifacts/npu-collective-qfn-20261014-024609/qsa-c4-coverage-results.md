# C4 graph coverage and short-context QSA qualification

The retained payload is `qsa_combined_coverage_extended_prefix`. Model
`qwen38-flash-next` remains on `http://192.168.53.187:8001/v1` with the same
four workers (579046–579049) and tracked weights. This is the two-card Ascend
TP4/EP4 QFN service, not the earlier CUDA/Ascend GLM experiment.

## C4 measurements

Four concurrent streaming requests use distinct short prompts (24–26 tokens),
seed 42, temperature zero and exactly 256 output tokens each. A host barrier
releases requests within one millisecond. Aggregate HTTP throughput is all
1024 output tokens divided by first-start to last-finish time, including
prefill and batch tails. Per-request decode uses the streaming N−1 estimate;
MTP can emit multiple tokens in one chunk. It is not aggregate throughput or
an isolated kernel measurement.

| Coverage | Batch time | Aggregate output tok/s | Median per-request decode tok/s |
| --- | ---: | ---: | ---: |
| Original `[3, 6]`, C4 eager fallback | 25.903 s | 39.53 | 10.33 |
| General graphs `[3, 6, 9, 12]` | 16.660 s | 61.47 | 16.34 |
| General and prefix graphs, first trial | 16.245 s | 63.04 | 16.82 |
| General and prefix graphs, return trial | 15.646 s | 65.45 | 17.42 |

The two prefix trials bracket the old-coverage control. They improve aggregate
throughput by 59.5–65.6% over that control. This is a small sample, not a
confidence interval. Distinct output text and speculative acceptance can vary;
these are identical prompt/budget workloads, not identical token replay.
The initial 64-token C4 smoke similarly improved from 34.63 to 53.16 aggregate
tok/s after general graph coverage was enabled. Do not mix those short-request
rates with the 256-token comparison.

The final C1 check measured median 31.31 output tok/s for the complete HTTP
request. Server request-log decode was 32.6 tok/s median, with runs of
32.6/34.1/32.2. This preserves the earlier C1 performance range; it does not
establish a new C1 speedup or meet the 40 tok/s objective. Evidence:
`decode-benchmark-c4-prefix-c1-final-v3.json` and
`c4-prefix-c1-request-timing-v3.json` in `qsa-resident-v1/`.

## Root causes and implementation

MTP2 needs three target tokens per decoding request. Four requests need a
12-token graph, and a three-request tail needs nine. The original largest
capture was six. The logged warnings at 14:13:14 and in the matched control at
14:28:40 explicitly show fallback for those missing sizes. The control was
temporary; it is not the retained configuration.

Changing only the dispatcher was insufficient. The first extended warmup was
rejected because GDN's metadata builder cached capacity six at construction.
All ranks acknowledged that Python admission failure. The original coverage
was restored and resumed without restarting workers before revising the fix.

The corrected cold transaction:

- Uses the existing coordinated pause, graph retirement, native lease release
  and capture path. It changes no weights, collective transport or replay observer.
- Updates configured capture sizes, maximum, runner sizes and dispatcher keys
  together. Empty target/draft attention parameter dictionaries retain their
  owners and receive the same keys.
- Resizes 13 owned GDN metadata buffers for each of three builders per rank,
  preserving dtype/device and CPU pinning. Original and partially allocated
  buffers remain owned. Buffer pointers change only after graph retirement;
  no recurrent-state or KV-cache tensor is resized.
- Extends QSA admission to the bounded `[3, 6, 9, 12]` profile. Explicit prefix
  coverage can be C1-only or all admitted sizes. Any request beyond the prefix
  limit selects its general graph. Derived GDN metadata stays independent
  between captures, and prefix graphs retain their separate allocation pool.
- Supports explicit restoration of sizes `[3, 6]`, including metadata capacity,
  rather than merely changing the command-line label.

All ranks' final inventory has four general and four prefix target variants.
Each general variant has 14 native graph segments and 13 eager selection breaks;
each prefix variant has two native segments and one PLE break. The native
INT64 geometry kernel still admits only shapes three and six; larger shapes
retain the original arithmetic. No runtime tracking counter was added.

## Qualification and limits

The final host suite passed 138 tests covering coverage changes, live-resource
rejection, dispatcher restoration, GDN buffer allocation failure and retention,
QSA selection, and prefix/general transitions for packed request counts 1–4.
Scoped repository hooks passed. This is not a full-repository test run.

The general C4 build passed the existing eight model smoke checks. The prefix
C4 build passed ten: short input, crossing 2048 tokens, above-boundary input,
8K input, return to short, four simultaneous contexts of 512/2030/2100/4096,
and return to short again. Each response met its exact token budget and the
fixed hash-table semantic smoke check. Separate four-topic C4 requests produced
coherent responses. These gates do not establish full-context model quality.

The configured context remains 262144. Model name, API port, all worker PIDs
and tracked weight storages remain unchanged. The 40 tok/s single-request
objective remains separate from the improved C4 aggregate result.

## Reproduction and recovery

The controller is `qfn-capture-coverage-switch.py`. Use the qualified
`qsa-resident-v1/restore-combined-after-matched-control-v1.json` base receipt,
this run's private control token and a new output receipt. `extended_prefix`
selects the winner; `extended` selects general C1–C4 with C1-only prefix;
`baseline` restores both dispatch and GDN buffers to the original profile.
A successful transaction preserves an already paused server; resume explicitly
only after checking all-rank acknowledgments. Failures retain the pause.

Do not apply these receipts to fresh unloaded workers. Native owners and the
prior combined qualification are required; this addition is not baked into
the launcher. Source guards pin the dispatcher, runner and GDN builder files.

Evidence is in `qsa-resident-v1/`:

- `install-c4-coverage-v2.json`, `install-c4-prefix-v3.json`
- `restore-c4-prefix-final-v3.json`
- `validation-c4-coverage-v2.json`, `validation-c4-prefix-v3.json`
- `concurrency-baseline-c4-256-v3.json`
- `concurrency-extended-c4-256-v2.json`
- `concurrency-prefix-c4-256-v3.json`, `concurrency-prefix-c4-256-final-v3.json`

Log remains
`/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v3/server.log`.

## Subsequent recovery and fresh profile

This run was superseded by `resident-live-v4` after one user-authorized recovery
restart. The same combined candidate and C1–C4 coverage were requalified; the
model and port are unchanged. Current log is
`/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v4/server.log`.
See [fresh measurements and graph findings](graph-refresh-results.md).
