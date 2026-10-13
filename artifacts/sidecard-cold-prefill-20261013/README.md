# Faster GLM cold prefill with the full Code prompt

The unchanged **23,820-token default Code request** takes **247.01 seconds**
to prefill, down from **327.70 seconds**: **24.6% less compute time** and
**96.4 prompt tokens/s** instead of **72.7**. Both paths use zero cached prompt
tokens, the same weights and token IDs, and the same sparse-attention admission.
The model's full vocabulary matches bit for bit across the final prefill output
and 64 teacher-forced decode steps. This measures faster model computation;
there is no prompt reduction or compact agent in this comparison.

See the [resulting design PNG](design.png) and
[raw matched measurements](probe/cold-benchmark.json).

## What changed

Prefill advances a bounded 64-token window by layer. CUDA still computes
attention, routing, hot experts and shared MLP in the original 32-token chunks,
in causal token order. Only stateless cold expert rows are concatenated into
one 64-row Ascend call per active chip. CUDA combines hot/rank-0/rank-1 partials
in the qualified order, splits them back into the original chunks and proceeds
to the next layer. The checkpoint and decode graph implementation are unchanged.

This gives each cold invocation more work and reduces repeated kernel/control
submission. It retains all resident weights, all 45 CUDA backbone layers, and
CUDA ownership of KV, recurrent and mHC state. One RTX PRO 6000 and both chips
of the assigned Ascend 310P Duo participate; physical Ascend devices 0 and 1
are outside this runtime. The DMA payload slots remain 2 MiB. Metadata buffers
and CUDA completion scratch are admitted for 64 cold rows independently of
32-row CUDA expert scratch. No environment variable was added.

Activations and FP32 result payloads cross PCIe through the existing direct
DMA transport. Both directions are Ascend initiated. Host commands still admit
NPU work, and small route metadata is validated on the CPU; this change does
not introduce an autonomous device dispatcher. GPU hot/shared work overlaps
cold work, and completion sequences release the ordered CUDA consumers.

Earlier experiments that increased the complete CUDA chunk to 64/128 rows
failed numerical parity. This implementation keeps that CUDA geometry at 32.
The earlier mismatch's precise operation remains unidentified; the new cold
batching design passes the qualified arithmetic reference.

## Matched real-weight measurements

| Prompt | Original cold prefill | Batched cold prefill | Original tokens/s | Batched tokens/s | Less prefill time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 tokens | 20.006 s | 15.545 s | 102.37 | 131.75 | 22.3% |
| 8,192 tokens | 69.546 s | 58.402 s | 117.79 | 140.27 | 16.0% |
| Full Code, 23,820 tokens | 327.696 s | 247.012 s | 72.69 | 96.43 | 24.6% |

These are isolated prefill compute durations, measured on the same loaded
runtime with fresh model state before each path. They exclude API queueing,
HTTP delivery and prefix snapshots. There is one run per path and fixture,
so these are measured pairs rather than medians or a variance study. The full
Code request contains 23 tool definitions and about 100 KB of serialized
instructions/context. The private request body is excluded from this evidence;
its SHA256 and the exact prompt token SHA256 are in the measurement receipt.
Generated tools are never executed by the benchmark.

All three fixtures match the full vocabulary over 64 teacher-forced output
steps, with no cached prompt tokens. Decode measurements include CPU hash/parity
observation and remain about 12 tokens/s; this is a prefill optimization and
establishes no decode speedup. The service still admits 65,536 total tokens.
A full Code arithmetic comparison does not establish general long-context
model quality at that limit.

## Qualification and deployment

The [2K prefill gate](probe/prefill-parity.json) matches every original 32-token
boundary plus 64 decode steps against both the unbatched path and the pinned
external reference, with maximum absolute logit error zero. The larger 8K and
full Code comparisons independently check the final prefill and every decode
logit. The probe closes both workers with exact poison guards and releases the
completion owner before the serving process loads.

The faster serving snapshot is
`/srv/ai/storage/artifacts/sidecard-glm-completion-20261013/cold-serve-v1`.
Its runtime enables `prefill_window_size: 64`, keeps `prefill_chunk_size: 32`,
and runs native repeatability, transport parity, DSA parity, prefill parity,
prefix-cache parity and real-weight startup inference before API readiness.
The endpoint is `http://192.168.53.187:8000/v1`, model `glm-sidecard`.
Serving qualification and a fresh cold API completion are recorded separately
under `server/` and `api-cold-2048.json`.
The cold 2K API completion reaches first text in **15.596 seconds** and decodes
64 output tokens at **11.927 tokens/s**, with zero cached prompt tokens.

The [executed probe source](probe/executed-source.tar.gz),
[driver](probe/driver.py.gz), compressed worker receipts and source hashes preserve the
experiment. The final serving source contains the same coordinator/dispatch
math and a named window constant plus the reusable benchmark CLI; it is
qualified again at startup, and its source hashes are recorded under `server/`.
Weights, private environment file contents and the captured request are excluded.
The [serving source comparison](serving-source-parity.json) matches all 110
runtime Python files. The separate, unused benchmark CLI differs only in its
prompt-digest field label. That label also changes in the public probe JSON to
avoid a credential-scanner false positive; the exact original JSON is preserved
in `probe/cold-benchmark.json.gz`, with the mapping and hashes recorded in
`probe/receipt-normalization.json`. No measured value changes.

## Checks and reproduction

The standalone sidecard suite passes **1,090 tests**. New tests cover causal
layer order, partial windows, deferred state, both ranks' admission, original
CUDA call sizes, cancellation, poisoned failures, configuration bounds, gate
restoration and matched-benchmark numerical rejection. All changed-file hooks
pass. Whole-repository `bash format.sh ci` retains the same seven baseline
failures: ruff check/format, codespell, typos, clang-format, markdownlint and
forbidden imports. It runs in an isolated worktree; its unrelated automatic
edits are discarded. See `unit-tests.log.gz`, `scoped-ci.log.gz` and
`full-ci-summary.json` for the checks and their limits.

```bash
python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard
python -m tools.sidecard.cold_prefill_benchmark \
  --config /absolute/path/to/64-row-cold-window/runtime.json \
  --chat-request /private/path/to/full-code-request.json
```

Close the serving coordinator through its normal lifecycle before loading the
benchmark's second complete model. The CLI bypasses prefix caching, resets all
model state per path, checks the pinned arithmetic reference and rejects any
full-vocabulary difference. Keep the authenticated checkpoint, kernel overrides,
placement, DMA completion library and assigned hardware from the qualified
runtime specification.
