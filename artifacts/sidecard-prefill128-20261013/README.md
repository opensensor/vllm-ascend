# GLM cold prefill: 128-row Ascend batches

The unchanged **23,820-token full Code prompt** now prefills in **190.486
seconds**, versus a fresh 64-row control at **246.264 seconds**. That is another
**22.6% reduction** in compute time and **125.05 prompt tokens/s** instead of
**96.73**. Both paths reset all model state, use zero cached prompt tokens and
match the full vocabulary bit for bit over the final prefill and 64 decode
steps. The [raw matched measurements](probe/cold-benchmark.json),
[summary](summary.json) and [design PNG](design.png) preserve the comparison.

The historical unbatched run of this same prompt took 327.696 seconds. The
new result is 41.9% below that earlier measurement; it is not a fresh three-way
control in this turn. All 64-row vocabulary hashes for 2K, 8K and full Code
also match the previous qualified measurements exactly.

## Implementation

Four original 32-token CUDA chunks now share one 128-row cold expert call
per active Ascend chip. CUDA projection, attention, routing, hot/shared MLP,
cache updates and ordered owner reduction retain the qualified arithmetic.
Weights remain resident on their owners. Decode math remains unchanged.
One RTX PRO 6000 and physical Ascend chips 2 and 3 participate; the other
Ascend card is outside this runtime.

The result occupies the existing 2 MiB slot exactly. The separate control
allocation carries the invocation header and completion sequence. The packed
header/result path retains its 64-row native limit. The consumer library now
advertises its row capacity, and startup rejects insufficient older libraries
before allocating exports or submitting future-value CUDA waits. Missing
capacity symbols retain the older 64-row admission. No environment variable
or driver/kernel-module change was introduced.

Both payload directions remain Ascend-initiated direct PCIe DMA. Host commands
admit work and small routing metadata reaches the CPU for validation. GPU hot
and shared work overlap the cold calls; the implementation retains the fixed
hot/rank-0/rank-1 merge order and its device completion contract.

## Matched compute measurements

| Prompt | 64 cold rows | 128 cold rows | Before tokens/s | After tokens/s | Less prefill time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 tokens | 15.928 s | 12.969 s | 128.58 | 157.92 | 18.6% |
| 8,192 tokens | 58.389 s | 50.721 s | 140.30 | 161.51 | 13.1% |
| Full Code, 23,820 tokens | 246.264 s | 190.486 s | 96.73 | 125.05 | 22.6% |

These are one matched pair per fixture on the same loaded model, with identical
prompt IDs and admission. They measure isolated prefill compute and exclude
API queueing, HTTP delivery and prefix snapshots. Teacher-forced decode timing
includes hash/parity observation, so those rates are diagnostic. This establishes
no decode acceleration or general long-context quality improvement. Context
admission remains 65,536 total tokens. The private full Code request and model
weights are excluded; prompt ID hashes are retained.

## Remaining cold compute cost

The [worker span summary](worker-span-summary.json) derives timings from the
existing bounded worker traces, without adding device events or barriers.
Snapshots after the completed full Code cases retain these cold calls:

| Cold rows | Rank | Retained calls | Mean native span |
| --- | ---: | ---: | ---: |
| 64 | 0 | 3,756 | 27.029 ms |
| 64 | 1 | 3,636 | 26.458 ms |
| 128 | 0 | 2,790 | 37.602 ms |
| 128 | 1 | 2,790 | 36.479 ms |

The native spans include host kernel submission and its completion wait. The
64-row snapshots are retained tail samples; the 128-row snapshots cover the
complete full-window cold calls. Different token coverage limits direct kernel
comparisons. Both ranks execute concurrently and CUDA overlaps their work;
these spans are not additive model wall-time partitions. The raw snapshots
also retain upload, metadata validation and output transfer spans.

Cold expert work remains substantial. Increasing the batch further may improve
weight reuse and kernel utilization, but 128 FP32 rows already fill a 2 MiB
result slot. A larger candidate needs larger dedicated slots and fresh transport
and model qualification. That candidate is not implemented in this delivery.

## Qualification and serving

The [raw transport probe](raw/transport-probe.json) completes **21 cases**:
three repetitions of 1, 4, 17, 32, 64, 96 and 128 rows. It checks input records,
sequence subranges, prequeued device waits, exact ordered FP32 addition and
clean mapping closure with intact guards. The library was compiled and pinned
in [build.json](raw/build.json). The previous 64-row service remained resident
during this small transport probe; the probe released its temporary owners.

The [2K gate](probe/prefill-parity.json) matches every original 32-token boundary
and 64 decode steps against the pinned external reference, with maximum absolute
logit error zero. The larger matched cases independently match the final prefill
and every decode output. Both model probe workers close with exact poison guards
before the new serving process loads. Compressed worker receipts preserve this.

The live runtime is
`/srv/ai/storage/artifacts/sidecard-prefill128-20261013/serve-v1`.
It enables `prefill_window_size: 128`, retains `prefill_chunk_size: 32` and
uses the authenticated separate-control completion library. Native repeatability,
transport parity, DSA parity, prefill parity, prefix-cache parity and real-weight
startup inference all pass before readiness. A fresh cold 2K API request reaches
first text in **12.801 seconds** and decodes 64 output tokens at **12.010
tokens/s**, with zero cached tokens. See `server/` and `api-cold-2048.json`.
The endpoint remains `http://192.168.53.187:8000/v1`, model `glm-sidecard`.

All 111 serving Python files match the delivery source byte for byte. Native
source and compiled library hashes are recorded separately. The exact executed
[source archive](executed-source.tar.gz), raw gates, model gates and compressed
phase receipts are included. Private environment file contents are excluded.

## Checks and reproduction

**1,121 standalone sidecard tests pass.** New tests cover native capacity
preflight before allocation, exact result capacity with separate controls,
older-binary rejection, larger transport cases, four CUDA chunks including
tails, causal state/order, configurable benchmark windows and rejection paths.
All changed-file hooks pass. Whole-repository `bash format.sh ci` retains the
seven existing failures: ruff check/format, codespell, typos, clang-format,
markdownlint and forbidden imports. It runs in an isolated worktree; unrelated
automatic edits are discarded. Check logs and the CI summary accompany this report.

```bash
python -m pytest -q --confcutdir=tests/ut/sidecard tests/ut/sidecard
nvcc -shared -Xcompiler -fPIC --cudart shared --fmad=false -arch=sm_120 \
  -o completion_merge.so tools/sidecard/completion_merge.cu
python -m tools.sidecard.cold_prefill_benchmark \
  --config /absolute/path/to/128-row/runtime.json \
  --chat-request /private/path/to/full-code-request.json \
  --windows 64 128
```

Close the serving model through its normal lifecycle before loading a second
complete model. Preserve checkpoint, native activation override, placement,
CUDA expert library, assigned hardware and library hashes from the qualified
runtime. The runtime JSON also needs the compiled completion library SHA256
and the pinned external prefill reference, as described in the
[completion runbook](../../tools/sidecard/GLM_COMPLETION.md).
