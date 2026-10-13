# GLM sidecard larger cold batches

This hardware study keeps CUDA projection, attention, routing, hot experts, and
shared MLP arithmetic at the qualified 32-row geometry. It combines eight or
sixteen of those chunks into a 256- or 512-row cold expert call on each active
Ascend chip. The result slots grow to 4 MiB or 8 MiB, with completion control
in its separate 64 KiB allocation. Decode retains its existing graphs.

## Hardware qualification

The existing UAPI 5 PCIe bridge admits transfers through 8 MiB. No kernel module,
driver, clock, ECC setting, or checkpoint weight changed. Physical Ascend chips
2 and 3 on one Atlas 300I Duo execute cold experts; the RTX PRO 6000 executes
the backbone and hot/shared experts. Physical 0 and 1 remain reserved.

- [4 MiB transport probe](raw256/transport-probe.json): 27 repeated cases.
- [8 MiB transport probe](raw512/transport-probe.json): 33 repeated cases.
- The 4 MiB probe includes 255/256-row boundaries; the 8 MiB probe also covers
  511/512 rows. Both preserve exact ordered FP32 rank sums and untouched result
  tails, and close both foreign mappings with unchanged host lifetime guards.
- [Native capacity](raw512/native-capacity.json): separate-control consumer
  advertises 512 rows; legacy packed control remains limited to 64 rows.
  A 513-row prepare call is rejected before submitting a kernel.

Slot capacity is derived from the admitted runtime cold window. Worker identity,
CUDA IPC descriptor, DMA-BUF result import, HBM input slot, metadata, merge
scratch, and lifetime guard agree on the same bounds. Older completion binaries
are rejected before allocating owners or submitting CUDA waits.

The first 8 MiB probe found a remaining 128-row Python mailbox admission limit.
Both producers exited after this software rejection and released their mappings;
no DMA error was reported. The raw CUDA probe retained its pending wait. After
verifying producer exit, bridge refcount returning to the live model's four
references, and live service health, only that raw probe context was terminated.
The [failure receipt](failed-raw512/transport-probe.json) and
[verified cleanup](failed-raw512/cleanup-admission-failure.json) are retained.
The codec now admits through 512 rows, and each raw shape is preflighted before
its future completion wait. The repeated 4/8 MiB qualifications then passed.

## Measurement method

One loaded runtime provisions the largest 512-row window and compares
`--windows 128 256 512`. Thus the 128-row baseline also has 8 MiB allocated
slots, while transferring only its own rows. All paths use the same frozen
source, native binaries, resident banks, and 32-row CUDA arithmetic. Every path
resets model state, bypasses prefix snapshots and the API queue, and records
zero cached prompt tokens. Timed prefill excludes full-vocabulary observers.
The 64 teacher-forced output logit vectors are compared bitwise; decode timing
includes those observers and is not the serving API decode measurement.

The full Code prompt retains its 23 tool definitions and all 23,820 token IDs.
No compact agent, shortened system prompt, context truncation, or warm cache
is used. Its token ID hash must match the preceding qualified study.
Only the token hash and benchmark receipts are published; the private request
and runtime environment files remain outside the artifact directory.

## Matched cold model performance

| Prompt | 128 rows | 256 rows | 512 rows | Time saved at 512 |
| --- | ---: | ---: | ---: | ---: |
| 2,048 tokens | 12.856s | 11.364s | 10.494s | 18.4% |
| 8,192 tokens | 50.133s | 45.978s | 43.922s | 12.4% |
| 23,820 tokens | 190.065s | 154.909s | 134.896s | 29.0% |

The full Code prompt improves from 125.3 to 176.6 prompt tokens/s. All nine
paths report zero cached tokens and bitwise-equal full-vocabulary logit vectors
for all 64 teacher-forced output steps. Those hashes also match the preceding
qualified 128-row study. The independent 2K startup gate compares every
original 32-row prefill boundary against the original serial path and the
saved reference, with zero maximum logit error.

These are single matched runs, not medians or a throughput guarantee.
[Exact benchmark](probe/cold-benchmark.json), [prefill gate](probe/prefill-parity.json),
[summary](summary.json), and [design PNG](design.png) preserve the evidence.

## Existing worker spans

The retained full Code calls give the following mean native host spans:

| Cold rows | Complete calls per rank | Rank 0 µs/row | Rank 1 µs/row |
| --- | ---: | ---: | ---: |
| 128 | 2,790 | 292.4 | 285.6 |
| 256 | 1,395 | 207.6 | 201.6 |
| 512 | 690 | 156.9 | 151.4 |

The span includes kernel submission and completion synchronization. Ranks and
CUDA overlap, so these values must not be added into an end-to-end critical
path. Filtering retained receipts to their complete windows excludes decode
and the final partial batch. No extra device timing barriers were inserted.
See [span summary](worker-span-summary.json) and compressed phase receipts.

## Deployment

The 512-row configuration was selected after the three-window cold comparison.
It uses the freshly authenticated 512-row native consumer, 8 MiB dedicated
slots, separate completion control, and 32-row CUDA chunks. The endpoint remains
`http://192.168.53.187:8000/v1`, model `glm-sidecard`, with 65,536-token context
admission. The cold benchmark does not establish full-length 65K performance.
Decode arithmetic and weights are unchanged. All serving startup gates passed.
The actual cold API request produced its first text in
10.585s with zero cached tokens, then decoded at
11.72 tokens/s. [API receipt](api-cold-2048.json) and
[health](health.json) record the serving check.

The [qualified source archive](qualified-source.tar.gz) matches all 111 current
runtime Python files and the CUDA completion source byte for byte. Its frozen
tree retains the preceding committed native helper sources. Private environment
files and model weights are not included.

## Validation

All 1,193 standalone sidecard unit tests pass. Actual 4/8 MiB DMA probes and
complete-model parity gates passed on the assigned hardware.

Full-repository manual checks ran in an isolated worktree and reported seven
existing failures: ruff check/format, codespell, typos, clang-format, markdownlint,
and forbidden imports. Unrelated auto-fixes were discarded; changed-file
manual checks pass. See [the full-check summary](full-ci-summary.json).
