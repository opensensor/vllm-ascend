# Completed-pool prefill — 2026-10-05

Experimental resident candidate:
`tools/glm_perf/resident_candidates/kpool_completed_prefill.py`.
Implemented offline after the previous hardware session was cancelled. The
user subsequently authorized the isolated measurement. That measurement loaded
no model and launched no server. A subsequent authorized serving trial is
recorded below. No production defaults were changed.

## Removed work

The current `_write_pools` gathers four key/gate rows and compresses them for
every input token. Only positions ending a four-token pool write a compressed
key. It also constructs state-scatter values for all input tokens, although
only the final pool (and the possible MTP1 rejection pool) needs persistence.

The candidate builds compact indices from existing CPU query boundaries once
per metadata build. Pool phase, position validity and the retained MTP tail
remain device-derived. For a single 640-token request it compresses **160
candidate pools instead of 640**, and prepares **at most five tail rows instead
of 640**. Per request, compression has at most `ceil(query_length / 4)` rows;
an incomplete last pool is masked out. It does not require a device scalar
read or device-to-host compaction. The one small metadata transfer is measured
separately below.

This requires scheduler-contiguous positions within each request. Existing
decode graph sizes 2/8 retain the original writer. Missing CPU metadata,
unsupported pool size and speculation beyond MTP1 also use the original path.
The metadata wrapper clears stale plans before rebuilding them.

Old state is gathered before either scatter, preserving recycled pages.
State writes still precede compressed-key writes. The compression arithmetic,
BF16 rounding and flat-storage scatter implementation are unchanged.

## CPU coverage

**82 tests pass**, covering every starting pool phase, no speculation/MTP1,
FP16/BF16 key caches, empty requests, invalid slots, mixed query lengths,
chunk continuation, recycled state pages, noncontiguous cache views and
shared physical backing. Full allocations are compared, including guards.
Tests also cover decode fallback, wrapper reuse, stale-plan clearing, absence
of scalar/list reads in the writer, and the reduced compression/scatter sizes.
The final test checks source execution under the resident harness's inherited
postponed annotations. A plain slotted plan container avoids `dataclass` looking
up the harness's intentionally unregistered module; tensor operations are
unchanged from the isolated NPU measurement.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  --confcutdir=tests/ut/glm_perf \
  tests/ut/glm_perf/test_kpool_completed_prefill.py -q
```

## Isolated NPU measurement

One Ascend 310P device, CPU affinity `8-11,40-43`, four PyTorch CPU threads,
real BF16 gathers and cache writes. Nine synchronized wall-clock samples per
implementation per case; the order alternates each pair, following warmup.
No model weights or inference requests are involved. CPU dispatch and device
execution are both included; these are not device-only kernel durations.

**24/24 cache parity cases passed**: six request/phase layouts, MTP0/MTP1,
and two physical cache layouts. The first layout adds row gaps and poisoned
guards. The second uses 160 compressed keys per page, contiguous key rows and
FP32 state pages padded to 40960 bytes, matching the qualified 640-token
attention-page geometry. Entire backing allocations match bitwise.

MTP1 medians for the second layout:

| Input shape | Starting position(s) | Original writer (ms) | Compact writer (ms) | Plan copy (ms) |
| --- | --- | ---: | ---: | ---: |
| 640 tokens | 0 | 30.199 | 17.065 | 0.605 |
| 640 tokens | 3 | 24.275 | 9.224 | 0.239 |
| 1280 tokens | 1 | 46.420 | 16.358 | 0.535 |
| 2560 tokens | 2 | 90.989 | 29.898 | 0.696 |
| 157 + 161 + 159 + 163 | 0, 1, 2, 3 | 24.903 | 10.225 | 0.290 |
| 0 + 319 + 0 + 321 | 0, 3, 0, 1 | 23.903 | 9.411 | 0.254 |

Plan construction/copy is outside writer timing because it occurs at metadata
build rather than per layer. Even charging a full plan copy to each writer
call leaves all these cases faster. Timings differ across cases, including
the two 640-token starts; do not interpret that as a proven position effect.
This establishes a local writer improvement, **not end-to-end TTFT or decode
tok/s**. Model integration, all-rank behavior and long-prompt serving remain
unvalidated.

The first probe found that `aclnnIndexSelect` rejects BF16 on this hardware.
Tail gathers now use the baseline's advanced-indexing dispatch, with a CPU
regression rejecting BF16 `index_select`. The failed probe is preserved in
`unsupported-index-select.log`; its timings are not used.

Raw evidence:

- [Row-gap results](strided-results.json)
- [Serving-style page results](page-strided-results.json)
- [Successful run log](page-strided.log)
- Probe: `tools/glm_perf/bench_completed_pools.py`
- Remote staging: `/home/matteius/experiments/glm-completed-pools-20261005`

Results include staged source hashes. AST comparison confirmed the extracted
baseline `_write_pools`, `_cache_tensor` and `_masked_storage_write` exactly
match those in `/srv/ai/src/glm-selective-w3-nz-test-20261004`.

## Next integration gate

Use the resident replacement factory on the next authorized running GLM
instance. Keep context, scheduler batch, expert kernels and MTP unchanged.
Validate metadata binding and cache writes on each rank, then measure a cold
prompt with the compact writer enabled and verify retrieval plus MTP tail
continuation. Check decode still takes the original path. Existing serving
measurements are recorded in the parent hardware report; no fresh baseline
sweep or automatic server launch is scheduled.

## Authorized serving trial

After the isolated result, the user requested starting the server with this
candidate. Launch uses the qualified 640-token batch, 311040 configured context,
TP4, MTP1, full decode graphs `[2,8]`, native mHC and live kpool scoring on 8001.
It retains the existing model ID `glm53-flash-selective-w3`.

- API PID at launch: `1892520`.
- Log: `/home/matteius/experiments/glm-kpool-live-score-20261005/serve-completed-pools.log`.
- `validate-serving.py` applies the candidate after readiness, verifies worker
  and weight identities through the resident harness, runs cold 8K retrieval,
  short c1/c4 and a tool call, and leaves the candidate active on success.
- `serving-audit.py` adds per-rank host counters to distinguish compact prefill
  calls from the original writer's decode path. It reads no device scalars.
- No baseline inference sweep is included. Full-length context remains untested.

**Passed; candidate left running on 8001.** All seven requests succeeded:

| Check | Result |
| --- | --- |
| Cold retrieval, 8199 actual prompt tokens | Correct code; **88.206 s TTFT** |
| Short c1 | **4.955 tok/s**, 256 output tokens |
| Short c4 | **12.794 aggregate tok/s**, all four outputs 256 tokens |
| Tool call | Correct `tool_calls` response |

Earlier 640-token serving runs recorded approximately 89.96–90.18 s cold 8K
TTFT. This candidate is roughly 2% lower in this run, **not a tightly paired
comparison**. The isolated writer improvement therefore does not translate to
a similarly large end-to-end speedup. Decode remains near prior results.

All four ranks report 192 compact-writer calls, 102672 layer-token visits and
a maximum batch of 640, confirming the new path actually executed. The 46
fallback calls occurred during graph capture; subsequent graph replay does not
increment Python counters. All graph states are clean, and worker PIDs plus
weight-storage digests remained unchanged across the switch. The `/v1/models`
response confirms 311040 configured tokens. No full-window or new 20-case
quality run is claimed.

Serving evidence:

- [Summary](serving-summary.json) and [request records](serving-results.jsonl)
- [Final rank receipts](serving-final-status.json)
- [CPU affinity](serving-affinity.json)
- `applied-candidate.py`: exact resident source, retained without reformatting
  to preserve its recorded digest
