# GLM KDA prefill state mask

Status: retained as a graph-safe cleanup after single-card parity and
four-rank real-weight validation. The 640-token server showed no measurable
TTFT improvement. No production launcher was changed. A combined candidate
launched with a
1,280-token scheduler chunk failed 128K cache admission before any request.

The saved October 4 rank-0 profile reports 34 `aclnnNonzeroV2` host API calls
totalling 1,130.7 ms in the captured request. This API time can include
waiting for earlier queued device work; it is **not** a predicted saving. The
profile also reports 34 `ChunkKdaFwd` calls. The GLM W2 prefill source had one
Boolean-indexed zero
assignment per KDA layer, `initial_state[~has_initial_state] = 0`; the trace
places `aten::bitwise_not`, `aten::index`, and `aclnnNonzeroV2` together.
This is a strong source-to-trace match, though the incomplete profiler export
does not contain a Python stack tying each event to the line.

The candidate keeps the FP16-to-FP32 carry conversion and exact zeroing
semantics, but uses a broadcasted `masked_fill_` on the newly gathered FP32
state. This avoids constructing a data-dependent index or synchronizing its
length with the host. Unlike multiplying by a zero mask, it also clears NaNs
and infinities in fresh cache slots exactly as the original assignment does.
The decode path is unchanged.

The one-card probe in `tools/glm_perf/probe_kda_prefill_mask.py` passed exact
output parity for one and four requests, including NaN cold slots. Median
synchronized-call latency over 30 repeats was 0.508 → 0.132 ms for one
request and 0.578 → 0.155 ms for four. This establishes that the replacement
operator runs on the 310P and is faster in isolation, not its full-model
impact. The focused CPU suite passes 13/13 tests across the new mask regression
and existing KDA tests. The new regression checks FP16, BF16, and FP32 cache
inputs at three- and four-dimensional state shapes, includes NaN/Inf cold
slots, and rejects a return to Boolean-indexed assignment. Ruff lint and
format checks pass. The isolated results alone did **not** establish graph
compatibility or end-to-end prefill speed; the serving gate below addresses
the former and found no measurable gain in the latter.

The isolated four-rank source tree at the unchanged 640-token scheduler
setting passed real-weight 128K startup (150,361 KV tokens), FULL_DECODE_ONLY
graph capture, a 32-token fault smoke, and exact 2K/8K retrievals. The matched
same-checkpoint baseline/candidate TTFT pairs were 30.546/30.554 s at 1,733
prompt tokens and 177.325/176.644 s at 7,877 tokens. This is effectively
unchanged at full-model level; the isolated 310P operator saving was hidden
under other work. The strict quality suite was 17/20, with the same
`instr_reverse` (`pial`), `instr_first` (`Red`), and `code_slice` (quoted
``'lan'``) failures known from earlier runs; the tool-call case passed, making
18/21 across both workloads. The 21 requests were all valid. The candidate
server was stopped after the gate, and all four NPUs were idle. This is not a
headline performance claim. Real-weight server and request logs are at
`/home/matteius/experiments/glm-gate-a-20261002/probe-kda-mask.fJdvNp/`:
`baseline.log`, `baseline-fault-2k-8k.jsonl`, `candidate640.log`,
`candidate640-fault-2k-8k.jsonl`, and `candidate640-quality-tool.jsonl`.

The next full-model gate, if this mask is revisited, should verify that
`aclnnNonzeroV2` actually drops from the KDA prefill window without an
equivalent expensive replacement and should test fresh, warm, and mixed
prefill states. Keep the existing launcher as rollback.

The larger-chunk route-tiling candidate remains unpromoted; it needs a
separate 128K-admissible scheduler setting and performance/quality gate.
