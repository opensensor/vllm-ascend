# Resident Sinkhorn normalization investigation

## Status

**Running on port 8001** as `glm53-flash-selective-w3`, resident candidate
`completed_pools_sinkhorn`. Context remains 311040, MTP1, full decode graphs
`[2,8]`, and the completed-pool prefill improvement. Full-length context is
still untested. All 76 offline ABI, dispatch, fallback, and reference checks pass.

## Serving results

| Workload | Comparison | Fused result | Change |
| --- | --- | --- | --- |
| c1, 256 tokens | Earlier completed-pool run: 4.9549 tok/s | 5.1814 tok/s | +4.6%, historical comparison |
| c4, 256 tokens each | Resident rollback control: 11.8742 aggregate tok/s | 12.3173 aggregate tok/s | +3.7%, one comparison |

The earlier c4 record was 12.7939 tok/s; comparison to it initially suggested
a regression. One c4-only resident rollback check, without reloading weights,
measured 11.8742 tok/s with the original normalization. The fused candidate
was then restored. This is one comparison, not a repeated statistical estimate.
Prefill batching and generated text differed, so aggregate decode windows
also differ; per-request c4 median was 3.1246 tok/s fused versus 3.1486 control.
Do not present this as a uniform per-request improvement.

All five throughput responses reached 256 tokens and the tool-call check
passed (6/6 serving checks). The full 20-case quality suite was not rerun.
Each resident worker independently passed 20 exact FP32 cases before enabling
dispatch. A real-request shadow run compared 60480 entries per rank,
including capture/warmup entries, with **zero FP32 or FP16 mismatches**.
Generated text across serving comparisons was not identical; exact operator
parity is not a claim of identical whole-server outputs across scheduling.

Worker PIDs 2332905/2333449/2333970/2334336 and weight-storage identities
remained unchanged through shadow, fused, rollback, and restoration switches.
Final status confirms graph mode, clean graphs on every rank, and unpaused
serving. API PID is 2330991. Manifests, timings, shadow receipts, and final
status are recorded alongside this README.

This candidate targets small decode batches only. Larger prefill batches
retain the original normalization; no cold-prefill improvement is claimed.

## Operator results

The first 120 cases using uniform pairwise or sequential reduction matched
after FP16 rounding, but had small FP32 differences. Investigation identified
the exact reference order: **sequential column sums, pairwise row sums**.
Mixed order (`order=2`, now the wrapper default) passed 60/60 cases with exact
FP32 equality: rows 1/2/4/8, iterations 1/20/64, and five logit scales from
zero through 100. Twenty changing-input graph replays also matched exactly.
These results are in `native-results.json`; initial results are retained in
`native-orders01.json`.

At the model's 20 iterations and epsilon `1e-6`, graph timing was:

| Rows | Original normalization | Native normalization |
| --- | --- | --- |
| 1 | 0.2060 ms | 0.0286 ms |
| 2 | 0.2253 ms | 0.0281 ms |
| 4 | 0.2296 ms | 0.0282 ms |
| 8 | 0.2334 ms | 0.0283 ms |

These are isolated normalization timings, excluding softmax and other mHC
work. They are not whole-model speedup estimates. The first standalone graph
attempt used default JIT compilation and rejected ReduceSum capture; the
rerun explicitly disabled JIT compilation, matching the serving runtime.

Before this fusion was enabled, the completed-pool server was profiled with
MTP1, 311040 configured context, and decode graphs `[2,8]`. That four-rank
trace contains **zero native Sinkhorn tasks**. It describes the earlier
unfused configuration; the current serving candidate enables native
normalization for eligible decode shapes.

In the approximate decode interval (first recurrent KDA task through trace
end), each rank has 65120 ReduceSum tasks plus 65120 associated MemSet tasks.
Across ranks these sum to 207–238 ms. Addition and division are further work.
These are summed task durations, not a measured critical-path saving, and not
every reduction is necessarily Sinkhorn. Source inspection identifies the
39-normalization loop at 20 iterations as a large repeated contributor.

The trace and task summary are under the neighboring
`projection-nz-resident-20261005` study. Existing whole-model reference:
4.955 tok/s c1, 12.794 aggregate tok/s c4, cold 8199-token retrieval TTFT
88.206 seconds. The serving results above describe the new measurements.

## Qualified implementation

`normalize.cpp` accepts the existing `softmax(logits) + epsilon` result.
It keeps all subsequent 4×4 alternating column/row normalizations in local
vector memory, writing only the final matrix back to device RAM. It offers
pairwise, sequential, and mixed reduction orders for parity investigation. Unlike
the old scalar full-Sinkhorn operator, it retains the qualified softmax and
uses vector arithmetic for normalization. This is a distinct implementation.

`tools/glm_perf/sinkhorn_native.py` owns graph-stable per-shape configuration
buffers for one through eight rows. No weight copies, host tensor scalar
reads, or new environment variables are needed. The resident candidate
enables dispatch only for its qualified small shapes, model epsilon and
iteration count. Larger prefill calls retain the original implementation.
The numerical, graph replay, actual-activation and bounded serving gates
are recorded above; the full quality suite and long-prompt gate remain pending.

Remote build directory:
`/home/matteius/experiments/glm-sinkhorn-resident-20261005/`.
The binary is `normalize-v1.bin`; bridge is `glm_sinkhorn_bridge_v1.so`.
The serving driver loads the versioned bridge and validates it independently
on all four workers before enabling dispatch.

## Live-control collision

At approximately 22:26:53 UTC, another controller's completed-pool
`direct-both` transaction overlapped this session's `sinkhorn_probe`
transaction. The resulting `generation has not been prepared` exceptions
left unequal numbers of pending replies in the executor response queues.
Repeated status calls can return plausible old status, so that alone is
insufficient evidence that the queues are aligned.

No Sinkhorn measurements were produced. All model worker PIDs and weight
storage remain present. Server was left paused while exclusive control was
requested, because resuming against mismatched replies risks incorrect
execution. A normal restore attempt was rejected for malformed worker
acknowledgments. Do not claim restoration succeeded or resume solely from a
clean-looking repeated status response; first recover the control transport.
Saved evidence: `conflict-status.json` and remote `probe.log`/server log.

A later read-only `/is_paused` check returned `false`: another controller
resumed the server. This session did not perform that recovery and has not
validated its result. Further live mutations were deferred pending
exclusive harness ownership.

After the user authorized this retest, the old server had already shut down
and all NPUs were free. The test began with isolated kernel checks and one
fresh qualified GLM launch. All later comparisons reused that server.
