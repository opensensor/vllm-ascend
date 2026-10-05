# Qwen W4 finalizer profile follow-up, 2026-10-04

The isolated runtime at
`/srv/ai/src/qwen38-cann-finalize-gate-20261004` retained the exact
`w4_moe.py` source from the earlier A/B (SHA-256
`c3e899547746805e971321c2a80b5fd923e12ddebd78a8c7d278b74e7d8506c6`).
The profiler harnesses were updated in that snapshot. All four 310P devices
had no running processes before this follow-up began.

## One-card epilogue and real-weight layer

The one-card gates repeated the unprofiled timing first, then captured three
calls per arm at 2,048 tokens. The profiler emitted an “Incorrect schedule”
warning at context exit, but parsed `kernel_details.csv` files for all four
captures. Each epilogue trace contains exactly three full operator sequences;
each layer trace contains six native W4 projections, as expected for three
layer calls. The trace spans agree closely with the unprofiled timings.

| Scope | Baseline median | CANN median | Saved | Baseline trace span, 3 calls | CANN trace span, 3 calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Epilogue | 7.216 ms | 0.954 ms | 6.262 ms | 21.490 ms | 2.791 ms |
| Real-weight layer, rank 0 partial | 59.451 ms | 53.363 ms | 6.088 ms | 178.182 ms | 159.944 ms |

The [epilogue records](epilogue.jsonl) also passed mixed and all-peer parity
at 3, 12, 512, and 2,048 tokens; changed-input graph replay passed at 3 and
12 tokens. The [layer records](layer.jsonl) used real checkpoint weights and
synthetic activations. At 2,048 tokens, relative L2 error was 0.000437.

The [epilogue trace summary](epilogue-summary.json) shows the baseline's three
calls using six `IndexSelect` tasks (6.769 ms summed), three in-place `Mul`
tasks (6.295 ms), three large casts (4.765 ms), and three `ReduceSum` tasks
(3.603 ms). The candidate uses three `MoeFinalizeRoutingV2` tasks (2.348 ms)
plus nine small casts (0.354 ms). The nearly 18.7 ms difference across three
calls matches the unprofiled epilogue saving.

The [layer trace summary](layer-summary.json) shows six native W4 projection
tasks in each arm: 128.981 ms baseline versus 129.172 ms candidate, summed
over three layer calls. The layer span fell by 18.238 ms across those calls.
This local trace supports the epilogue as the source of the layer speedup;
the projection kernels were effectively unchanged. Summed task durations are
work attribution, not an additive full-service critical path.

The complete profiler directories, including their generated metadata and
logs, are retained in [raw-traces.tar.gz](raw-traces.tar.gz).

The next repeated activation cost is the down-projection input path. In each
2,048-token layer call, the existing pack kernel takes about 2.62 ms for its
20,480 routed rows, after separate SwiGLU and multiply tasks. A
[source-only fused-pack experiment](../../../tools/qwen4exp/W4_PREFILL_SWIGLU_PACK_EXPERIMENT.md)
now permits that row count in the existing decode-only fused kernel and queues
exact parity and timing checks. It has not been built or run on NPU hardware;
the model does not use it for prefill.

## Service capture and hardware release

A four-rank baseline service was started with a four-iteration profiler
configuration. It was stopped during checkpoint loading when the user asked
for the NPUs back, so **no four-rank service trace was captured**. The partial
[startup log](service-torch.log) is retained only to document that attempt.
The service process and worker processes exited; `npu-smi info` then showed
no running processes on all four devices. No candidate service was started in
this follow-up.
