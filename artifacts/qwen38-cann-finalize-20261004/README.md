# Qwen W4 CANN finalizer gate, 2026-10-04

The experiment ran on a Threadripper host with four Ascend 310P3 devices.
One-card operator and real-weight layer gates used device 0; the serving A/B
used TP4/EP4. The test used an isolated copy of the qualified
`qwen38-head-unified-runtime-20261001` snapshot at
`/srv/ai/src/qwen38-cann-finalize-gate-20261004`. The source file
`vllm_ascend/models/qwen4_exp/w4_moe.py` had SHA-256
`c3e899547746805e971321c2a80b5fd923e12ddebd78a8c7d278b74e7d8506c6`.
The retained runtime and home launcher were not edited.

The installed `torch_npu.npu_moe_finalize_routing` rejected FP16 routed rows
with FP32 route weights: its 310P tiler required equal dtypes for `expanded_x`
and `scales` ([error log](gate.log)). FP16-converted route weights worked and
were included in every candidate timing. The output was widened to FP32 for
the existing TP reduction.

## One-card results

| Tokens | Current epilogue | CANN epilogue | Max absolute error |
| ---: | ---: | ---: | ---: |
| 512 | 1.766 ms | 0.273 ms | 0.00153 |
| 2,048 | 7.159 ms | 1.005 ms | 0.00133 |

The [operator records](gate-fp16.jsonl) also include 3- and 12-token cases,
changed inputs, all-peer exact-zero checks, and successful changed-input graph
replay at the small shapes. The candidate was roughly equal at the small
shapes; the intended benefit is grouped prefill.

| Real-weight layer partial | Current | CANN | Relative L2 error |
| ---: | ---: | ---: | ---: |
| 512 tokens | 15.579 ms | 13.994 ms | 0.000440 |
| 2,048 tokens | 59.554 ms | 53.417 ms | 0.000437 |

The [layer records](real-layer.jsonl) used layer 0, TP rank 0, real checkpoint
weights, and synthetic activations. They measured the whole local grouped MoE
path without a collective, shared expert, or attention.

## TP4 cold-prefill comparison

Both arms used the same checkpoint, runtime snapshot, 2,048-token prefill
chunks, MTP2, and decode graph configuration. The candidate launcher was a
copy of the qualified launcher with only `"grouped_finalize": "cann_v2"` added
to its `--hf-overrides` JSON ([launcher diff](candidate-launcher.diff)). Three
matching 23,410-token prompts were sent to
each arm, with 32 generated tokens. Every request reported `cached_tokens: 0`.

| Paired case | Current TTFT | CANN TTFT | Reduction | Text identical? |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 73.278 s | 69.712 s | 3.566 s | No |
| 1 | 72.652 s | 69.365 s | 3.287 s | Yes |
| 2 | 72.864 s | 69.338 s | 3.525 s | No |
| **Median** | **72.864 s** | **69.365 s** | **3.499 s (4.8%)** | |

The changed text was a short phrase: “code excerpts” versus “repository
excerpt.” MTP acceptance was 19/26 versus 20/24 in those two pairs. See the
[current requests](service-torch.jsonl), [candidate requests](service-cann-v2.jsonl),
[current server log](baseline-server.log), and [candidate server log](cann-server.log).
The server timing logs independently show 23,410 computed prompt tokens and
zero cached tokens for every long request. They report 0.0-0.1 ms of queue time,
so API queueing did not account for the multi-second difference.

This is a measured cold-prefill improvement, with numerical and generation
differences. The `cann_v2` path remains experimental. Broader logit and
quality tests, plus a reverse-order service repeat, are needed before any
default change. Both services were stopped and all four NPUs were idle after
the comparison.

## Offline attribution and next profiler capture

The 2,048-token epilogue saved **6.154 ms** while the whole real-weight layer
saved **6.137 ms**. The 0.017 ms difference suggests that the epilogue accounts
for nearly all of the layer gain. The model has 48 expert layers and the
23,410-token request spans 11 full 2,048-token chunks plus an 882-token
remainder. A linear estimate for the remainder from the measured 512- and
2,048-token epilogue savings predicts **3.38 s** saved across those layers;
the observed median TTFT reduction was **3.50 s**. This is an inference from
separate timings, not a kernel timeline or a critical-path proof.

An older [September server trace](../qwen38-w4-offline/grouped-native-20260927/server-trace-summary-r1.json)
shows grouped W4 projections at 40.58%, collectives at 14.13%, and
cast/layout/copy tasks at 10.44% of rank 0's summed cold-capture task time.
That trace used an earlier W4A16 grouped backend. Its percentages cannot be
assigned to this native-W4A8 run, and summed task times can overlap.

The saved [September native-INT4 layer trace](../qwen38-w4-offline/grouped-native-20260927/native-trace-summary-r4.json)
is closer to the current backend, but predates later projection optimizations.
Its three 512-token calls contain 12 `IndexSelect` kernels (2.617 ms summed),
three in-place `Mul` kernels (2.290 ms), and 15 `ReduceSum` kernels (4.702 ms).
Other parts of the layer use some of these same operators, so those totals are
not an isolated epilogue cost. The old three-call trace spans 492 ms; the
current unprofiled layer median is 15.579 ms per call. It cannot establish
today's kernel percentages.

The original October 4 comparison did **not** record an NPU kernel trace. The
[epilogue gate](../../tools/qwen4exp/benchmark_w4_finalize_routing_310.py)
and [real-weight layer gate](../../tools/qwen4exp/benchmark_w4_finalize_layer_310.py)
now accept `--trace-dir`. They capture separate current and CANN traces at
2,048 tokens *after* unprofiled timing. A later
[one-card follow-up](profile-followup-20261004/README.md) captured both arms
after the hardware became available; its kernel counts and spans support the
epilogue attribution. The
[experiment runbook](../../tools/qwen4exp/W4_CANN_FINALIZE_EXPERIMENT.md)
has the capture commands.
