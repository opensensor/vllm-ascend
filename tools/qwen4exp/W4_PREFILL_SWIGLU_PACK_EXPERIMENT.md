# Qwen W4 prefill SwiGLU pack experiment

The October 4 native-INT4 layer trace found a second repeated activation cost
after the CANN route finalizer. Across three 2,048-token layer calls, the two
`QwenW4A8PackV310` tasks per call took about 1.08 ms for gate/up inputs and
2.62 ms for the routed down-projection inputs. Separate SwiGLU, multiply, and
cast tasks also run before the down pack. The projection kernels themselves
took about 129 ms across those three calls and are unchanged by this idea.

The 23,410-token cold request needs 16 chunks at the selected 1,536-token
limit. With 48 MoE layers, that is approximately 768 grouped expert layer
invocations. The saved layer trace shows two dependent native projection
kernels per invocation: gate/up, then down after activation. This count is
consistent with the model and chunk schedule; the trace has not revealed an
accidental duplicate projection call.

The existing `QwenW4A8SwigluPackV310` kernel already combines FP32 SwiGLU,
FP16 rounding, and the exact native-INT4 activation pack for short decode
rows. Its implementation strides row batches across AI cores, but the host
adapter and tiler capped it at 128 routes. This experiment raises only those
two validation caps to 20,480 routes, allowing both the selected 15,360-row
chunk and the older 20,480-row maximum. The model retains the opt-in
`grouped_activation=cann_swiglu_pack` path, but its larger-row parity gate
failed. Native INT4 grouped prefill now defaults to the measured
`cann_builtin_fp16` path. An explicit `grouped_activation=torch` restores the
reference path.

## October 4 result and next gate

The custom fused path differs by one packed byte at 15,360 synthetic rows and
was 0.6% slower in a real-weight 1,536-token layer. It is not a candidate for
promotion under the exact-parity gate. The built-in FP16
`torch_npu.npu_swiglu` path was 6.5% faster in that real-weight layer with
bitwise-identical output, although it also changed one packed byte on a
synthetic input. This leaves full-service quality and TTFT as required gates.

The first TP4 service launch reached graph capture after loading real weights
but failed because an older host extension omitted `chunk_fwd_o_vllm`. The
isolated runtime extension was rebuilt from its complete source. The second
launch served three matched 23,410-token cold prompts: mean TTFT improved
**68.920 → 67.225 seconds (2.46%)**, and client prompt tok/s improved
**339.7 → 348.2**. Two outputs were identical to baseline; the third changed
one opening phrase. The user accepted this difference for the native INT4
default. Broader quality and thermal checks remain useful, and the custom
fused-pack path is still experimental. Exact numbers, package hashes, and the
running service status are in the
[experiment record](../../artifacts/qwen38-prefill-swiglu-pack-20261004/README.md).

## Next prefill experiments

1. **Combine built-in SwiGLU with the CANN route finalizer at the selected
   1,536-token chunk.** The earlier, separate 2,048-token finalizer gate cut
   the epilogue from 7.216 to 0.954 ms and the real-weight layer from 59.451
   to 53.363 ms. Its TP4 service arm improved median cold TTFT by 4.8%, but
   changed generated text. Those numbers use the older activation and chunk
   schedule, so they are evidence for testing the combination, not an
   additive speedup prediction. First compare one real-weight layer's output
   error and latency with the new default at 1,536 tokens. Then compare full
   service TTFT, longer-answer quality, MTP acceptance, and concurrency with
   matched prompts. Keep the current service available until a separate NPU
   window. The October 4 finalizer profile is retained in the shared
   workspace for this gate.
2. **Retune projection tiles before increasing the physical expert batch.**
   The [batching sweep](W4_PREFILL_BATCHING_EXPERIMENT.md) found 1,536-token
   chunks faster than 2,048, but a sharp regression between 1,638 and 1,639
   tokens when the native kernel changes tile schedule. A 2,560-token chunk
   would also exceed the current 20,480-route operator cap at top-10. Test a
   25,600-route package with a revised schedule and workspace check, then
   measure four-request capacity and cold TTFT. Raising only the scheduler
   batch previously did not enlarge the actual expert batch or improve TTFT.
3. **Profile the residual activation pack and projection memory traffic.**
   The older 2,048-token trace attributed about 80% of summed device task
   time to the two native W4 projections. The built-in activation removed
   about 2.4 ms from a 1,536-token real-weight layer, while its down-input
   pack still runs separately. A future fused pack must preserve the selected
   FP16 activation's numerical behavior and beat the current combined path;
   the first custom fused pack failed synthetic parity and was slower in the
   real-weight layer. Trace the current default before changing that kernel.

The earlier decode route-lookup gate improved an adverse operator pattern but
regressed the paired four-request serving result, so it ranks below these
prefill opportunities. No further NPU benchmark was started during this review
because the running Qwen service had active requests. Thermal comparison
remains unmeasured.
