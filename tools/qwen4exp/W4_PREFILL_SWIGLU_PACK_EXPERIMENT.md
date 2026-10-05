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
chunk and the older 20,480-row maximum. The model now has an opt-in
`grouped_activation=cann_swiglu_pack` path that feeds the fused pack outputs
directly to native down projection. The default remains the measured torch
path until the NPU parity and service gates pass.

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
one opening phrase, so the path remains opt-in pending broader quality work.
Thermal behavior was not measured. Exact numbers, package hashes, and the
running service status are in the
[experiment record](../../artifacts/qwen38-prefill-swiglu-pack-20261004/README.md).
