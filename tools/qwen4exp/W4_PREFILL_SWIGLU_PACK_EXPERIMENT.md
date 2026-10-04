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

## Staged CPU-side package

On October 4, the host extension and a coherent five-operator 310P package
compiled successfully on Threadripper. The package contains the four native
W4 operators plus `RecurrentGatedDeltaRuleV310`; it is installed only under
`/srv/ai/src/qwen38-prefill-swiglu-opp-20261004`. The isolated runtime is
`/srv/ai/src/qwen38-prefill-swiglu-runtime-20261004`, with the new extension
and the opt-in launcher setting. Host symbol, recurrent FP16/FP32 variant,
launcher syntax, runtime coherence, and `--show` checks passed. The package
has **not** been executed on an NPU. Build hashes and exact paths are in the
[staging record](../../artifacts/qwen38-prefill-swiglu-pack-20261004/README.md).

## Gate for a later NPU window

Use the isolated host API, tiler, and kernel package as described in the
[310P runtime runbook](../../docs/source/developer_guide/performance_and_debug/qwen38_310p_runtime_runbook.md).
Do not promote the opt-in model path until the NPU parity and timing gates
pass.

```bash
python -m tools.qwen4exp.benchmark_w4_prefill_swiglu_pack_310 --dry-run
ASCEND_RT_VISIBLE_DEVICES=0 python -m tools.qwen4exp.benchmark_w4_prefill_swiglu_pack_310 \
  --output /path/to/new/prefill-swiglu-pack.jsonl \
  --trace-dir /path/to/new/prefill-swiglu-traces
pytest -sv tests/e2e/nightly/310p/single_node/ops/test_qwen_w4_swiglu_pack_310.py
```

The benchmark checks exact equality of all four packed outputs against the
current FP32 SwiGLU, FP16 rounding, and pack sequence, then alternates timed
arms at 5,120, 15,360, and 20,480 routed rows. Separate baseline and fused
profiler captures follow the unprofiled timing at the selected 15,360-row
shape. Enable the opt-in model path for service testing only if exact parity
holds at all three shapes and the fused path is faster at 15,360 rows. The
e2e test adds full prefill-row parity and rejects rows beyond the cap.

If that gate passes, compare the opt-in model path on a real-weight layer and
then matched long-prefill service requests. Record NPU temperature
throughout sustained runs because shorter compute can improve thermal
headroom; the existing one-card traces did not measure temperature or power.
Alternate baseline and candidate service order, begin each arm at a comparable
idle temperature, and sample all four devices at a fixed interval during the
requests. Compare per-device peak temperature and sustained prefill tok/s
separately; the host's current `npu-smi info` reports power as `NA`.
The 48 layers and approximately 16 chunks in the 23,410-token request make
even small per-layer savings worth measuring, but they do not establish a
speedup before the larger-shape kernel is tested.
