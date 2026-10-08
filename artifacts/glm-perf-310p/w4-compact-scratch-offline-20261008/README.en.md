# W4 scratch reduction — offline delivery, October 8, 2026

[Chinese report](README.md)

Added paired gate/up and down W4 entries with smaller UB scratch allocations.
W4 already copies permanent packed nibbles directly, but the generic kernel
still reserves the W2/W3 reconstruction buffers. The new entries remove that
dead capacity while preserving each buffer's remaining live uses. This applies
to the existing 22 W4 layers in the permanent checkpoint; no weight widening or
requantization is required for those layers.

| Schedule | Raw scratch | Decoded scratch | Gathered scratch | Total reduction per kernel instance |
| --- | ---: | ---: | ---: | ---: |
| Generic | 16,448 B | 65,536 B | 32,768 B | Reference |
| W4 decode, M16/K128 | 1,088 B | 4,096 B | 8,192 B | 99 KiB |
| W4 decode, M16/K256 | 2,112 B | 4,096 B | 8,192 B | 98 KiB |
| W4 prefill, M32/K64 | 2,112 B | 65,536 B | 8,192 B | 38 KiB |

These are **UB allocation reductions**, derived by compiling the same C++ header
used by the kernel. They do not represent freed HBM/KV capacity, measured
bandwidth, throughput, or latency. More room for scheduling is an opportunity;
actual speed requires hardware profiling.

## Lifetime checks and complete integration

- Keep raw scratch large enough for `ActivationWide`'s activation packing.
- Keep all 64 KiB of M32 decoded scratch: it holds two INT32 readback planes
  and two FP32 result planes after weight preparation.
- M16 decoded scratch retains the eight FP32 weight broadcast vectors.
- Move W4 scale factors to the start of its otherwise unused gathered buffer;
  retain the full 8 KiB needed by four cached rows at maximum K.
- Keep mask/quantizer scratch and retained row indices unchanged. FP32 products,
  scale arithmetic, accumulation order, output casts and quantization stay intact.

The builder's explicit `--compact-w4-scratch` flag adds
`glm_fused_gate_up_w4.bin` and `glm_fused_down_w4.bin`. Both have static W4
admission and direct-copy preparation. W2/W3 use the existing generic/W3 entries,
including mixed gate/up and down widths. Old bundles keep their existing
dispatch. Prepared weights are mandatory; lookup-table builds are rejected.

The complete path includes dispatch, frozen profiler checks, real-weight gate
reports, resident qualification manifests, permanent checkpoint bundle copying,
and paired full-MoE graph testing. Missing or undeclared W4 pairs are rejected
before loading. All coupled binary/helper checksums remain mandatory. The
profiler now identifies specialized W3 and W4 stages separately from generic
handles while reporting their corresponding pipeline stage.

## Offline evidence

- Decode **v958** compiled on the v956 schedule; prefill **v959** compiled on
  v957. Both use dav-2002 and retain the previous rounded-scale and native route
  column candidates. [Compile log](compile-only.txt) records both completions.
- Raw-scale decode **v960** compiled on v952; raw-scale prefill **v961**
  compiled on v953. [Raw-scale compile log](compile-only-raw.txt) records both
  completions. These accept the current raw FP32 scales and retain the baseline
  scale casts, allowing an isolated W4 scratch test without checkpoint conversion.
- [CPU suite](cpu-tests.txt): **1,622 passed**. Continue excluding the three
  files needing unavailable local upstream/NPU dependencies:
  `test_glm5next_w2_assembly.py`, `test_grouped_gate_up.py`, `test_kpool_ops.py`.
- Tests compile the actual scratch header, check allocation bounds, verify
  builder outputs for generic/W3/W4 entries, exercise mixed bank dispatch at
  2/17/640 tokens with A4/A8, reject partial/tampered bundles, and enforce the
  explicit device flag and real-weight qualification markers.
- [Final dispatch regressions](final-dispatch-regressions.txt): **39 passed**
  after adding the actual GLM method-selection check from the parallel scan.
- [Scratch audit](scratch-audit.json) and its saved C++ source reproduce the
  sizes above using a host compiler.
- [Archive](w4-compact-v958-v961-20261008.tar.gz) contains **84 verified files**:
  kernels, bridges, frozen helpers, build options, compile log and source
  snapshot. [Manifest](compiled-artifact-manifest.json) and
  [archive checksum](archive-sha256.json) record hashes. Delivery helpers and
  kernel/header sources match the compiled provenance.

The SDK compiler initializes ACL for compilation; it never selects a device,
loads a kernel or submits inference. PyTorch backend autoload was disabled for
remote CPU compilation. No serving files, checkpoint bytes, graph configuration
or running service were changed. New kernels remain hardware-unvalidated.
ACLGraph, EP/communication, MTP, multimodal, capacity, cold prefill and decode
performance were not tested this turn because device work remains deferred.

## Deferred full-pipeline gates

The remote artifact root is
`/srv/ai/artifacts/glm-prefill-weight-reuse-20261007`. Once device access resumes,
run the bitwise graph replay/timing comparisons with exclusive device access.
Use raw-scale variants first when the resident checkpoint has raw FP32 scales:

```bash
ROOT=/srv/ai/artifacts/glm-prefill-weight-reuse-20261007
python -m tools.glm_perf.route_columns_probe \
  --baseline "$ROOT/build-v952" --candidate "$ROOT/build-v960-w4-compact" \
  --feature compact_w4_scratch --allow-device-gate \
  --output /tmp/w4-compact-decode-pair.json
python -m tools.glm_perf.route_columns_probe \
  --baseline "$ROOT/build-v953" --candidate "$ROOT/build-v961-w4-compact" \
  --feature compact_w4_scratch --allow-device-gate \
  --output /tmp/w4-compact-prefill-pair.json
```

These commands were **not executed**. They compare complete native routed MoE
pipelines for W2/W3/W4, A4/A8, 2/8/17/640 tokens with changed graph inputs and
alternating replay timing. This is an operator gate, not full-model qualification.
Next require independent real-weight gates, including W4, and target/draft graph
replay before measuring cold long-prompt TTFT and c1/c4 generation end to end.

The rounded-scale comparisons use v956/v958 and v957/v959 instead.
The frozen v958/v959 builds also require the prior permanent rounded-scale
contract. Raw resident scales must not use them. To combine the candidates,
the CPU `prerounded_scale_checkpoint` exporter can use the new verified bundle;
it now includes both W4 entries. No W4 code conversion is needed for existing W4
banks. The prior lossless W3-to-W4 disk exporter remains a separate experiment
with its explicit memory allowance. Neither actual checkpoint export nor model
startup was run in this delivery.

## Dispatch notes from the parallel code scan

The source selects `run_stateful_kda_310` for NPU inputs in
`glm5next_w2/model.py::_bind_eager_kda_forward`; prefill uses `chunk_kda_fwd`.
The Python timestep loop in `kda.py` belongs to its CPU oracle/fallback branch.
The class/function names contain "eager", but that does not establish Python
recurrence on the device. No live worker dispatch was queried this turn.

The shipped decoder's mHC path uses patched `MHCPreOp`/`MHCFusedPostPreOp`.
Its `hc_*_fn`, base and scale parameters are created in FP32 in
`glm5next/model.py`, so `.to(float32)` in the host reference is not evidence of
a per-forward weight allocation. KDA gate scale/bias operands are already
prepared after loading in `kda_310.py::prepare_kda_gate_weights`. The next scan
should trace the actual native/patch paths and checkpoint flags before ranking
remaining activation conversions, reductions and FP32 projection work.

The permanent `glm_native_int4` loader assigns `module._method` to
`NativeInt4MoEMethod`; `Glm5NextW2MoE.method` consumes that assignment. Its native
call uses resident banks and returns FP32, bypassing the legacy FP64 host oracle,
per-expert CPU staging, and legacy Python SwiGLU path. Prepared geometry rejection
raises rather than silently entering the legacy scheme. The legacy paths remain
important for other configurations. This dispatch is additionally checked through
the actual GLM routed-forward method in the final CPU regression.

The shared `csrc/gmm/w2_blocked_dequant_matmul_v310` family expands weights to
FP16 before its Cube products. Its GM workspace traffic model should not be
attributed to the separate native `tools/glm_perf/glm_fused_moe.cpp` pipeline.
That pipeline reconstructs W2/W3 signed INT4 codes or copies W4 nibbles, then
uses INT4 Cube products. The prior scale optimization is also in its
`PrepareScales`, guarded by `GLM_PREROUNDED_WEIGHT_SCALES` and the permanent
`fp16_rounded_fp32_v1` marker. The CPU exporter stores `scale.half().float()`
in FP32 tensors; config, manifest, safetensors metadata and bank markers agree.
FP32 scale multiplication and accumulation remain. This does not identify a
Qwen format or establish that a Qwen scale path has the same contract.
