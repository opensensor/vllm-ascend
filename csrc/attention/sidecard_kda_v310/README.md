# Sidecard Native KDA Decode Candidate

This isolated candidate implements the one-row recurrence from
`tools/sidecard/layer_math.py:recurrent_kda`. It is uncompiled and unqualified
on hardware until T12; portable reference tests are not native-kernel evidence.
It does not register itself or change the existing recurrent operator.

## Math and Ownership

- Original q/k/v are contiguous FP16 `[1,H,128]`; H is 8..128 divisible by eight.
- Gate is FP32 `[1,H,128]`, beta FP32 `[1,H]`, carry FP32 `[H,128,128]`.
  Carry indexes **head, value, key** and is mutated in place. All inputs share
  the owner-local NPU device. No q/k normalization, beta, or carry is downcast.
- Normalize q/k in FP32 with epsilon 1e-6; multiply normalized q by `128**-0.5`
  as a separate FP32 step. Apply FP32 exponential gate before innovation.
- Compute prediction, innovation, update and output in FP32 with separate
  vector multiply/add operations; output alone rounds to FP16, nearest even.
- Value-row groups are disjoint, with sixteen rows per task. No atomics,
  peer communication, speculative state, or cross-call caches are involved.
  Calls on the same carry must be ordered on the owner's stream.
- Query/key/value/gate/beta must not alias the mutable carry. A caller must
  keep all buffers alive through stream completion; carry remains owner-local.

## Numerical Contract

These are **proposed, fixed before device tests** bounds, not measured accuracy:
one-step carry `atol=5e-5, rtol=5e-5`, FP16 output `atol=2e-3, rtol=2e-3`,
256 changed-input sequential steps carry `atol=1e-3, rtol=1e-3`.
Carry/output must remain finite. FP32 reduction order, vector exponential and
reciprocal square root may differ from the eager framework; bitwise equality
is not claimed. T4/T11 must review these bounds against the frozen math contract
before accepting the candidate. A failed bound disables the candidate; it must
not be loosened after observing errors to obtain a passing benchmark.
Full-model logits, routing, quality, long context and speed gates are separate.

## Private ABI and Integration

The CANN name is `SidecardKdaV310`, kernel symbol `sidecard_kda_v310`.
The torch namespace is `_C_ascend`, matching existing custom operators.
The operator is `_C_ascend::sidecard_kda_decode_310`; the schema registered
inside that namespace (also in `tools/sidecard/npu_kda.py`) is:

```text
sidecard_kda_decode_310(Tensor query, Tensor key, Tensor value,
                      Tensor gate, Tensor beta, Tensor(a!) state) -> Tensor
```

T11 must explicitly enable `-DSIDECARD_BUILD_KDA_V310=ON` in the custom OPP
build and include `sidecard_kda_v310` when filtering with `ASCEND_OP_NAME`;
the ordinary build discovers this folder but leaves the candidate disabled.
`SIDECARD_BUILD_KDA_V310` is a CMake cache option, **not an environment variable**.
Append `-DSIDECARD_BUILD_KDA_V310=ON -DASCEND_OP_NAME=sidecard_kda_v310` to
the frozen custom OPP CMake configuration (or include it in the integration
operator selection list). This package requires the existing parent CMake
helpers; it cannot configure standalone. Build with the frozen CANN
9.1.0/Ascend310P compiler and torch_npu 2.13.0.rc1 ABI, generate
`aclnnSidecardKdaV310` through the existing `add_modules_sources` machinery,
include `sidecard_kda_310_torch_adpt.h` after the existing command-dispatch and
`c10_npu::OptionalNPUGuard` declarations, and register the schema plus
PrivateUse1 implementation under
`TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)` in its isolated integration
tree, following `csrc/torch_binding.cpp`. The Python
resolver uses `torch.ops._C_ascend.sidecard_kda_decode_310`. The C++ adapter's
`vllm_ascend` namespace is independent of Torch registration. Shared binding
files remain untouched here.
The torch adapter passes the carry both to CANN's `state` input and its
`state_out` output; the kernel supports this intentional in-place alias.
It rejects non-NPU inputs and selects `query.device()` with an owner-local
`OptionalNPUGuard` before output allocation and native dispatch, including for
direct `torch.ops` callers with another current NPU context.
Verify generated ACLNN output-alias handling in the build/device gate.
Provide a meta adapter returning an empty tensor with the value shape/dtype,
while preserving schema mutation annotations for graph capture integration.

Tiling accepts only these static ranks/shapes; declares zero external workspace
bytes. This and the following memory figures are offline code estimates,
pending compile and device gates.
Per-core UB uses 3,840 FP32-buffer bytes plus 320 FP16-buffer bytes (4,160 total),
excluding `TPipe`/compiler overhead. Blocks are min(vector-capable cores, H*8).
The host follows the established v310 `GetCoreNumAiv()` path, falling back to
`GetCoreNumAic()` when legacy 310P metadata reports only AI cores. Tasks stride
across blocks. All DMA row lengths are 32-byte aligned. The candidate is not a
multi-row prefill implementation; unqualified or multi-row calls keep the
original eager recurrence at the integration dispatch site. The private
Python adapter rejects unsupported calls rather than silently using fallback.

## Probe

Only an orchestrator holding the hardware lease may invoke the probe. Configure
an absolute library path, its SHA256, physical chips `[2,3]`, head counts, warmup
and repeats; do not use this config if device visibility remaps chip indices.

```json
{
  "library": "/absolute/isolated-build/operator.so",
  "library_sha256": "<sha256>",
  "physical_chips": [2, 3],
  "heads": [16, 32],
  "sequence": 256,
  "warmup": 10,
  "repeats": 100
}
```

```bash
python -m tools.sidecard.kda_probe --config probe.json --output rank0.json --rank 0
python -m tools.sidecard.kda_probe --config probe.json --output rank1.json --rank 1
```

The probe measures first call separately and reports warmed synchronized wall
time including launch overhead against the eager recurrence. It compares
changed inputs and nonzero carry on device outside timing loops. It imports
accelerator modules lazily; `--help` is safe offline. Synthetic operator timing
never constitutes GLM server throughput or real-weight/model qualification.
