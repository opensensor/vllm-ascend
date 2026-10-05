# Resident decode fusion experiments

GLM stays loaded on four 310P devices, port 8001, configured context 311040,
MTP1 and full decode graphs `[2,8]`. Full-length context is still untested.

## Attribution

The paused-worker probe compared graph replay at two and eight rows:

- Native mHC post saves approximately 0.02–0.03 ms per call. Six of 131072
  eight-row output elements changed after FP16 rounding; this experiment
  does not change serving's mHC dispatch.
- Query Hadamard rotation and BF16/FP16 casts cost approximately 0.33–0.43 ms
  per call for synthetic FP16 input. Actual GLM queries are BF16, so the
  subsequent kernel gate uses BF16 as well.

## Fused query rotation

`rotation.cpp` performs seven FP32 Hadamard stages, normalization,
round-to-nearest-even BF16, and the qualified 310P saturating FP16 conversion
in one kernel. FP16 input is widened normally. BF16 input is widened exactly
by gathering its 16-bit words into FP32 storage, avoiding a BF16 cast.
Only the final FP16 query is written back to device memory.

Each eight-core launch processes four heads per tile. Stage gather offsets,
signs, and per-shape tiling tensors belong to the loaded resource instance
and are allocated once before graph capture. Graph replay reads current
query contents. No weights or cache layout are changed.

The versioned bridge is `glm_rotation_v1::launch`, separate from the already
loaded prefill bridge. `rotation-manifest.json` identifies the exact loaded
files. Its asset hashes, rather than a later reformatted source copy, are
the runtime authority. The resident candidate replaces only decode scoring's
query preparation; prefill and unsupported shapes use the existing path.

## Operator gates

- 24 NPU checks passed with exact output equality: rows 1/2/4/8, normal,
  BF16, small, large, impulse, and zero inputs.
- All four resident workers independently passed exact BF16 parity at
  rows 1/2/4/8 when loading the native resource, without reloading weights.
- 20 CPU integration regressions passed: all decode row counts 1–8,
  FP16/BF16, nonzero cache offsets, storage aliasing, and unsupported-input
  fallback. Targeted Ruff checks passed.

Separate-process graph timings with BF16 input:

| Rows | Baseline rotation and casts | Fused native |
| ---: | ---: | ---: |
| 2 | 0.453 ms | 0.057 ms |
| 8 | 0.559 ms | 0.174 ms |

These are operator timings, not serving speedups.

## Full-model gate and ABBA serving comparison

| Run order | Dispatch | c1 tok/s | c4 aggregate tok/s |
| ---: | --- | ---: | ---: |
| 1 | Baseline | 4.936 | 11.006 |
| 2 | Fused rotation | 4.917 | 13.393 |
| 3 | Fused rotation | 4.904 | 11.369 |
| 4 | Baseline | 4.962 | 11.258 |

Each run used the same five short requests, seed 42, greedy sampling and
256-token completion cap. All twenty completions reached 256 tokens.
Worker PIDs and weight-storage digests stayed unchanged. Some generated
text differed, so these are matched settings, not identical token traces.

**Do not promote this as a stable 22% gain.** That first-pair c4 improvement
did not repeat: the reverse pair improved only 0.98%, while c1 was essentially
flat/slightly slower. The isolated operation is faster, but a repeatable
serving improvement has not been established. Baseline dispatch was restored.

The first-pair c4 server iterations had median durations 611.01 ms baseline
and 499.22 ms candidate, with 133 versus 132 four-request iterations. Those
measurements establish that the observed run was faster, not why its advantage
was substantially smaller in the next pair.

Quality remained **17/20**: `instr_reverse` emitted no final answer,
`instr_first` emitted `Red`, and `code_slice` emitted a backtick-wrapped
`lan`. The three misses match the established gate.

To check whether the new arithmetic changed real model activations, `shadow.py`
computed both rotations inside the captured graphs and accumulated mismatch
counts on device. Across the quality suite, warmups and a separate short c4
request, each rank compared **20,635,648 values at two rows** and
**3,899,392 values at eight rows**, with **zero mismatches**. Diagnostic
timings are excluded from the serving comparison. The shadow patch was removed.

## Native mHC RMSNorm: staged, not serving-qualified

`mhc_native_norm.py` replaces the decomposed FP32 norm with the existing
`torch_npu.npu_rms_norm`, retaining FP32 input/weight computation and only
converting to the model dtype at the end. Six CPU tests check precision,
input preservation and epsilon propagation; combined new CPU tests: **26 passed**.

Four-rank isolated graph medians:

| Rows | Existing norm | Native norm |
| ---: | ---: | ---: |
| 2 | 0.032 ms | 0.025 ms |
| 8 | 0.034 ms | 0.025 ms |
| 640 | 0.393–0.403 ms | 0.193–0.203 ms |

This is not bitwise equivalent at every shape: 2/32768 output elements
changed at eight rows and 128/2621440 changed at 640 rows. The tested error
stayed within `rtol=1e-3, atol=2e-6`; maximum absolute differences were
0.000244 and 0.003906 respectively. No full-model speed or quality result
is claimed. The small absolute savings do not justify promoting it alone;
keep it available for a larger fusion bundle.

## Reproduction and final state

The NPU host experiment directory is
`/home/matteius/experiments/glm-decode-fusions-resident-20261005`.
`rotation-loaded.py` preserves the exact loaded wrapper; `rotation.py` has
formatting cleanup. Neither the native manifest's assets nor the loaded
library were overwritten after registration.

Build the binary with the checked-in standalone compiler from the adjacent
native-resident study (CANN 9.1.0, `dav-2002`), then build the bridge with
`ninja -f build.ninja`. The compiler links `acl_rtc` and `ascendcl`.
The manifest records source-asset and binary SHA256 hashes. A different binary
must use a new resource/library version while these workers remain alive.

`validate-rotation.py` prepares the resource and performs four-rank admission
checks. `kpool_rotation_direct.py` is the reversible serving candidate;
`compare.py`, `compare-reverse.py`, and `quality.py` reproduce the gates.
All controller scripts restore baseline and resume serving in `finally`.

Final state: baseline full graphs, MTP1, 311040 configured context, public
port 8001, with the same model weights and workers. Native resources remain
loaded but neither new candidate is selected. No server restart occurred.

The next substantial prefill work remains increasing actual expert batches
and avoiding repeated whole-expert FP16 materialization. These small fusions
are reusable components, not evidence that the main bottleneck is resolved.
