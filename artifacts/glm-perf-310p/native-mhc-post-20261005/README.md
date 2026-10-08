# GLM packed-W3: native mHC post and FP16 rounding

## Purpose

The existing mHC post mixer forms a large broadcast intermediate before the
next sublayer. Earlier 1,280-token prefill experiments hit an allocation failure
here. The staged Python streaming mixer reduced that workspace, but retained
several passes over device memory and separate FP16 cast buffers.

`GlmMhcPostV310` loads a tile of all four residual streams into 33 KiB of local
storage, reuses those values for all four outputs, adds the layer output, and
rounds FP32 → FP16 → FP32 before the final device-memory write. It allocates one
output plus the CANN library workspace. It does not expand an expert matrix or
change expert routing. Its intended second benefit is making larger **actual
expert batches** practical by reducing the competing activation workspace.

Selection is explicit and off by default:

```json
{"ascend_glm_mhc_fp16_state": true, "ascend_glm_native_mhc_post": true}
```

This is incompatible with selecting the separate Python experiment
`ascend_glm_prefill_mhc_post`. Only contiguous four-stream FP16-state prefill
with 640–32,768 tokens and a width divisible by 32 (at most 16,384) selects the
kernel. Decode, small tails, BF16 state, and unsupported layouts retain the
existing path. The final standalone mHC post is outside this change.

## Operator measurements

One Ascend 310P device, FP16-rounded inputs in FP32 storage, width 4,096,
nine timings with rotated implementation order. Every variant includes its
FP16 round trip. These are operator measurements, not model throughput.

| Tokens | Original ms | Streaming ms | Native ms | Original peak MiB | Streaming peak MiB | Native peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 640 | 6.455 | 4.911 | 1.282 | 280.3 | 100.0 | 42.0 |
| 1,280 | 12.770 | 9.595 | 2.400 | 560.6 | 200.0 | 82.0 |
| 2,560 | 25.124 | 18.645 | 4.563 | 1,122.0 | 400.0 | 162.0 |

Peak means incremental allocated bytes, including the output; it excludes
already-resident inputs. See [raw timings](operator-results.json).

The changed accumulation order is not bitwise equivalent to the original
`einsum`. At 1,280 tokens, 937 of 20,971,520 rounded outputs differed; maximum
absolute difference was 0.00390625. The separate Python streaming mixer had the
same mismatch count and maximum difference. That does not by itself prove
bitwise equality between the two candidates.

## Validation and fixes

- 57 CPU tests passed: 40 new native dispatch/configuration/order checks and
  17 existing streaming regressions.
- CANN 9.1.0 operator compilation and supplemental PyTorch binding compilation
  succeeded; packages were first built without using NPUs.
- After hardware authorization, **15 NPU tests passed**, including realistic
  shapes through 2,560 × 4,096, a 32-value tail, graph replay with all inputs
  changed, queued temporary-input lifetimes, invalid inputs, and cast edges.
- Initial edge tests exposed two issues, corrected before timing: torch_npu's
  qualified FP16 cast saturates overflow and nonfinite values; 310P vector
  comparisons need full 64-value FP32 repeats. The tail mask now pads safely.
- Finite parity uses `rtol=1e-3, atol=2e-6`; isolated cast-edge parity is exact.
  This is not a claim of general bitwise parity or full-model quality parity.

Evidence: [hardware gate](hardware-v4.log), [initial cast probe](boundary-probe.log),
[final kernel build](build-v4.log), [binding build](binding-build.log).

## Reproduction

All source edits are in shared main. The CANN source mirror is only a compiler
input. The separate package does not replace a live server's OPP or binding.

- Kernel source: `csrc/gmm/glm_mhc_post_v310/`.
- Build-only entry point: `build-only.sh`.
- Hardware gate and matched operator benchmark: `test-kernel.sh`.
- The supplemental binding is registered once by `native_mhc_extension.py`.
- `serve-candidate.sh` retains MTP1, four-way TP, full decode graphs `[2,8]`,
  and port **8001**. Arguments 18/19 select `baseline|native` and resident
  controls `on|off`. Resident controls bind to loopback; normal serving disables
  development endpoints and binds to all interfaces.
- `compare-resident.py` applies identical worker affinity, clears prefix caches
  between rounds, compares baseline/native/native/baseline, and records worker
  PIDs and weight-storage digests. The candidate logs selected prefill shapes
  once per mHC module on rank zero so a silent fallback is visible.
- `launch.py --context -1` requests upstream automatic sizing against the
  profiled KV budget. A configured maximum is not a tested full-length prompt.

## Serving qualification

A resident **baseline/native/native/baseline** comparison completed on all four
NPUs with MTP1, full decode graphs `[2,8]`, 640-token chunks, fixed worker
CPU affinity, and prefix caches cleared before each request. Worker PIDs and
weight-storage digests remained unchanged. All four 8,197-token retrieval
requests answered correctly.

| Mode | Cold TTFT seconds | Mean seconds |
| --- | --- | ---: |
| Baseline | 95.6645, 94.9806 | 95.3226 |
| Native | 89.8840, 89.5947 | 89.7393 |

The native candidate reduced matched cold TTFT **5.86%**. See
`resident640-results.jsonl` and `compare-resident.log`. This retrieval check
is not the complete 20-case quality suite.

Auto-fit configured 311,040 tokens (311,513 cache-token equivalents, minimum
rank KV allocation 5.12 GiB); a full-length prompt was not tested. Short
decode became slower at that capacity with both mixer variants. The fixed
graph selector gathered/scored the configured key range even for short
requests. Clamping its stale constructor bound to 311,040 did not help:
the page table already imposed that bound. See the separate
`../kpool-live-score-20261005/` study for the live-length scorer candidate.

The diagnostic server was stopped and the devices released. The user deferred
all further server launches, so the larger 1,280-token prefill experiment is
still pending. Keep the native flag opt-in until that test and broader serving
quality qualification finish.
