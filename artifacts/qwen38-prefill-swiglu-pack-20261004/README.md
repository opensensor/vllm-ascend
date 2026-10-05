# Qwen W4 prefill SwiGLU experiment, October 4

The custom fused SwiGLU-plus-INT4-pack path did not pass its synthetic exact
parity gate and did not improve the measured real-weight layer. A second,
opt-in candidate uses `torch_npu.npu_swiglu` in FP16 before the existing
native-W4 down projection. It improved both the real-weight layer and all
three matched cold-prefill requests. The Qwen candidate service remains up on
port 8001 for handoff.

## Isolated paths

| Item | Path or SHA-256 |
| --- | --- |
| Build source and results | `/srv/ai/src/qwen38-prefill-swiglu-src-20261004` |
| Runtime | `/srv/ai/src/qwen38-prefill-swiglu-runtime-20261004` |
| Coherent OPP vendor | `/srv/ai/src/qwen38-prefill-swiglu-opp-20261004/vendors/qwen38_swiglu_prefill_transformer` |
| Five-operator installer | `725e5bfe346acf4c35e42919762c85159d90bf9c25f1bf21fa4ba7af0cdae63c` |
| Rebuilt runtime host extension | `c46bb512374b2e06644949410fcd0329eeb156c2323bae5c64c679e50e62122e` |
| Coherent host API library | `f1a9515baa78ccf4318415ec4bad710d79ad6390515d834ff74217b3e5ea89c6` |

The retained 1,536-token scheduler and native-W4 configuration remain the
comparison point. The saved three-case baseline is
[`service-batch1536.jsonl`](../qwen38-prefill-batching-20261004/service-batch1536.jsonl).
No baseline restart is needed.

## Measurements completed before deferral

| Gate | Result |
| --- | --- |
| Custom fused pack, 15,360 synthetic rows | One packed byte and eight replicated sum lanes differed from the current torch path; 20,480 rows also differed. |
| Custom fused pack, real layer 0/rank 0, 1,536 tokens | Bitwise output match; median 37.039 ms versus 36.834 ms torch, 0.6% slower. |
| Built-in FP16 SwiGLU plus pack, 15,360 synthetic rows | Median 2.684 ms versus 5.043 ms torch, but one packed byte and eight sum lanes differed. |
| Built-in FP16 SwiGLU, real layer 0/rank 0, 1,536 tokens | Bitwise output match; median 34.504 ms versus 36.904 ms torch, 6.5% faster. |
| Production-shape chunk-GDR head-state and new-value tests | 2 passed on one 310P. |

The real-layer timing covers grouped dispatch, both native-W4 projections,
activation, and route finalization. It excludes attention, collectives, and
service scheduling. The synthetic mismatch prevents a claim of universal
bitwise parity even though the tested checkpoint layer matched. Copies of the
[custom fused layer result](real-layer-v2-20261004.json),
[built-in layer result](real-layer-builtin-20261004.json),
[synthetic built-in probe](builtin-swiglu-probe.log), and
[parity diagnosis](diagnose-swiglu-parity-v2.log) are saved here. The
[formula probe](formula_probe.py) helps isolate the FP16 rounding difference.
The originals are under the build source's `results/` directory.

## Service attempt and packaging repair

The first TP4 built-in candidate loaded all real weights, then exited during
graph capture because its host extension lacked the `chunk_fwd_o_vllm` Torch
registration. That extension had been built from an older isolated source
snapshot. The service produced no request or tok/s result; its log is
`results/serve-builtin-20261004.log`.

The host extension was rebuilt directly from the isolated runtime source,
which contains both chunk-GDR registrations, without reinstalling the OPP.
A standalone load of the rebuilt extension confirmed
`chunk_gated_delta_rule_fwd_h`, `chunk_fwd_o_vllm`,
`npu_qwen_w4_a8_pack_310`, and `npu_qwen_w4_a8_int4_matmul_310` are registered.
The embedded OPP also contains `chunk_fwd_o_vllm` kernel configuration.
The second TP4 startup and three real-weight requests completed successfully.

## Matched cold-prefill service result

Both arms used the same native-W4, 1,536-token grouped chunk, 2,048-token
scheduler batch, TP4, MTP2, 23,410-token prompts, and 32-token deterministic
decode setting. The saved baseline was not restarted.

| Case | Baseline TTFT | Built-in FP16 TTFT | Saved | Output |
| --- | ---: | ---: | ---: | --- |
| 0 | 69.080 s | 67.428 s | 1.652 s | Exact text match |
| 1 | 68.729 s | 67.152 s | 1.577 s | Exact text match |
| 2 | 68.950 s | 67.093 s | 1.857 s | One phrase changed |

Mean TTFT was **68.920 → 67.225 s**, saving **1.695 s (2.46%)**.
Prompt tokens divided by client TTFT increased from **339.7 → 348.2 tok/s**
(2.52%); this includes service overhead and is not an isolated kernel rate.
The third answer changed only “code excerpts” to “repository excerpt” in its
opening sentence. All three requests returned 32 tokens and identical prompt
token counts; the first two had the same output hash as baseline. MTP
acceptance was identical for cases 0 and 1 and changed from 19/26 to 20/24
for case 2. The user accepted this small difference and promoted built-in
FP16 SwiGLU to the native INT4 grouped-prefill default. Strict bitwise service
parity still does not hold; the explicit `grouped_activation=torch` override
remains available for comparisons.

The [candidate request record](service-builtin-v2-20261004.jsonl) holds the
per-request usage, hashes, TTFT, and decode times. The first two decode rates
were close to baseline; three 32-token continuations are too short to infer a
decode throughput change. Temperature was not captured during this run because
the telemetry script was missing from the isolated source; it is now staged
for a later sustained comparison.

## Runtime and remaining gate

The isolated runtime launcher currently selects
`grouped_activation=cann_builtin_fp16`. The prepared
[`start-builtin-service.sh`](start-builtin-service.sh) writes fresh `v2` logs;
[`service_client.py`](service_client.py) accepts `--arm builtin_fp16` and
recorded the three cold prompts. The source default now selects this path for
native INT4 grouped prefill. A broader quality check and matched thermal
comparison remain before claiming general production qualification. The
service is running on port 8001; do not shut it down after reading these
results.
