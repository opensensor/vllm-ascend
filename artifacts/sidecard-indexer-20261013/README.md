# GLM sidecard fused sparse-indexer rotation and larger context

The CUDA indexer replaces seven eager FP32 Hadamard butterflies and their
normalization/casts with one Triton kernel. Each vector preserves the original
add/subtract direction at every stride, the FP32 normalization constant, and
BF16 then FP16 rounding. Floating-point fusion is disabled. The CUDA output
remains FP16; NPU and CPU owners keep the existing reference path.

`cuda_indexer_hadamard` is a boolean runtime configuration switch, defaulting to
false. It is validated before model allocation and applied only to CUDA-owned
indexers. This is not an environment variable. The production native kernels,
weights, expert placement, 1024-row cold windows, 32-row CUDA arithmetic, and
cross-vendor DMA ownership protocol remain unchanged.

## Qualification method

The standalone probe checks complete output bytes across five vector counts
and five scales, plus four graph replays with changed inputs. It rejects invalid
width and dtype before launch. The common 32-vector rotation falls from
364.28 to 27.97 microseconds in this host-inclusive microbenchmark. These values
measure one transform and do not imply that the whole model accelerates 13 times.

The complete-model benchmark loads one runtime and switches only indexer
rotation between the reference and fused paths. Both modes reset model state,
disable prompt prefix snapshots, and process identical 2K, 8K, and complete
23,820-token Code prompts. All 64 output-vocabulary hashes must match the
authenticated preceding 16 MiB benchmark. Decode timings exclude logit observers.
Prompt IDs and all 23 tools in the private Code request remain unchanged.

The context test reserves 131,072 tokens, processes 131,008 actual prompt tokens,
and completes 64 outputs. It snapshots the full prompt state and repeats all
output steps with the original teacher tokens, requiring identical full-vector
hashes. This checks cache capacity, sparse selection, finite output, and repeat
stability. It does not establish task accuracy at this context length.

The checkpoint declares 1,048,576 positions. The previous 65,536 admission was a
runtime setting. Attention and compressed-pool caches remain on the RTX; Ascend
memory holds cold expert weights. This change tests a larger CUDA cache allocation
and does not pool the cards' memory into a unified KV store. Prefix snapshots
retain their existing two-GiB budget.

## Ascend attribution

A separate native stage probe uses real 144-expert banks, clipped frozen kernels,
synthetic cold routes, and one decoded row padded to 17. It measures eager stage
events and synchronized graph replay, requiring bitwise-equal outputs. It runs
on physical chip 2 after serving performance tests complete. It creates no DMA
mappings and does not modify the serving process. Eager kernel spans include
submission gaps and are separate from production graph replay latency.

## Reproduction

Run `tools.sidecard.indexer_rotation_probe` in the CUDA virtual environment for
standalone numerical and timing qualification. Use a frozen runtime snapshot,
`prompt_prefix_cache: false`, `cuda_indexer_hadamard: true`, and the authenticated
16 MiB benchmark reference for `tools.sidecard.indexer_benchmark`. Its
`--complete-context` option performs actual prefill and generation to the model's
configured admission boundary; it does not disconnect after HTTP headers.

`tools.sidecard.native_stage_profile` requires `ASCEND_RT_VISIBLE_DEVICES=2,3`
and uses only logical device 0. Its source and native helpers must come from the
qualified snapshot, rather than unrelated local native-kernel experiments.
Private request and environment files are excluded from public receipts.

## Measured model results

| Prompt | Reference cold prefill | Fused cold prefill | Reference decode | Fused decode |
| --- | ---: | ---: | ---: | ---: |
| 2,048 tokens | 10.128 s | 9.932 s | 11.024 tok/s | 11.667 tok/s |
| 8,192 tokens | 43.130 s | 40.912 s | 11.685 tok/s | 12.555 tok/s |
| 23,820 tokens | 123.258 s | 119.370 s | 12.464 tok/s | 12.609 tok/s |

All six paths match every authenticated output-vocabulary hash and use zero
cached prompt tokens. Decode gains are 5.8%, 7.4%, and 1.2%. The full Code
prefill improves 3.2%, to about 199.5 prompt tokens/s. These are one paired run
per fixture, not medians or API measurements. Ascend remains the largest
latency dependency; the transform microbenchmark gain does not remove it.

See [matched benchmark](probe/indexer-benchmark.json) and
[independent prefill gate](probe/prefill-parity.json).

## Completed 128K context

The actual 131,008-token cold prompt took 670.793 seconds to prefill and then
completed 64 outputs at 11.745 model-forward tokens/s. All output vectors were
finite and the teacher-forced replay was bitwise identical. Its 1,716,176,896-byte
prompt snapshot retained about 4.24 GiB free on the RTX. The larger admission
is 131,072 total tokens. This checks real prefill and decoding, rather than
admission alone. The synthetic repeated prompt is a capacity/stability fixture;
long-context task accuracy is not evaluated.

This experiment does not establish a 1M-token service, a maximum hardware
capacity, or cold first-text latency for arbitrary 128K agent requests. Full
context cold prefill still takes minutes; cached follow-up requests are a
separate behavior. The other Ascend card remains available for unrelated work.

## Serving and client configuration

The qualified service runs at `http://192.168.53.187:8000/v1`, model
`glm-sidecard`, with 131,072 total-token admission and fused CUDA rotation.
Three matched cold 2K API requests retain identical prompt IDs, output text,
and 64 completed outputs. Median decode rises from 11.426 to 11.808 tokens/s
(3.3%). Median time to first text falls from 10.130 to 9.922 seconds. This is a
small three-fixture comparison; individual workload gains differ.

[API summary](api-summary.json), [baseline requests](baseline-api.json), and
[candidate requests](candidate-api.json) record the serving comparison.

The `vllm-sidecard` entry in `~/.config/kilo/kilo.jsonc` now advertises
128K context, 122,880 input tokens, and 8,192 output tokens. Its output settings,
other providers, and default model are preserved. A local backup retains the
previous configuration; credentials and the entire client configuration are
excluded from this artifact. [Public client limits](kilo-limits.json) record only
the changed sidecard entry.

## Native stage result and next candidate

For four synthetic cold routes, the graph replay takes about 2.16 ms across
layers 18, 24, and 31. Eager event medians attribute about 0.98 ms to
gate/up plus SwiGLU/quantization and 0.54 ms to down projection. Stable
reduction contributes only 0.04–0.05 ms. Projection work dominates this fixture;
removing the final reduction alone would save little. Stage event timings and
graph replay have different instrumentation and must not be added together.

For exactly one nonzero cold route, the existing unpadded one-row graph matches
the padded graph bitwise across all 21 changed-input comparisons. Median replay
falls from roughly 1.01 ms to 0.78 ms (about 22%). This is a narrow candidate,
not a production switch. Multiple contributing routes must retain the stable
reducer because atomic addition order can vary. The candidate needs both a
validated per-rank route-count contract and full-model qualification.

[Native stage profile](native-stage-profile.json) records all nine cases.
The first launcher omitted the existing CANN Python-path suffix and failed to
import the CANN compiler Python package before loading banks. The corrected launcher preserves that suffix;
no driver or kernel-module change was needed. Both attempts' logs are retained.
The native profiling source matches the delivered file byte for byte.

All 1,299 standalone sidecard unit tests pass. Changed-file manual checks pass;
full repository checks retain the seven known legacy failures. The serving
startup gates pass, and both experiment workers closed normally with exact
poisoned guards before the 128K service started. The bridge retains its six
expected live serving references.
