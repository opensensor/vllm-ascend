# GLM sidecard single-contribution Ascend decode

The direct-DMA worker counts positive cold contributions for each chip while
validating its existing CPU routing metadata. A one-row request with exactly
one owned contribution selects a separately captured unpadded Ascend graph.
Zero or multiple owned contributions retain the qualified 17-row stable graph;
repeated expert IDs count as separate contributions. Prefill is unchanged.
This avoids a new device readback or synchronization to choose the graph.

`direct_dma.npu_single_route_graphs` is a boolean runtime option, defaulting to
false, and requires `npu_expert_graphs`. The extra graphs use the existing
qualified native binaries, weights, and ownership split. Capture temporarily
changes the actor's padding setting and restores it even if capture fails.
There is no native-kernel rebuild, driver change, or new environment variable.

The one-contribution graph uses the existing unpadded native path. It never
performs a multi-contribution atomic reduction. Every chip still publishes its
complete FP32 result before the completion sequence. CUDA consumes results
through the existing ordered PCIe DMA completion contract. Activations and
expert outputs do not pass through host staging; routing validation and kernel
submission still involve the host.

## Numerical and performance qualification

`tools.sidecard.single_route_probe` loads each resident layer in isolation,
without creating DMA mappings or stopping the service. It restricts visibility
to physical chips 2 and 3 and checks both ranks separately. Each layer checks
all eight route slots and four activation scales, with non-unit routing weights
and positive peer-owned contributions. All 960 comparisons match the qualified
padded output bytes. The same gate runs during candidate worker startup, before
DMA slots are created.

`tools.sidecard.single_route_benchmark` authenticates the preceding fused-CUDA
benchmark before allocating model state. It resets state between 2K, 8K, and
complete 23,820-token Code prompts, disables prefix snapshots, and requires
all 64 complete output-vocabulary hashes to match. Prompt IDs and all 23 tools
in the Code request remain intact. Timed decode measures 63 model forwards and
excludes CPU logit observation. Historical timings come from the preceding
qualified runtime; this is an across-runtime comparison, not a paired switch
inside one loaded model.

The matched serving comparison uses the same three cold 2K requests, zero
cached prompt tokens, and 64 completed outputs before and after deployment.
The repeatable scalar and text receipts contain synthetic fixture output;
private Code prompts and environment files are excluded. Three fixtures are a
small sample, rather than a broad workload study.

## Context and hardware allocation

The total context admission remains 131,072 tokens. The preceding fused-CUDA
runtime completed an actual 131,008-token cold prefill and 64 outputs with finite,
bitwise-repeated output vectors. This decode-only change does not alter cache
allocation or attention math; it does not repeat that long capacity experiment
or establish long-context task accuracy.

The RTX holds attention, KV state, shared experts, and hot experts. A single
Ascend Duo holds cold experts in layers 18–32 across physical chips 2 and 3.
The other Ascend card remains excluded. Logical 16 MiB DMA slots still use two
8 MiB transfer regions with the qualified UAPI 5 bridge and six persistent
serving references. Cold prefill retains 1,024-row windows and CUDA 32-row math.

## Checks and reproducibility

Run the probe in the qualified NPU virtual environment, preserving the existing
CANN Python path after adding the frozen source snapshot. Run the full-model
benchmark in the CUDA environment with `prompt_prefix_cache: false`,
`cuda_indexer_hadamard: true`, and both NPU graph options enabled. Supply the
preceding benchmark and its SHA-256 using `--reference` and
`--reference-sha256`; the Code request remains a private local fixture.

Unit tests exercise exact per-rank contribution counts, duplicate IDs,
selection fallbacks, invalid settings, capture failure cleanup, device scope,
reference authentication, and numerical-gate failure receipts. Repository-wide
checks run in an isolated worktree to keep unrelated local research intact.

## Result: numerically qualified, not deployed

| Prompt | Prior decode | Single-route decode | Candidate cold prefill |
| --- | ---: | ---: | ---: |
| 2,048 tokens | 11.667 tok/s | 11.634 tok/s | 10.093 s |
| 8,192 tokens | 12.555 tok/s | 11.312 tok/s | 41.113 s |
| 23,820 tokens | 12.609 tok/s | 12.777 tok/s | 120.031 s |

All 192 complete output vectors match the authenticated fused-CUDA reference.
The full-Code run improves 1.3%, but the 8K run regresses 9.9%; aggregate decode
fails the performance gate. The candidate was closed normally and the preceding
128K fused-CUDA service restored. The option remains disabled in serving.
These timings come from different runtime loads and do not establish the cause
of the regression. The next optimization uses alternating modes in one loaded
runtime to reduce this comparison's sensitivity to run-to-run drift.

The retained worker profile contains 5,602 decode calls; only 405 (7.2%) use
the single-contribution graph. Its observed native median is 0.810 ms. Ranks
have independent sequence counters, so these receipts cannot pair chip calls
or determine which chip was on each model dispatch's critical path. They do
show that the optimization has limited coverage.

The three API fixtures rerun after restoration complete successfully with
identical prompt IDs and text. Their median decode is 12.438 tok/s and median
first-text latency is 9.732 s. These are restoration checks, not candidate
serving measurements. The earlier baseline median was 11.861 tok/s, showing
that this small benchmark can drift materially between service loads.

[Complete-model gate](probe/single-route-benchmark.json),
[deployment rejection](probe/deployment-decision.json),
[route profile](route-profile-summary.json), and
[restored API requests](restored-api.json) preserve the evidence.
Both experiment workers report normal closure and exact poisoned guards.
