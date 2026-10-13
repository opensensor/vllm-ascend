# GLM sidecard captured CUDA indexer projections

Each sparse-attention indexer can replay its fixed one-row projections in one
CUDA graph: FP32 key projection and layer normalization, FP32 compression-gate
projection, query projection and original rounding/rotation, and routing-weight
projection/scaling. Hidden-state conversion is shared. The graph uses private
owner buffers and copies the current hidden state and query latent before replay.

Compressed-pool append, incomplete tails, causal sparse selection, cache lengths,
and attention remain outside capture. Multi-row prefill uses the existing eager
path, as do requests below the sparse-index threshold. No cache position is
captured. Reset and prefix restoration preserve existing cache tensor addresses.

`cuda_indexer_projection_graphs` is a boolean runtime option, defaulting to
false, and requires `cuda_indexer_hadamard`. Configuration is checked before
model allocation. Capture happens after loading the CUDA layers; close releases
its private graph buffers. Ascend experts, placement, quantization, transfer
geometry, native binaries, and completion ownership are unchanged. The preceding
single-route Ascend candidate remains disabled after its performance rejection.

## Method

The standalone probe uses real weights from layers 19 and 43, five input scales,
and four changed-input repetitions per layer. All four output tensors are
byte-identical in all 40 cases. Median host-inclusive projection time changes
from 252.871 to 59.564 microseconds on layer 19 and from 251.889 to 54.824 on
layer 43. These microbenchmarks measure one indexer projection, not whole-model
throughput.

The full-model benchmark authenticates the preceding fused-indexer reference,
processes unchanged 2K, 8K, and 23,820-token Code prompts once, and snapshots
complete recurrent, KV, pool, and tail state. It alternates eager/captured
64-output decode twice per prompt in the same loaded runtime, restoring the
identical prompt state each time. Every complete output vector must match the
historical qualified reference. The first run supplies the same teacher tokens
for subsequent runs. Timings cover 63 model forwards and exclude CPU logit
observation and snapshot restoration. The separately measured cold prefill
uses the eager projection path and zero cached prompt tokens.

This separates a decode optimization from prompt caching. Snapshot restoration
is used to compare the same decode states; it is not included as a speedup in
cold prefill. The Code prompt and all 23 tools remain intact. No compact client
profile or shortened input is used.

Matched serving tests then use three cold 2K requests with unchanged prompt IDs,
zero cached prompt tokens, and 64 completed outputs before and after deployment.
These are a small sample and do not establish broad workload accuracy.

## Hardware and context

One RTX PRO 6000 holds the CUDA backbone, attention caches, shared experts, and
hot experts. One Ascend Duo holds cold experts in layers 18–32, using physical
chips 2 and 3. The other Ascend card is excluded. Activations and outputs use the
qualified cross-vendor direct PCIe DMA completion contract; the host still
validates routing metadata and submits NPU kernels.

Cold prefill retains 1,024-row windows and CUDA 32-row math. Logical 16 MiB DMA
slots still use two 8 MiB export regions with the unchanged UAPI 5 bridge.
The total context admission remains 131,072 tokens, and the Kilo entry remains
128K with 8,192 reserved output tokens. The prior runtime's actual 131,008-token
prefill plus 64-output test remains the capacity qualification; this change does
not rerun that long test or establish long-context task accuracy.

## Reproduction

Run `tools.sidecard.indexer_projection_probe` in the qualified CUDA virtual
environment against a frozen runtime configuration. Run
`tools.sidecard.indexer_projection_benchmark` with captured projections enabled,
unchanged Ascend decode graphs, the private complete Code request, and the
preceding fused-CUDA benchmark supplied with its SHA-256. Neither the private
request nor credentials/environment files are included in public receipts.

The CPU tests cover complete causal selections and pool/tail bytes across every
incomplete-tail phase, multi-row fallback, invalid settings, shape/dtype checks,
mode restoration after exceptions, and graph close. Hardware gates check real
projection tensors and complete model vocabulary vectors.

## Qualified decode result

| Prompt | Eager projection decode | Captured projection decode | Gain |
| --- | ---: | ---: | ---: |
| 2,048 tokens | 12.179 tok/s | 12.525 tok/s | 2.85% |
| 8,192 tokens | 12.466 tok/s | 12.638 tok/s | 1.38% |
| 23,820 tokens | 13.151 tok/s | 13.400 tok/s | 1.89% |

Each row combines the two measured 63-forward repetitions per mode; rates use
total forwards divided by total time. Both individual pairs improve for every
prompt size. All 768 complete vocabulary comparisons match bitwise. The
aggregate/per-fixture performance gate passes, so the projection graph is
enabled in the deployed candidate. The option still defaults to false for
unqualified configurations. Cold prefill is unchanged by this decode-specific
optimization; its separately recorded measurements are not paired speedups.

[Paired model benchmark](probe/projection-benchmark.json),
[model summary](model-summary.json),
[component gate](projection-probe.json), and
[deployment decision](probe/deployment-decision.json) preserve the evidence.
The executed source archive, runtime manifest, and delivered sidecard sources
match byte for byte. Both experiment workers closed normally with exact
poisoned guards before serving restarted.

## Serving result and final checks

The service runs at `http://192.168.53.187:8000/v1`, model `glm-sidecard`,
with captured indexer projections enabled and 131,072 total-token admission.
All six startup parity gates pass, and serving retains six expected bridge
references. [Serving receipt](sidecard-projection-serving.json) records public
settings and GPU memory allocation without environment files or credentials.

The three cold 2K API fixtures retain identical prompt IDs and output text,
with 64 completed outputs each. Median decode changes from
12.438 to 12.639 tokens/s (1.6%). Median first-text
latency is 9.732 versus 9.810 seconds. This decode optimization does not
claim a cold-prefill gain. [API summary](api-summary.json),
[baseline requests](baseline-api.json), and [candidate requests](candidate-api.json)
provide the measured client comparison.

All 1,336 sidecard unit tests pass. Changed-file manual checks pass. Full
repository checks retain the seven known legacy failures: Ruff check/format,
Codespell, Typos, Clang-format, Markdownlint, and forbidden imports. Those checks
ran in an isolated worktree; all 13 unrelated tracked research files retain
their original hashes. Private prompts, credentials, and model payloads remain
excluded from the public artifacts. The other Ascend card remains excluded.

[Design PNG](design.png) shows the deployed architecture and measured results.
The drawing source is retained alongside the exported figure.
