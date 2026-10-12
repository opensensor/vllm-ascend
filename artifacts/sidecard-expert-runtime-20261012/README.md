# GLM expert-only Ascend sidecard: hardware implementation

The complete GLM-5.3-Flash backbone now runs on the RTX PRO 6000 Blackwell.
One Atlas 300I Duo supplies independent resident cold experts on its two
assigned chips, physical **2 and 3**. The other Ascend card remains available
for other work. The live endpoint is `http://192.168.53.187:8000/v1`, model
`glm-sidecard`; requests use real checkpoint weights.

The measured V2 snapshot was restored after profiling as
`/srv/ai/storage/artifacts/sidecard-expert-runtime-20261012-v3`, using the
exact same serving Python source. The
[complete latency profile](../sidecard-expert-profile-20261012/README.md)
records the delay breakdown and a fresh 9.64-token/s serving confirmation.
Read the [runtime runbook](../../tools/sidecard/EXPERT_RUNTIME.md) and
[implemented design PNG](architecture.png).

## Completed performance

The same 2,048-token prompt and 64-token greedy completion fixture used by
the previous layer runtime gives:

| Runtime | Decode tokens/s | First text, seconds |
| --- | ---: | ---: |
| Earlier prepared complete-layer split | 7.8325 | 29.9477 |
| Expert-only, middle backbone eager (V1) | 8.6205 | 27.1977 |
| Expert-only, middle KDA backbone graph replay (V2) | 9.5795 | 27.3462 |

V2 improves decode by **22.3%** against that earlier measured split and
reduces time to first text by **8.7%**. These are completed single-request
measurements, not throughput projections or averaged population results.
Different short fixtures reach approximately 12 tokens/s; that figure is
not interchangeable with the fixed 2K benchmark.
The historical two-Ascend configuration reached approximately 9.5–9.7
tokens/s with different fixtures and MTP enabled. V2 is near that range,
without establishing an exact comparison or a large speedup.

The final graph-enabled version passes five elementary arithmetic prompts,
three repeated 64-token completions, and malformed/over-budget admission
checks. [API validation](validation.json) completes 4K and 16K capacity
tests, with first text at 45.87 and 170.14 seconds respectively.
The [full capacity gate](context-gate.json) completes **65,504 prompt tokens
plus 32 output tokens**, filling the 65,536-token configured limit. First
text takes 726.56 seconds, followed by 9.58 decode tokens/s. Capacity is
not a long-context quality claim.
[Operational validation](operational-gate.json) passes concurrent queued
request isolation, streaming with usage, cancelled-stream recovery, one
synthetic retrieval needle, and generation after 315 seconds idle. The
single needle is a smoke test, not a general retrieval benchmark.

## Implemented ownership and overlap

The main weight payload is 131.92 GiB, larger than RTX VRAM. CUDA receives
84.98 GiB of weight payload: all backbone weights, all experts in layers
3–17 and 33–44, and 22 static hot experts per layer 18–32. The static set
uses only prefill observations from three earlier prompt families; no
held-out decode or repeated 2K benchmark routing informed its selection.
See [placement.json](placement.json) and the exact
[selection provenance](placement-provenance.json). Training comprises only
48 prefill tokens. This is an initial placement, not a claim that hotness
generalizes to all workloads or that the memory allocation is optimal.

CUDA owns attention, recurrent/conv state, KV and indexer caches, mHC,
routing, shared MLPs, embedding, normalization, head, and greedy sampling.
Each Ascend chip keeps its original 144-expert bank per middle layer,
including dormant copies of CUDA hot experts. Nonzero work goes to exactly
one owner. No HCCL, TP2 attention, shared MLP collectives, or weight paging
remain in the expert-only path.

The private socket payload contains 8,256 input bytes per row: FP16 x,
INT32 global IDs, and FP32 cold weights. Each active chip returns a single
16,384-byte weighted FP32 partial per row. Both chips receive work before
CUDA hot experts and shared MLPs launch, and before either response is
awaited. An inactive chip receives no call. GPU/rank-0/rank-1 partials merge
in fixed order with the original normalized route weights.
Every reply checks its sequence, layer, and row count.
When both chips have work at all 15 middle layers, host traffic is 739,200
bytes per generated token. Smaller payloads per call do not eliminate the
latency of up to 30 chip calls per token.
[Offline held-out observations](placement-heldout-observations.json) apply
this placement to earlier TP2 decode traces: only 4.0% of routes in the
matched 2K fixture are hot, with 29.70 chip requests per token on average.
The three other prompt families have 29–33% hot routes, but still wait at
nearly every middle layer. These are earlier observations, not measured
routing from V2. They explain why this placement does not remove the cold
latency floor and why a representative calibration set matters.

KDA spans outside the middle block replay on CUDA. Middle KDA attention,
mHC, and router preparation also replay on CUDA, with native cold calls
outside capture. DSA and indexer length updates remain eager. Prefill uses
32-row chunks; overlap is inside each expert call, while layer dependencies
remain serial. This implementation does not use the former two-slot stage
pipeline. API requests serialize and reset caches.

The [memory receipt](memory-receipt.json) records 8,524,136,448 free GPU bytes
after loading, graph capture, and clearing unused allocator storage.
Raw CUDA expert allocation is 73,417,802,752 bytes; Torch allocation is
19,098,409,472 bytes. They are distinct allocations; payload size does not
include all execution storage. Context admission still requires completion
of the actual capacity gate, rather than inference from free memory alone.

## Numerical and ownership validation

- **602 portable tests pass**, including static placement authentication,
  arbitrary global-to-local expert mapping, mixed and inactive workers,
  fragmented private messages, response mismatch rejection, overlapping
  dispatch order, graph capture boundaries, and frozen placement isolation.
- [Final dispatch gate](dispatch-gate-v2.json): **64 hardware cases pass**
  for KDA layer 18 and DSA layer 19, rows 1/4/17/32, hot-only, cold-only,
  mixed, both chips, and the ends of arbitrary expert banks. CPU references
  unpack authenticated checkpoint experts and exercise SwiGLU clipping.
  Maximum absolute error is 0.0000076294; repeated output is bitwise equal.
- [Backbone gate](backbone-gate.json): **48 hardware cases pass** against
  CPU references for full-head CUDA KDA/DSA, shared MLP, mHC pre/post,
  FP32 router logits, and exact top-8 IDs. BF16-range residual cases are
  included. These are component gates, not a broad task accuracy benchmark.
- Full-model startup requires three identical 64-step teacher-forced
  full-vocabulary traces, local native repeat gates, and real-weight chat.
  [Graph parity](graph-parity.json) additionally proves all output byte
  hashes match between V1 eager middle KDA and V2 graph replay over the same
  64 teacher-forced steps in all three repeated trials. It does not claim
  bitwise agreement with the
  former TP2-attention architecture.

The clipped native kernels, A4 activation quantizer, checkpoint identity,
TF32-disabled CUDA policy, fixed-slot Ascend reduction, full FP32 final
normalization, and FP32 residual range remain unchanged. The checkpoint
manifest SHA-256 is
`a40e4312e4cd4162e7947aeb28c6fb80484a7500dc617f226eada1ed274d6a9f`.
Placement SHA-256 is
`aaa70408a6046e4e7fafaf62173d95476617ae9d3a9ed3e01c31f760872caad1`.

## Evidence and limits

Executed Python file hashes are in [source-sha256.json](source-sha256.json);
dispatch snapshot hashes are in
[component-source-sha256.json](component-source-sha256.json), and the earlier
backbone gate snapshot is in
[backbone-source-sha256.json](backbone-source-sha256.json).
The [runtime specification](runtime-specification.json) retains unused
legacy `init_method` and `npu_decode_graphs` fields. This architecture never
initializes HCCL; `CudaExperts` also forces `tp_rank_inputs=false` internally.
Runtime snapshots are immutable, exclude private environment contents,
and retain their own placement. V1 was preserved as a scoped automatic
fallback during V2 startup; no fallback was needed.

The service supports greedy text chat/completions, streaming, and
tokenization. It has no batching, prefix cache, MTP, vision, tool execution,
or nongreedy sampling. Capacity tests do not establish long-context quality;
arithmetic smoke is not a general accuracy benchmark.

Repository-wide `bash format.sh ci` ran in an isolated checkout. Existing
failures in ruff, spelling, clang formatting, Markdown, and forbidden imports
remain outside this change. Changed files pass the scoped manual hooks;
no formatter edits to unrelated files are included.
