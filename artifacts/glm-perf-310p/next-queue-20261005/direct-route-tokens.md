# Eighth experiment: derive token IDs from the expert permutation

## Avoided work

The grouped expert dispatcher flattens routes in token-major order. It builds
`arange(num_tokens).expand(..., top_k).reshape(-1)`, then GLM gathers that array
by `dispatch.order` to identify each sorted route's input token.

Because `order` contains original route numbers, the same token IDs are
`order // top_k`. GLM uses integer floor division by eight.
The candidate removes the expanded array from the private dispatch descriptor
and replaces its gather with direct arithmetic. Merely changing the gather
would leave the original expansion allocated; both are removed here.

This is a metadata optimization, not a replacement for the larger weight-reuse
experiments. No throughput estimate is claimed. At 640 tokens/top-8, the
expanded INT64 token-ID array is only 40 KiB; repeated allocation and launches
are relevant in addition to its memory traffic.

## Scope and exactness

Candidate: `tools/glm_perf/resident_candidates/direct_route_tokens.py`.

- It makes a private function from the current dispatcher source. Stable expert
  sorting, peer sentinel handling, compare/histogram counts and inverse sorting
  are retained. The shared Qwen dispatcher is not modified or monkeypatched.
- It changes only GLM's grouped-method local builder binding and sorted-token
  assignment. Expert GEMMs, activation, output combination and reduction stay
  unchanged, including the original peer masking before dispatch.
- Source guards reject changed metadata expressions or additional token-ID
  consumers. Repeated preparation unwraps the original grouped method.
- INT32 is used only when the route count fits: permutation values are in
  `[0, num_routes)`. Larger permutations retain INT64. All top-k values
  use integer floor division. The original scalar shift passed eager parity but
  failed full graph capture with a synchronous scalar copy on this runtime.
- There is no device-to-host scalar read, dynamic compaction, model-weight
  allocation or native-library dependency.

**36 CPU tests pass**: current dispatcher parity at 0/1/8/640/1280 tokens,
noncontiguous IDs/weights, peer-only routes, compare/histogram modes, several
top-k sizes, direct input-gather equality, source guards and repeated preparation.
An operation check confirms the original token-ID arange/expansion is absent;
the comparison-count path still builds its necessary expert-ID arange.

## Pending hardware gate

1. Confirm INT32 cast, division and activation `index_select` are supported on the
   qualified 310P path and do not introduce slower fallback operations. CPU
   equality does not establish NPU dispatch quality.
2. Compare exact metadata and gathered activations with the baseline on actual
   local, peer-only and mixed routes. Include the active histogram threshold,
   graph rows 2/8 and larger prefill batches.
3. Compare isolated dispatch time with both variants warmed, then c1/c4 and cold
   prefill through the resident harness. Count changed operations in a short
   trace only if needed to explain a regression or an unexpectedly flat result.
4. Test independently from experiment six: both replace `_apply_device_grouped`.
   Loading one replaces the other's candidate; they are not automatically
   composable. A combined variant needs a separate explicit gate after wins.

This remains experimental. No NPU, server, shared dispatcher or model weights
were touched while preparing it.

Hardware progress and the graph-capture correction are recorded in
[the hardware study](hardware/README.md).
