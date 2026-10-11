# Qwen local INT8 math hardware receipts

Current live candidate: `qwen_qsa_owner_local_int8_math`, on six Ascend 310P3
chips. The 1,048,576-token limit is retained; full 1M accuracy is not qualified.

- `qwen-qsa-owner-prefill-20261011`: owner transport, shadow comparisons,
  normal cutover controls, and worker storage/cache receipts.
- `qwen-owned-activation-20261011`: isolated exact activation and graph probes.
- `qwen-owned-route-control-20261011`: exact stable dispatch, initial reducer
  mismatches, and the matching FP16 accumulation variant's results.
- `qwen-local-math-20261011`: deployed immutable asset manifest, real-weight
  partial/replay timings, full-model text/image gates, and cutover receipts.
- `qwen-local-finalize-20261011`: matching reducer real-weight qualification,
  held outside serving because its critical-rank timing was neutral/slower.

Compiled binary paths and SHA-256 digests are in the asset manifests; binaries
remain on the NPU host. `.py.txt` and `.cpp.txt` preserve exact frozen runtime
and compiled source bytes, including earlier attempts, without formatting
changes to their provenance. Maintained implementations are in
`tools/qwen4exp/`; source snapshots are evidence, not importable modules.

Only the full-model coding gate measures end-to-end decode. Dispatch,
activation, and routed MoE timings are stage measurements. Cold prefill had
zero cached tokens. Fresh/cached vision probes used the existing one-image
limit. The 50% throughput target over the older baseline is not achieved.

Model graph cutovers reset reusable prefixes; isolated probes preserve them.
No model reload, weight conversion, or device restart occurred in this phase.

The focused CPU suite passed 69 tests. Tests cover operand admission, stable
graph descriptor lifetime, owner rotation with a four-layer sparse interval,
stream/group transport, and collective qualification failure before capture.
