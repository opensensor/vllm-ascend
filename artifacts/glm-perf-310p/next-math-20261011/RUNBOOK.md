# Offline candidates and hardware qualification

These are default-off cold-load experiments. Do not hot-swap ownership, head
partition, or a token/bulk metadata ABI into an active model. Stage the committed
loader/helpers into a separate installed checkout before loading a candidate.
Keep the two-card W2 and three-card W3 checkpoints and qualified launcher intact.

## Repeat offline gates

From the repository root, with NPU backend autoload disabled:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. \
pytest --confcutdir=tests/ut/glm_perf -q \
  tests/ut/glm_perf/test_router_owner.py \
  tests/ut/glm_perf/test_indexer_heads.py \
  tests/ut/glm_perf/test_channel_weight_pipeline.py \
  tests/ut/glm_perf/test_channel_token_prefill_cpp.py \
  tests/ut/glm_perf/test_channel_prefill_protocol_cpp.py \
  tests/ut/glm_perf/test_channel_scale_runtime.py

TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. \
pytest --confcutdir=tests/e2e/nightly/glm_channel_scale -q \
  tests/e2e/nightly/glm_channel_scale/test_next_math_transport.py \
  --glm-next-math-cpu-transport

TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. \
python -m tools.glm_perf.next_math_audit --output /tmp/glm-next-math-ledger.json
```

The C++ tests execute production control loops with scalar/vector or Cube stubs.
They do not execute Ascend instructions, prove device scheduling, or reproduce
FP16 hardware numerics. Gloo tests are synthetic CPU process integration.

## Independent cold-load options

Merge one optional block into the existing loader-extra configuration. Examples
assume 640 batched tokens. They are not complete launcher configurations.
Both `glm_native_int4` and `glm_channel_scale_candidate` accept these options.

TP2 router experiment, with shared-expert overlap only above eight rows:

```json
{
  "router_owner_rank": 0,
  "router_exchange_budget_bytes": 262144,
  "router_overlap_prefill": true
}
```

For TP4, use at least 307200 bytes; 393216 leaves a small margin. This reservation
covers packet storage, not HCCL or graph resources. The owner is local to each
TP group, so TP2/PP3 has three independent owners. All ranks still join the
collective and compute their local expert/shared partials.

Alternative prefill head experiment:

```json
{
  "indexer_head_sharding": true,
  "indexer_head_budget_bytes": 201326592,
  "indexer_score_elements": 33554432
}
```

This example reserves 192 MiB separately for cloned local projection weights and
score scratch at 8192 context and 640 query rows. Full original indexer weights
and caches remain allocated. Do not combine it with `indexer_owner_rank`.
The planner rejects insufficient complete-column score capacity before weight
I/O. Decode remains original; smaller score tiles increase prefill collective
count. Partitioned FP32 sums need near-tie selection/quality validation.

Kernel experiments use the full original-FP8 channel-W4 overlay and
`glm_channel_scale_candidate`; these kernels cannot interpret the W2/W3 weight
formats directly. Replace only the existing `channel_bundle` value with the
versioned bundle root. Keep overlay, G256/A8 mode, token budget, context, PP/TP,
graphs, and quantization fixed across the v14/v17 comparison. All extra dense,
KV, graph, HCCL and PP reservations remain the launcher's responsibility.

## Hardware gates when the devices are reserved

The existing real-bank operator gate accepts the new bundles:

```bash
pytest -q tests/e2e/nightly/glm_channel_scale/test_moe_operator.py \
  --glm-channel-overlay /srv/ai/storage/models/GLM-channel-overlay \
  --glm-channel-bundle /srv/ai/storage/artifacts/glm-offline-next-math-20261011/v17-bulk-cache-stream \
  --glm-channel-device 0
```

Replace the example overlay with its verified staged path. This command uses an
NPU and is deferred until the devices are reserved. It is an operator gate, not
a complete serving benchmark.

1. Run the existing real-bank gate with each bundle, including prefill beyond
   16 rows and changed inputs. Compare streamed products and FP16 boundaries
   against that bundle's unmodified numerical reference. Check compiler resource
   admission and peak device memory.
2. Qualify TP router AllGather with asymmetric owner arithmetic and changed-input
   decode graphs; test every PP stage. Byte parity must include INT64 IDs that
   cannot survive an FP32 conversion. Then compare real router IDs/weights and
   local combined outputs with ownership disabled.
3. Qualify the real indexer head path at contexts below/above pool-selection
   thresholds, partial pools, recycled pages, prefix hits, multiple requests, and
   long continuations. Compare scores and selected indices, including near ties.
   Decode must dispatch the original native path. If selection changes materially,
   keep this candidate disabled even if CPU score tolerances pass.
4. Measure cold-prefill prompts of multiple lengths and cached continuations.
   Record operator/bundle hashes, absolute scalar/vector/Cube/memory time,
   time to first token, chunk throughput and steady c1/c4 decode. Report token
   counts and cache reuse for each request; changing prompt/chunk sizes invalidates
   a matched comparison.
5. Measure v14 and v17 first. Add router ownership independently, then head
   sharding independently. Combine only winners. Token v18/v19 saves workspace
   but has an unfavorable weight-read count; require a capacity reason and a
   separate wall-time test before using it.

On any failure, stop the trial and cold-load the retained qualified image.
Synthetic transport, successful compilation, and reduced work/bytes are not
serving qualification. This delivery did not use NPU hardware or modify a live
service.
