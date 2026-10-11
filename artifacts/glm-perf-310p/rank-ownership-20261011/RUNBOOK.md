# Rank-owned GLM indexer candidate gate

## Offline checks

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. \
  pytest --confcutdir=tests/ut/glm_perf -q \
  tests/ut/glm_perf/test_indexer_owner.py \
  tests/ut/glm_perf/test_native_loader.py \
  tests/ut/glm_perf/test_channel_scale_loader.py \
  tests/ut/glm_perf/test_indexer_bundle.py
```

Real CPU process protocol check, independent of any serving endpoint:

```bash
OMP_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. \
  python -m torch.distributed.run --standalone --nproc-per-node=6 \
  -m tools.glm_perf.probe_indexer_owner \
  --backend gloo --tp-size 2 --owner-rank 1 --tokens 1 4 37 640 \
  --output /tmp/glm-owned-indexer-cpu-gate
```

Use four processes, `--tp-size 4 --owner-rank 3`, for the two-card group protocol.
These use synthetic state/indices, not GLM model weights.
Repeat with `--overlap-prefill` to test asynchronous posting and deferred
completion for rows 37 and 640. Rows 1 and 4 keep the synchronous path.

## Explicit NPU transport and capture gate

Run only after exclusive reservation of all six chips, or four chips for TP4.
This command opens the requested devices. It does not load a model or establish
real-indexer parity:

```bash
pytest --confcutdir=tests/e2e/nightly/glm_channel_scale -q \
  tests/e2e/nightly/glm_channel_scale/test_indexer_owner_transport.py \
  --glm-indexer-owner-hccl --glm-indexer-owner-tp-size 2 \
  --glm-indexer-owner-rank 1
```

The gated test uses existing visibility to map process local ranks to chips.
It checks byte results and changed-input asymmetric capture at sizes 1 and 4,
plus eager rows 37 and 640. Failure is fatal; do not fall back silently to a
replicated indexer during an active session.
Repeat with `--glm-indexer-overlap-prefill` to gate the deferred large-batch
transfer. The synthetic independent work is not a real MLA overlap benchmark.

## Cold serving candidate

Keep the same qualified checkpoint, native kernels, quantization, TP2/PP3,
640-token scheduler chunk, context, request inputs, and selector mode as the
reference. Start on an independently reserved service allocation using the
normal launcher admission workflow. For `glm_native_int4`, add only:

```json
{
  "indexer_owner_rank": 0,
  "indexer_exchange_budget_bytes": 12582912
}
```

Pass this through `--model-loader-extra-config`. The example admits 12 MiB per
rank, sufficient for the calculated 10.625 MiB TP2 slab at 640 rows. TP4 needs
at least 21.25 MiB; reserve 24 MiB (`25165824`) at 640 rows. If the incoming
scheduler chunk is larger, run the offline geometry audit and increase this
explicit reservation first. The transfer buffer follows the configured batch
capacity, not the maximum context window.

The same two keys are optional additional fields of
`glm_channel_scale_candidate`'s existing loader config. First qualify ownership
on the existing native checkpoint to avoid confounding it with new quantization.

After qualifying the synchronous ownership candidate, repeat with the same
config plus `"indexer_overlap_prefill": true`. The optional mode overlaps local
Q/K preprocessing and KV insertion with the pending owner result for batches
larger than eight rows. It joins before attention consumes any indices and
before the next sparse layer can reuse transfer storage. It requires a supported
310P frontend for every indexer and at most eight decode sequences. Receive
memory and collective count are unchanged; decode sizes 1..8 remain synchronous.
Record whether HCCL's implementation actually permits compute overlap rather
than relying on the `async_op` argument alone.

Start with eager parity, then match the reference decode graph mode. For graphs,
the model must already have `capture_safe_selection` prepared through its normal
config/composition path. The candidate does not turn it on. Any reference change
needed to enable that path is a separate qualification step. MTP, sequence/context
parallelism, DP greater than one, and KV transfer are explicitly rejected.

The load report logs the local owner, TP rank, stage-local indexer count,
calculated transfer bytes, retained peer cache allocations, and unqualified
hardware status. Local owner zero maps to a different global rank in each PP
stage; do not force global rank zero to own later stages.

## Acceptance and rollback

1. Compare real indexer selections and owner cache continuity with the unchanged
   replicated implementation, including pool tails, dense/sparse boundary,
   prefix reuse, long contexts, request recycling, and concurrent requests.
2. Replay graphs with changed positions, request pages, and cache contents.
   Verify every rank participates in exactly one transfer per sparse layer and
   receives the current indices. Check process liveness and collective order.
3. Run matched coding/tool requests and deterministic output/logprob parity.
   Peer MLA caches and TP outputs must still agree with the reference.
4. Measure absolute projection, pool update, scoring, transfer, peer copy, and
   attention times. Report cold-prefill latency and c1/c4 decode separately,
   together with actual peak memory and graph replay evidence.
5. Retain only if serving measurements justify the collective. Reduced total
   operations or peer idle time alone is not a speed result.

Rollback uses a fresh load without these options. Do not restore replicated
forward methods in-place: peer indexer caches have not been updated. No peer
cache or weight memory is reclaimed in this candidate.

The next architecture candidate is head-sharded indexer projection/scoring
with an explicit score-reduction contract. It must preserve the per-head ReLU
before score summation and qualify FP32 reduction/tie behavior. It is not part
of the fixed-owner implementation.
