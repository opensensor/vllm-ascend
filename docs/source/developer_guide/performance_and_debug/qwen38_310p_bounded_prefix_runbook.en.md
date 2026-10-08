# Qwen 310P bounded prefix Mamba retention

This is the English companion to the
[Chinese runbook](qwen38_310p_bounded_prefix_runbook.md).
The operator authorized a new isolated NPU qualification run on October 8.
Its measured results and current service state belong in the
[validation report](../../../../artifacts/qwen38-prefix-npu-20261008/REPORT.en.md).

## Purpose and scope

Attention KV capacity and retained Mamba checkpoints have separate budgets.
The previous three-slot image server exhausted 63 primary slots and 175–194
archive slots per group while attention KV usage was near 10%. Its read-only
snapshot recorded 412 spills and zero restores across ranks and groups.
Those counters establish transferred payload, not transfer duration or the
cause of the thermal shutdowns.

Select `PrefixMambaBoundedScheduler` explicitly. It evicts old Mamba prefix
hashes before asking workers to retire the corresponding state. Hybrid prefix
lookup then uses a shorter complete prefix or computes the prompt again.
Request-owned states and pending copy-on-write sources remain protected.
Attention hashes retain their existing policy. Workers preserve primary,
archive and swap storage, the null slot, recurrent precision and graph-visible
addresses. Retirement drains pending writers once across groups and does not
copy retired checkpoints to CPU.

The synchronous, standalone Qwen4Exp 310P configuration supports align-mode
prefix caching. Three slots with MTP2 reserve four working windows per request
and retain at most 27 cached checkpoints per group; four slots retain 15.
Owned states are additional. The default scheduler remains unchanged.

## Deployment

Start from a complete qualified snapshot and preserve the CANN/OPP ordering in
the [runtime runbook](qwen38_310p_runtime_runbook.md). Backport the scheduler,
state-tier retirement helper and optional `_update_states` snapshot handling
as a matched set. Do not overwrite qualified model files with a different
version of the fork.

Add this argument to the image-enabled launcher:

```bash
--scheduler-cls vllm_ascend.core.prefix_mamba_scheduler.PrefixMambaBoundedScheduler
```

The initial queued profile used three requests, MTP2, graph sizes `[3,9]`, a
1,024-token scheduler batch, 262,144 maximum sequence length and one image per
prompt. After hardware validation and the operator request, the live profile
uses a 2,560-token scheduler batch and selected native HC residual; see the
validation report for image, cold/repeat-prefix and transfer results.
The planner's aggregate token capacity does not establish long-context quality
or sustained throughput. Retention can reduce historical prefix reuse and
increase prefill work. It does not promise zero spills under every workload.

## Qualification

Use the pinned OpenSensor vLLM `3ab5dda29` source for host regression tests:

```bash
PYTHONPATH=/path/to/pinned-vllm VLLM_TARGET_DEVICE=cpu VLLM_PLUGINS='' \
python -m pytest --noconftest -q \
  tests/ut/_310p/test_prefix_mamba_scheduler.py \
  tests/ut/_310p/test_prefix_mamba_state.py \
  tests/ut/qwen38_1m/test_prefix_npu_validation.py \
  tests/ut/qwen38_1m/test_resident_reset.py
```

All 78 checks passed before the new hardware run. They cover real BlockPool
canonical and partial hashes, copy-on-write protection, serialized snapshots,
group validation, slot reuse, failure cleanup and diagnostic transfer receipts.
These CPU results are not NPU throughput measurements.

The diagnostic worker extension operates only on an exclusively owned, drained
and cache-cleared test server. It writes known values into the loaded model's
actual checkpoint tensors, compares baseline and bounded retention, verifies
values and storage addresses, then resets metadata. Its elapsed time is a
checkpoint microbenchmark, separate from real-model image and concurrent
inference checks. Save every rank's acknowledgments and transfer deltas.
Keep the thermal watchdog active and pause new request batches for cooling.
