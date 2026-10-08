# Resident GLM control qualified on TP4

Date: 2026-10-05. The user reauthorized all four NPUs. Work stayed on the
shared main checkout; only the required harness files were staged to the
existing runtime mirror.

## Results

- **30 CPU regressions passed**, with Ruff checks and formatting clean.
- **Both hardware gates passed** by executing the shipped end-to-end test
  functions against the real-weight server, with every control response and
  inference result recorded. The first gate switches four execution modes,
  applies the normalization reference, recaptures, restores baseline Python,
  and recaptures again. The second injects a Python capture failure, verifies
  the scheduler stays paused and all worker replies remain readable, restores
  baseline, recaptures, resumes, and sends an inference request.
- All four worker PIDs and storage fingerprints stayed unchanged throughout.
  Fingerprints now include registered parameters, grouped packed expert banks,
  and per-expert views, including shape, dtype, stride, offset and device.
  They check allocation/layout identity, not weight contents.
- Reference-patch recapture: **4.336 seconds**; baseline restoration recapture:
  **4.361 seconds**. First gate: **52.33 seconds** including requests and resets.
  The follow-up mode switches, including pause/reset/resume, took
  **1.51–1.57 seconds**. These are control-operation timings, not serving speedups.
- **MTP repetition remains unresolved.** All 12 follow-up scored cases failed:
  sum, subtraction and short retrieval in each of `graph`, `direct-target`,
  `direct-draft`, and `direct-both`. Each case started with an independent
  pause/cache reset; all requests completed at the transport level. Early
  arithmetic successes in direct modes did not generalize. This does not
  isolate the problem to graph replay and is not a quality promotion.

## Fixes exposed by the first hardware attempt

The initial attempt is recorded in `../resident-12/`.

1. Renew the shared target/draft graph-pool handle when clearing captures.
   Reusing the retired handle failed in torch-npu's allocator at capture begin.
2. Return capture failures as per-rank error receipts, then raise in the client
   after consuming all acknowledgments. An exception escaping the worker RPC
   had left queued rank errors, preventing recovery through the control API.
   Other native/transport failures may still require a restart.
3. Include unregistered packed expert storage in the allocation fingerprint.

No model weights or quantization were changed by these fixes. The failed first
attempt required one restart; the successful second attempt kept weights
resident through both gates, both restorations, and the 12-request comparison.

## Server left running

- Address: **127.0.0.1:8001 on threadripper**, model `glm53-flash-selective-w3`.
  Loopback binding protects the experimental Python-execution control API;
  use an SSH tunnel for remote access.
- API PID 3580788; workers 3583059, 3583475, 3583955, 3584905.
- Baseline Python, full decode graphs `[2, 8]`, MTP1, TP4, synchronous scheduling,
  batch 640, four sequences, prefix caching on, aligned Mamba state.
- **32768 is a diagnostic context cap**, not a measured capacity limit.
- Worker affinity uses disjoint four-core groups with their SMT siblings.
- Final receipts: all ranks `mode=graph`, `candidate=baseline`,
  `graphs_dirty=false`; scheduler `is_paused=false`.
- This remains a diagnostic server with known repetition. Bounded metadata
  audits also add synchronization; these results are not throughput benchmarks.

## Evidence

- `gate.jsonl`, `gate.log`: mode switching, reference patch and restoration.
- `failure-gate.jsonl`, `failure-gate.log`: injected failure and resident recovery.
- `repetition.jsonl`, `repetition.log`: scored mode comparison and worker receipts.
- `final-status.json`, `affinity.json`, `source-hashes.json`, `cpu-tests.log`.
- `serve.log`: startup, graph captures, requests and failure diagnostics.
- `check_repetition.py`, `run_failure_gate.py`: reproducible diagnostic drivers.
  The first gate driver is `../resident-12/run_gate.py`.

The hardware gates exercise short single-request inference after captures.
They do not qualify long-context behavior, concurrency performance, or MTP
answer quality.
