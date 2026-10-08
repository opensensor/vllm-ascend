# Initial resident harness hardware check

TP4, MTP1, full decode graphs `[2, 8]`, context cap 32768, batch 640,
CPU affinity applied. Server process 3548534; workers 3551037, 3551636,
3552190, 3552712. This attempt is stopped.

All four execution modes switched without restarting workers or changing the
registered-parameter storage digest. The initial digest did not include packed
expert containers; the retry extends that coverage.

| Execution | Arithmetic response | Finish |
| --- | --- | --- |
| graph | Started with 45, then unrelated/repeated output | length (32 tokens) |
| direct-target | 45 | stop (4 tokens) |
| direct-draft | 45 followed by repeated thinking delimiters | length (32 tokens) |
| direct-both | 45 | stop (4 tokens) |

These are diagnostic observations, not a performance or quality qualification.
The shipped smoke gate checks transport and nonempty output, so these requests
satisfied that gate despite two malformed answers.

Applying the reference normalization candidate discarded the old captures.
Recapture failed on all ranks at `NPUGraph.capture_begin` with
`NPUCachingAllocator.cpp:2518: it->second->use_count > 0`. The wrappers still
held the retired pool handle. The scheduler remained paused. Later status RPCs
received queued capture errors: the executor raises on the first failed rank
without draining the remaining replies. The attempted recovery candidate was
never applied. A restart was necessary.

The retry renews the shared target/draft graph-pool handle, returns capture errors
as rank receipts, and fingerprints packed expert banks and views. Evidence:
`gate.jsonl`, `gate.log`, `serve.log`. The one-time `recover_pool.py` file is an
unapplied recovery attempt, not a required runtime component.
