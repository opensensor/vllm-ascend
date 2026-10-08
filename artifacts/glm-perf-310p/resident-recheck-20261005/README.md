# Fresh GLM resident harness check

Date: 2026-10-05. Tested the already running real-weight TP4 GLM server on
Threadripper, port 8001. No model or worker restart was performed.

## Passed live checks

- Switched `graph`, `direct-target`, `direct-draft`, and `direct-both` with the
  existing `completed_pools` candidate and identical prompts/seeds.
- Applied `mtp_norm_reference`, explicitly recaptured target/draft graphs,
  and sent an inference request.
- Restored the exact original `completed_pools` source, recaptured again,
  and sent one single request plus four concurrent requests.
- **10/10 requests returned exactly `45` and stopped normally.**
- Installed the validated client module into the runtime mirror, then switched
  through its ordinary import path and verified a fresh multiplication request
  returned exactly `56`. This additional request also preserved all identities.
- All four worker PIDs and weight-storage fingerprints matched before/after:
  `1894308`, `1894739`, `1895109`, `1895534`.
- Final state: original candidate digest
  `dbfcc5534d4e205ccc0c7c86ab3d040dc8ea81a7c0ecae8432149ec9cc529445`,
  `mode=graph`, `graphs_dirty=false`, scheduler resumed.

The configured 311040-token context limit was retained. These bounded arithmetic
checks qualify control, patching, recapture, restoration, and concurrent request
completion; they do not qualify long-context accuracy or general model quality.
Switch transaction durations are recorded in `summary.json` and include pause,
physical cache resets, acknowledgments, and resume. They are not serving speedups.

## Failure exposed and recovery

The first run passed its first three modes, then an apply RPC raised
`generation has not been prepared`. Workers had already reached the requested
generation, but the exception left rank reply queues out of alignment. Later
calls received old preparation/status replies or null replies. The scheduler
stayed paused, and all workers/weight storage remained intact.

Recovery repeated only side-effect-free preparation with one fixed generation,
sent apply once, and read status until every rank confirmed graph mode. The
original candidate resumed without reloading weights. Evidence is in
`failed-receipts.jsonl`, `recovery-receipts.json`, and `recovered-status.json`.

The client now confirms generation-specific acknowledgments. It retries only
preparation and uses status reads after a mutation; it never repeats the mutation
to compensate for an old reply. The successful rerun used the updated client
from an isolated temporary file, whose hash is in `client-sha256.txt`.
The same module was then installed atomically into the existing runtime mirror;
`deployed-client-smoke.json` records the final switch and multiplication request.

Server-side application is now idempotent for a repeated generation, rejects
changed controls reusing an applied generation, and returns preparation/application
errors as rank receipts. These additional Python error paths passed CPU
regressions; the existing running workers retained their loaded implementations.
**45 focused CPU tests passed**, including stale replies and repeated apply.

## Evidence and reproduction

- `summary.json`: live gate result and switch durations.
- `receipts.jsonl`: complete control responses and scored inference outputs.
- `initial-status.json`, `final-status.json`: all rank/PID/storage identities.
- `original-candidate.py`: exact source restored to the server.
- `run_live_gate.py`: bounded gate that preserves the source active on entry.

Run the driver from the staged runtime, with the candidate snapshot at the
recorded path, and optionally supply an isolated updated client module:

```bash
PYTHONPATH=/srv/ai/src/glm-selective-w3-nz-test-20261004 \
  /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
  /path/to/run_live_gate.py /path/to/results /path/to/resident_harness.py
```

Use one administrative experiment controller at a time.
