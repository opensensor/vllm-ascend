# Qwen sustained thermal shutdown and deferred request hold

The previous short functional checks passed, but sustained interactive use
failed thermal qualification. The existing watchdog sent SIGTERM to API PID
3820668 at **23:37:55 UTC on October 8**, after a 96°C reading. All four NPUs
were idle afterward and cooling. The API process remained temporarily alive
waiting for HTTP connections after its engine and workers stopped.
The operator explicitly deferred starting another server. No server or
thermal controller was started in this follow-up.

## Observed evidence

| First observed hottest-device threshold | UTC time |
| --- | --- |
| 80°C | 23:21:06 |
| 84°C | 23:25:05 |
| 88°C | 23:28:23 |
| 90°C | 23:30:20 |
| 94°C | 23:34:13 |
| 96°C | 23:37:43 |

The watchdog sampled independently and acted at 23:37:55. Temperature records
are roughly 5.8 seconds apart. The full log contains zero Mamba device-tier
spill warnings, zero CPU-to-NPU checkpoint restore warnings and zero decode
graph fallback warnings. There is no final worker-counter snapshot after the
shutdown, so warning absence is not a measured transfer counter.

When another long prefill joined decoding, recorded generation throughput
repeatedly fell to 0.1–0.6 tok/s while attention KV usage increased. For example,
a request computed 19,486 new prompt tokens in 58.971 s while the other request
remained active. This is consistent with serialized model steps delaying
streamed decode, as in the earlier controlled test. Temperature and request
logs cannot identify which transfers or operators dominate heat. Host/NPU
copies, device-local memory traffic and compute still need separate profiling.
The bounded-checkpoint fix does not establish a sustained thermal solution.

## Implemented fallback, hardware gate pending

The requested opt-in controller reads all four 310P chip temperatures exposed
by `npu-smi info`. At any reading **>=94°C**, it uses
`POST /pause?mode=keep&clear_cache=false`. The pinned vLLM implementation sets
`PAUSED_ALL`, freezes active requests and queues arrivals. It does not abort
requests, clear prefix/KV/Mamba caches or offload model weights. Work already
in flight must reach the engine pause boundary before the hold is effective.

The controller remains latched until every valid reading is **<=85°C**, then
resumes a pause it requested. Readings in the 86–93°C band do not release the
hold. Missing, partial or invalid temperature readings request a hold and
cannot authorize a resume. An already paused server is treated as externally
paused and is never automatically resumed by this controller. Lost HTTP
acknowledgments are reconciled through the engine pause state; a failed resume
does not clear the thermal latch early. Loopback controls bypass HTTP proxies.

The controller checks the API PID's start time, exits on PID reuse or death,
and does not resume on exit. It never launches or restarts a server. The
existing **96°C emergency watchdog remains** as a separate last resort.
The pause API does not expose ownership tokens or its pause mode: coordinate
manual pause/resume and resident reconfiguration with this controller while it
owns a hold. Stopping it while held deliberately leaves the server paused.
Client timeouts may still expire during a cooling hold; application retries
must account for that. Hardware pause latency, cooling duration, streaming
continuity, image processing and state preservation remain pending NPU tests.

## Offline validation and deferred launcher

All **24 offline tests passed**, covering exact 94/85 boundaries, every-device
cooling, hysteresis, missing readings, manual pauses, pause/resume failures,
lost acknowledgments, incomplete HTTP response retries, PID reuse, and a real loopback HTTP test of the exact
keep/no-cache-clear query. These are controller logic and HTTP checks; the
fake engine tests do not validate physical cooling or NPU state preservation.
Ruff, shell syntax and scoped formatting checks are recorded with this report.
All applicable scoped code/documentation hooks passed. The required
`bash format.sh ci` ran in an isolated worktree and failed on existing
repository-wide Ruff, spelling, Clang, Markdown and forbidden-import issues;
those unrelated changes were not carried into this fix. Raw captured logs are
preserved verbatim and excluded from spelling checks.

`start-fast-image-paced.sh` preserves the last image-enabled 2,560-token,
three-slot, MTP2, `[3,9]`, bounded-checkpoint profile and starts the controller
alongside the existing emergency watchdog on a future authorized launch.
Its control log is `thermal-control-8001.jsonl` beside the watchdog log.
The native HC selection remains the previously qualified resident transaction.
The controller module and deferred launcher have been staged in the isolated
runtime, **without executing either**. See `deferred-profile.json`.

Source files: `tools/qwen4exp/thermal_controller.py` and
`tests/ut/qwen38_1m/test_thermal_controller.py`. Incident records are
`incident.json`, `thermal.jsonl`, `watchdog.log` and `server.log`.
[Temperature and throughput plot](thermal.png).
[Chinese companion](REPORT.zh.md).
