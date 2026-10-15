# Engine-control upgrade

Status: the corrected hook is active in `resident-live-v6`; engine status,
public model identity, and worker configuration requalification passed.
The service remains on port 8001 as `qwen38-flash-next`, with max model length
262,144 and 1,068,936 planner cache tokens.

The existing authenticated resident interface changes worker code. It cannot
install methods in the separate, already-running engine process. That process
still waits for sampled output, updates the synchronous scheduler, and submits
the next step. Removing that round trip requires engine-side work as well as
device-side state and input preparation.

[`qfn_engine_control.py`](../../tools/sidecard/qfn_engine_control.py) adds an
opt-in startup hook and authenticated `/qfn_engine_control` route. A top-level
multiprocessing entrypoint installs the engine utility under spawn or fork.
The process-manager launch hook selects that target after Ascend's late
platform patching, preserving the Ascend child configuration wrapper.
The existing real-loopback/token middleware protects it. Preparation requires
a source digest and fresh revision. Application requires a paused, drained
engine, rechecks admission, and retains owners on uncertainty without retry or
automatic resume. Enabling this control does not enable asynchronous scheduling.

The 48 scoped engine/server tests pass, including a real spawned-process test,
late platform entrypoint replacement, authentication, staged application,
changed admission, and failure ownership. The initial v5 startup exposed a
missed integration condition: Ascend replaced the early hook, and spawn
reimported its original wrapper without our engine method. Its control route
returned HTTP 500. Both v4 and v5 shut down with four closed collective receipts.
The user then authorized further restarts as needed. The corrected v6 startup
returns successful engine status; the scheduler remains synchronous.
The live transaction gate rejected unpaused updates and an incorrect digest,
then prepared/applied a diagnostic candidate in engine PID 1942251. This gate
deliberately left scheduling unchanged. The service resumed afterward.

The restored device transition passed 197 steps, 216 bank-content comparisons,
and 188 input batches per rank. Both the base and transition ten-request suites
passed, including 8K prefill and mixed-length C4. General and prefix graph
captures cover `[3,6,9,12]`. Verification was disabled after qualification.

## Concrete restart scope

1. Finish/drain current requests and retire the custom collective through the
   existing shutdown guard. Never bypass failed peer retirement.
2. Launch `start-qfn-engine-control-candidate.sh --run-root
   /srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v6`.
   It preserves the model, port, weights, memory, MTP and initial graph settings;
   the new flag is `--engine-control`. It initially retains synchronous scheduling.
3. Use `qfn-recovery-v6.py` to repeat the existing W4, HC, gather and geometry
   loading/validation sequence, then restore the combined `[3,6,9,12]` general
   and prefix graph candidate. Warm and validate compact model mappings.
4. Run `restore-transition-v6.py` to reload the retained native commit owner and
   install the already-qualified transition bridge. It enables diagnostics;
   complete the model gate, then disable diagnostics through a drained control.
5. Verify model identity, `/health`, the new engine status route, and C1/C4.
   Continue scheduler and device-history development through the new control.

This restart installs a control capability and restores the verified worker
path. It does **not** itself complete the requested host-independent iteration.
Device PLE history/lookup, cache ownership across queued steps, and engine
output scheduling still require implementation and qualification. Do not bypass
the current QFN async guards merely to report async mode enabled.

`requalify-engine-v6.sh` records the kernel and model restoration checks.
Private control tokens remain only on the hardware host.
