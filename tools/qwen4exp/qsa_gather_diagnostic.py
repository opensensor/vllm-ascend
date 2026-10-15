# SPDX-License-Identifier: Apache-2.0
"""One-shot private gather trace; retain profiler ownership on uncertain failure."""

from pathlib import Path
from types import SimpleNamespace

import torch

from tools.qwen4exp import qsa_gather_staging
from tools.qwen4exp.resident_candidates.qfn_native_profile import dependencies, observe_lifecycle


def profile(worker, state, device, *, rows, repeats, output, dependency_factory=dependencies, torch_module=torch):
    if state.get("profile_session") is not None:
        raise RuntimeError("private gather profiling is one-shot; retain previous session")
    if any(
        getattr(worker, name, None) is not None
        for name in ("profiler", "_qfn_cpu_profile", "_qfn_native_profile_session")
    ):
        raise RuntimeError("another profiler owns the worker")
    directory = Path(output) / f"rank{worker.rank}"
    session = SimpleNamespace(
        wrapper=None, output_dir=directory, tag="qsa-gather-private", delay=0, maximum=1, starts=0, stops=0, error=None
    )
    # Publish before construction/start: traceback and this session retain every
    # successful native profiler object, including uncertain initialization.
    state["profile_session"] = session
    worker._qfn_native_profile_session = session
    directory.mkdir(parents=True, exist_ok=False)
    config_type, wrapper_type = dependency_factory()
    config = config_type(
        profiler="torch",
        torch_profiler_dir=str(directory),
        torch_profiler_with_stack=False,
        torch_profiler_with_memory=False,
        ignore_frontend=True,
        delay_iterations=0,
        max_iterations=1,
    )
    wrapper = wrapper_type(config, f"qsa_gather_private_rank{worker.rank}")
    session.wrapper = wrapper
    wrapper._start = observe_lifecycle(session, "start")
    wrapper._stop = observe_lifecycle(session, "stop")
    worker.profiler = wrapper
    wrapper.start()
    if session.error is not None or not wrapper._active or not wrapper.is_running or session.starts != 1:
        raise RuntimeError(f"private gather profiler failed to start exactly once: {session.error}")
    # Timing remains the same matched sequence, including queue-drain costs.
    # The trace's actual kernel intervals, not these instrumented timing values,
    # distinguish compute duration from host submission and event gaps.
    result = qsa_gather_staging.timing(
        state["resources"]["gather"],
        device,
        rows=rows,
        repeats=repeats,
        record_function=torch_module.profiler.record_function,
    )
    torch_module.npu.synchronize()
    wrapper.stop()
    if session.error is not None or wrapper._active or wrapper.is_running or (session.starts, session.stops) != (1, 1):
        raise RuntimeError(f"private gather profiler failed balanced stop: {session.error}")
    if worker.profiler is not wrapper or worker._qfn_native_profile_session is not session:
        raise RuntimeError("private gather profiler ownership changed")
    worker.profiler = None
    worker._qfn_native_profile_session = None
    return dict(
        passed=result.get("passed") is True,
        scope="instrumented private gather trace; parse native kernel intervals offline",
        output_dir=str(directory),
        backend_starts=session.starts,
        backend_stops=session.stops,
        ownership_released=True,
        records=result["records"],
    )
