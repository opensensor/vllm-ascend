# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded resident NPU profiling through two existing cold diagnostic RPCs.

Start with a JSON object containing output_dir, tag, delay and max. Stop without
arguments after the controlled request drains. No graph or global profiler
configuration changes occur. Backend errors retain ownership for inspection.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import regex as re

WORKER_TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310"
MAX_DELAY_ITERATIONS = 128
MAX_TRACE_ITERATIONS = 64


def parse_config(payload):
    values = json.loads(payload)
    if not isinstance(values, dict) or set(values) != {"output_dir", "tag", "delay", "max"}:
        raise ValueError("expected output_dir, tag, delay and max only")
    if not isinstance(values["output_dir"], str) or not Path(values["output_dir"]).is_absolute():
        raise ValueError("output_dir must be an absolute path")
    if not isinstance(values["tag"], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", values["tag"]) is None:
        raise ValueError("tag must be a short filename component")
    for name, minimum, maximum in (("delay", 0, MAX_DELAY_ITERATIONS), ("max", 1, MAX_TRACE_ITERATIONS)):
        if type(values[name]) is not int or not minimum <= values[name] <= maximum:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return values


def dependencies():
    # Worker-only imports: CPU tests and resident preparation need no NPU init.
    from vllm.config import ProfilerConfig

    from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

    return ProfilerConfig, TorchNPUProfilerWrapper


def receipt(worker, session):
    wrapper = session.wrapper
    result = dict(
        rank=worker.rank,
        tag=session.tag,
        output_dir=str(session.output_dir),
        armed=bool(wrapper._active),
        running=bool(wrapper.is_running),
        backend_starts=session.starts,
        backend_stops=session.stops,
        delay=session.delay,
        max_iterations=session.maximum,
    )
    if session.error is not None:
        result["error"] = session.error
    return result


def observe_lifecycle(session, operation):
    """Observe errors before WorkerProfiler's logging-only exception handler."""
    original = getattr(session.wrapper, f"_{operation}")

    def observed():
        try:
            result = original()
        except BaseException as error:
            session.error = f"backend {operation}: {type(error).__name__}: {error}"
            raise
        if operation == "start":
            session.starts += 1
        else:
            session.stops += 1
        return result

    return observed


def make_rpcs(original_start, original_stop, *, dependency_factory=dependencies):
    def start(worker, payload=None):
        try:
            if getattr(worker, "_qfn_native_profile_session", None) is not None:
                raise RuntimeError("native profiler session already exists")
            if getattr(worker, "profiler", None) is not None:
                raise RuntimeError("worker profiler already exists")
            if payload is None:
                return original_start(worker)
            if getattr(worker, "_qfn_cpu_profile", None) is not None:
                raise RuntimeError("CPU profiler is active")
            values = parse_config(payload)
            config_type, wrapper_type = dependency_factory()
            output_dir = Path(values["output_dir"]) / values["tag"] / f"rank{worker.rank}"
            config = config_type(
                profiler="torch",
                torch_profiler_dir=str(output_dir),
                torch_profiler_with_stack=False,
                torch_profiler_with_memory=False,
                ignore_frontend=True,
                delay_iterations=values["delay"],
                max_iterations=values["max"],
            )
            output_dir.mkdir(parents=True, exist_ok=False)
            wrapper = wrapper_type(config, f"{values['tag']}_rank{worker.rank}")
            session = SimpleNamespace(
                wrapper=wrapper,
                output_dir=output_dir,
                tag=values["tag"],
                delay=values["delay"],
                maximum=values["max"],
                starts=0,
                stops=0,
                error=None,
            )
            wrapper._start = observe_lifecycle(session, "start")
            wrapper._stop = observe_lifecycle(session, "stop")
            worker._qfn_native_profile_session = session
            worker.profiler = wrapper
            wrapper.start()
            if session.error is None and (not wrapper._active or (values["delay"] == 0 and not wrapper.is_running)):
                session.error = "profiler did not enter the requested active state"
            return receipt(worker, session)
        except BaseException as error:
            return dict(rank=worker.rank, error=f"{type(error).__name__}: {error}")

    def stop(worker, output=None):
        try:
            session = getattr(worker, "_qfn_native_profile_session", None)
            if session is None:
                if output is not None:
                    return original_stop(worker, output)
                raise RuntimeError("native profiler was not started")
            if output is not None:
                raise RuntimeError("stop native profiling without an output argument")
            if worker.profiler is not session.wrapper:
                raise RuntimeError("worker profiler ownership changed")
            if session.error is not None:
                return receipt(worker, session)
            session.wrapper.stop()
            if session.error is None and (
                session.wrapper._active or session.wrapper.is_running or session.starts != session.stops
            ):
                session.error = "profiler did not finish a balanced stop"
            result = receipt(worker, session)
            if session.error is None:
                worker.profiler = None
                worker._qfn_native_profile_session = None
                result.update(stopped=True, ownership_released=True, recorded=session.starts > 0)
            return result
        except BaseException as error:
            return dict(rank=worker.rank, error=f"{type(error).__name__}: {error}")

    return start, stop


def replacements():
    from vllm_ascend._310p.worker_310p import NPUWorker310

    start, stop = make_rpcs(NPUWorker310.qfn_cpu_profile_start, NPUWorker310.qfn_cpu_profile_stop)
    return {f"{WORKER_TARGET}.qfn_cpu_profile_start": start, f"{WORKER_TARGET}.qfn_cpu_profile_stop": stop}
