# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold RPC probe of qualified graph record versus execute semantics.

Temporarily reuses qfn_cpu_profile_stop() with no output argument. Supplying
an output path retains the original profiler-stop behavior. Only private tiny
local tensors and this probe's graph handles are modified. No model, cache,
collective owner, graph factory, or process-global configuration is changed.
"""

TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"


def _error(error):
    return f"{type(error).__name__}: {error}"


def _reset_owned_capture(capture, capture_type, stream):
    errors = []
    try:
        stream.synchronize()
    except BaseException as error:
        errors.append(_error(error))
    # __enter__ can fail after installing the thread-local active capture.
    # Close only our own context, never a preexisting model capture.
    if capture_type.current() is capture:
        try:
            capture.__exit__(None, None, None)
        except BaseException as error:
            errors.append(_error(error))
    graphs = {}
    for segment in capture.segments:
        graph = getattr(segment, "__self__", None)
        if graph is not None and callable(getattr(graph, "reset", None)):
            graphs[id(graph)] = graph
    pending = getattr(capture, "_current_graph", None)
    if pending is not None:
        graphs[id(pending)] = pending
    reset_count = 0
    for graph in graphs.values():
        try:
            graph.reset()
            reset_count += 1
        except BaseException as error:
            errors.append(_error(error))
    capture.segments.clear()
    return dict(reset_graphs=reset_count, reset_errors=errors)


def observe_capture(torch_module, capture_type, *, mixed):
    """Observe values; do not assume capture records-only or also executes."""
    capture = None
    stream = None
    record = dict(case="graph_eager_graph" if mixed else "graph_only", passed=False)
    try:
        if capture_type.current() is not None:
            raise RuntimeError("cannot probe during an existing graph capture")
        stream = torch_module.npu.Stream()
        stream.wait_stream(torch_module.npu.current_stream())
        device = f"npu:{torch_module.npu.current_device()}"
        with torch_module.inference_mode(), torch_module.npu.stream(stream):
            state = torch_module.zeros(1, dtype=torch_module.float32, device=device)
            # Warm the scalar-add kernel independently of the measured state.
            scratch = torch_module.zeros_like(state)
            for increment in (1.0, 10.0, 100.0):
                scratch.add_(increment)
            stream.synchronize()
            capture = capture_type()
            eager_calls = [0]
            eager_observations = []

            def eager_increment():
                eager_calls[0] += 1
                before = float(state.cpu().item())
                state.add_(10.0)
                after = float(state.cpu().item())
                eager_observations.append(dict(invocation=eager_calls[0], before_eager=before, after_eager=after))

            with capture:
                state.add_(1.0)
                if mixed:
                    capture.add_eager(eager_increment)
                    state.add_(100.0)
            stream.synchronize()
            after_capture = float(state.cpu().item())
            calls_after_capture = eager_calls[0]
            capture.replay()
            stream.synchronize()
            after_replay = float(state.cpu().item())
            expected_delta = 111.0 if mixed else 1.0
            record.update(
                state_after_capture=after_capture,
                state_after_one_replay=after_replay,
                replay_delta=after_replay - after_capture,
                expected_replay_delta=expected_delta,
                eager_calls_after_capture=calls_after_capture,
                eager_calls_after_replay=eager_calls[0],
                eager_observations=eager_observations,
                native_graphs=capture.num_graphs,
                eager_breaks=capture.num_eager_breaks,
                capture_type=f"{type(capture).__module__}.{type(capture).__qualname__}",
            )
            record["passed"] = (
                record["replay_delta"] == expected_delta
                and capture.num_graphs == (2 if mixed else 1)
                and capture.num_eager_breaks == int(mixed)
                and calls_after_capture == int(mixed)
                and eager_calls[0] == 2 * int(mixed)
            )
    except BaseException as error:
        record["error"] = _error(error)
    finally:
        if capture is not None:
            cleanup = _reset_owned_capture(capture, capture_type, stream)
            record.update(cleanup)
            if cleanup["reset_errors"] or cleanup["reset_graphs"] != capture.num_graphs:
                record["passed"] = False
    return record


def run_probe():
    # Resolve precisely the same qualified capture class used by QFN, after
    # worker initialization has installed the normal Ascend platform support.
    import torch
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    records = []
    for mixed in (False, True):
        record = observe_capture(torch, BreakableCUDAGraphCapture, mixed=mixed)
        records.append(record)
        if not record["passed"]:
            break
    return dict(
        scope="tiny_local_state_capture_semantics_not_model_parity",
        passed=len(records) == 2 and all(record["passed"] for record in records),
        records=records,
    )


def make_rpc(original_stop, probe=run_probe):
    def qfn_cpu_profile_stop(self, output=None):
        if output is not None:
            return original_stop(self, output)
        try:
            if getattr(self, "_qfn_cpu_profile", None) is not None:
                raise RuntimeError("CPU profiling is active; supply its output path to stop it normally")
            report = probe()
            return dict(rank=self.rank, capture_semantics=report)
        except BaseException as error:
            return dict(rank=getattr(self, "rank", None), capture_semantics=dict(passed=False, error=_error(error)))

    return qfn_cpu_profile_stop


def replacements():
    from vllm_ascend._310p.worker_310p import NPUWorker310

    return {TARGET: make_rpc(NPUWorker310.qfn_cpu_profile_stop)}
