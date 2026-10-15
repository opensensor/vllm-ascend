# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold private-tensor gate for the exact blocked QSA recapture window.

This never enables or calls the QSA model replacement. The temporary RPC is
qfn_cpu_profile_stop() with no argument; normal output-path calls delegate.
Passing this window gate does not qualify QSA or model-state recapture.
"""

from tools.qwen4exp.resident_candidates.qfn_capture_semantics_probe import (
    TARGET,
    _error,
    _reset_owned_capture,
    make_rpc,
)


def _scalar(tensor):
    return float(tensor.cpu().item())


def observe_window(torch_module, capture_type, monitor, dispatcher, window, *, mixed):
    """Use the supplied exact window; synchronize only after consumer enqueue."""
    record = dict(case="window_graph_eager_graph" if mixed else "window_graph_only", passed=False)
    capture = None
    capture_stream = None
    caller_stream = torch_module.npu.current_stream()
    previous_monitor = monitor.cudagraph_capturing_enabled
    resources_before = dispatcher.recapture_resources
    try:
        if caller_stream != torch_module.npu.default_stream():
            raise RuntimeError("window probe must enter on the normal default stream")
        if capture_type.current() is not None:
            raise RuntimeError("cannot probe during an existing graph capture")
        device = f"npu:{torch_module.npu.current_device()}"
        with torch_module.inference_mode():
            # Warm independent scratch before the asynchronous visibility test.
            scratch = torch_module.zeros(1, dtype=torch_module.float32, device=device)
            for increment in (1.0, 10.0, 100.0):
                scratch.add_(increment)
            caller_stream.synchronize()
            state = torch_module.zeros_like(scratch)
            state.add_(7.0)  # Producer write on the default calling stream.
            eager_snapshots = []
            with window(dispatcher):
                capture_stream = torch_module.npu.current_stream()
                if capture_stream == caller_stream or capture_stream == torch_module.npu.default_stream():
                    raise RuntimeError("exact recapture window did not select a distinct non-default stream")
                if not monitor.cudagraph_capturing_enabled:
                    raise RuntimeError("exact recapture window did not open the capture monitor")
                producer_visible = state.clone()
                capture = capture_type()

                def eager_increment():
                    before = state.clone()
                    state.add_(10.0)
                    eager_snapshots.append((before, state.clone()))

                with capture:
                    state.add_(1.0)
                    if mixed:
                        capture.add_eager(eager_increment)
                        state.add_(100.0)
                after_capture = state.clone()
                capture.replay()
                after_replay = state.clone()
            # The consumer is enqueued immediately on the restored caller
            # stream, relying on the window's completion event/wait contract.
            consumer = state * 2.0
            caller_stream.synchronize()
            captured, replayed = _scalar(after_capture), _scalar(after_replay)
            observed_producer, observed_consumer = _scalar(producer_visible), _scalar(consumer)
            observations = [
                dict(before_eager=_scalar(before), after_eager=_scalar(after)) for before, after in eager_snapshots
            ]
            resources = dispatcher.recapture_resources
            delta = 111.0 if mixed else 1.0
            record.update(
                producer_value=observed_producer,
                state_after_capture=captured,
                state_after_one_replay=replayed,
                replay_delta=replayed - captured,
                expected_replay_delta=delta,
                consumer_value=observed_consumer,
                expected_consumer_value=replayed * 2.0,
                eager_observations=observations,
                native_graphs=capture.num_graphs,
                eager_breaks=capture.num_eager_breaks,
                capture_stream_nondefault=capture_stream != torch_module.npu.default_stream(),
                caller_stream_restored=torch_module.npu.current_stream() == caller_stream,
                monitor_before=previous_monitor,
                monitor_after=monitor.cudagraph_capturing_enabled,
                resources_retained=resources is not None and len(resources) == 3 and resources[0] == capture_stream,
                resources_reused=resources_before is None or resources is resources_before,
                dispatcher_failed=dispatcher.failed,
            )
            record["passed"] = (
                observed_producer == 7.0
                and replayed - captured == delta
                and observed_consumer == replayed * 2.0
                and len(observations) == 2 * int(mixed)
                and all(item["after_eager"] - item["before_eager"] == 10.0 for item in observations)
                and capture.num_graphs == (2 if mixed else 1)
                and capture.num_eager_breaks == int(mixed)
                and record["capture_stream_nondefault"]
                and record["caller_stream_restored"]
                and record["monitor_after"] == previous_monitor
                and record["resources_retained"]
                and record["resources_reused"]
                and not dispatcher.failed
            )
    except BaseException as error:
        record["error"] = _error(error)
    finally:
        if capture is not None:
            cleanup = _reset_owned_capture(capture, capture_type, capture_stream)
            record.update(cleanup)
            if cleanup["reset_errors"] or cleanup["reset_graphs"] != capture.num_graphs:
                record["passed"] = False
        # Restoration is part of the tested contract, including failure. Do
        # not mask a broken window by silently fixing its stream/monitor here.
        record["caller_stream_restored"] = torch_module.npu.current_stream() == caller_stream
        record["monitor_restored"] = monitor.cudagraph_capturing_enabled == previous_monitor
        if not record["caller_stream_restored"] or not record["monitor_restored"]:
            record["passed"] = False
    return record


def run_probe():
    import torch
    from vllm.compilation import monitor
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    from tools.qwen4exp.resident_candidates import qsa_graph_dispatch

    dispatcher = qsa_graph_dispatch.PrefixDispatcher({})
    records = []
    for mixed in (False, True):
        record = observe_window(
            torch,
            BreakableCUDAGraphCapture,
            monitor,
            dispatcher,
            qsa_graph_dispatch.serialized_recapture_window,
            mixed=mixed,
        )
        records.append(record)
        if not record["passed"]:
            break
    return dict(
        scope="exact_recapture_window_private_tensors_not_qsa_or_model_qualification",
        window="tools.qwen4exp.resident_candidates.qsa_graph_dispatch.serialized_recapture_window",
        qsa_factory_still_blocked=not qsa_graph_dispatch.RESIDENT_RECAPTURE_WINDOW_QUALIFIED,
        passed=len(records) == 2 and all(record["passed"] for record in records),
        records=records,
    )


def replacements():
    from vllm_ascend._310p.worker_310p import NPUWorker310

    return {TARGET: make_rpc(NPUWorker310.qfn_cpu_profile_stop, probe=run_probe)}
