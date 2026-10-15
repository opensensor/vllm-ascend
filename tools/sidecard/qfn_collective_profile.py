# SPDX-License-Identifier: Apache-2.0
"""Fresh C1 custom-collective profiling, separate from unprofiled throughput.

Use the qualified QFN launcher environment. Failure retains native owners for
inspection. No model, precision, cache or graph settings are changed here.
"""

import argparse
import json
import signal
import time
import traceback
from pathlib import Path

from . import qfn_collective_benchmark as benchmark
from .qfn_collective_worker import QfnCollectiveWorkerExtension

PROFILE_STEPS = 8
PROFILE_DELAY = 8
TRIALS = (("warmup", 32), ("before-a", 512), ("before-b", 512), ("npu", 96), ("cpu", 128), ("after", 512))


class QfnProfileWorkerExtension(QfnCollectiveWorkerExtension):
    """Cold RPCs only; sampling is explicitly restricted to diagnostic trials."""

    def qfn_native_profile_start(self, payload=None):
        """Arm bounded recording without replacing model code or graphs."""
        from tools.qwen4exp.resident_candidates import qfn_native_profile

        if not isinstance(payload, str):
            return dict(rank=self.rank, error="native profiling requires a JSON configuration string")
        start, _ = qfn_native_profile.make_rpcs(None, None, dependency_factory=qfn_native_profile.dependencies)
        return start(self, payload)

    def qfn_native_profile_stop(self):
        """Release the recorder after requests drain, retaining it on failure."""
        from tools.qwen4exp.resident_candidates import qfn_native_profile

        _, stop = qfn_native_profile.make_rpcs(None, None, dependency_factory=qfn_native_profile.dependencies)
        return stop(self)

    def qfn_native_profile_status(self):
        """Inspect owned recording state without touching device execution."""
        from tools.qwen4exp.resident_candidates import qfn_native_profile

        session = getattr(self, "_qfn_native_profile_session", None)
        if session is None:
            return dict(rank=self.rank, owned=False, profiler_present=getattr(self, "profiler", None) is not None)
        return dict(owned=True, **qfn_native_profile.receipt(self, session))

    def qfn_graph_inventory(self):
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

        records = []
        for wrapper in tuple(BreakableCUDAGraphWrapper._all_instances):
            # Only our configured wrapper owns a collective observer.
            if wrapper.__dict__.get("_collective_graph_observer") is None:
                continue
            for entry in wrapper.entries.values():
                capture = entry.capture
                if capture is not None:
                    records.append(
                        dict(
                            descriptor=str(entry.batch_descriptor),
                            graphs=capture.num_graphs,
                            eager_breaks=capture.num_eager_breaks,
                            segments=[getattr(fn, "__qualname__", type(fn).__name__) for fn in capture.segments],
                        )
                    )
        return records

    def qfn_cpu_profile_start(self):
        import cProfile

        if getattr(self, "_qfn_cpu_profile", None) is not None:
            raise RuntimeError("CPU profiler already exists")
        self._qfn_cpu_profile = cProfile.Profile()
        self._qfn_cpu_profile.enable()
        return dict(rank=self.rank, started=True)

    def qfn_cpu_profile_stop(self, output):
        profiler = getattr(self, "_qfn_cpu_profile", None)
        if profiler is None:
            raise RuntimeError("CPU profiler was not started")
        profiler.disable()
        path = Path(output) / f"rank{self.rank}.pstats"
        profiler.dump_stats(str(path))
        self._qfn_cpu_profile = None
        return dict(rank=self.rank, stopped=True, filename=path.name)


def engine_options(config, output):
    if config.get("execution_mode") != "graph" or config.get("graph_replay_tracking", False) is not False:
        raise ValueError("profiling requires the production unobserved graph collective")
    options = benchmark.engine_options(config, graph_profile="c1-c2")
    options["worker_extension_cls"] = "tools.sidecard.qfn_collective_profile.QfnProfileWorkerExtension"
    options["profiler_config"] = dict(
        profiler="torch",
        torch_profiler_dir=str(Path(output).absolute() / "npu-trace"),
        torch_profiler_with_stack=False,
        torch_profiler_with_memory=False,
        ignore_frontend=True,
        delay_iterations=PROFILE_DELAY,
        max_iterations=PROFILE_STEPS,
    )
    return options


def run(output, config, *, llm_factory=None, sampling_factory=None, collect_diagnostics=True):
    output = Path(output).absolute()
    options = engine_options(config, output)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / "cpu-profile").mkdir()
    (output / "engine-options.json").write_text(json.dumps(options, indent=2) + "\n")
    if llm_factory is None:
        from vllm import LLM, SamplingParams

        llm_factory, sampling_factory = LLM, SamplingParams
    llm = llm_factory(**options)
    initial = llm.collective_rpc("npu_collective_status")
    benchmark.verify_workers(initial, custom=True)
    result = dict(
        graph_inventory=llm.collective_rpc("qfn_graph_inventory"),
        measurements=[],
        graph_profile="c1-c2",
        profile_steps=PROFILE_STEPS,
        profile_delay=PROFILE_DELAY,
        collect_diagnostics=collect_diagnostics,
        scope="short-prompt C1; profiled rates are diagnostic only; unprofiled rates include prefill",
    )
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [dict(role="user", content=benchmark.PROMPTS[0])],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for label, tokens in TRIALS:
        if not collect_diagnostics and label in ("npu", "cpu"):
            continue
        before = llm.collective_rpc("npu_collective_status")
        if label == "npu":
            llm.start_profile("qfn_custom_c1")
        elif label == "cpu":
            llm.collective_rpc("qfn_cpu_profile_start")
        try:
            params = sampling_factory(temperature=0, max_tokens=tokens, min_tokens=tokens, ignore_eos=True)
            started = time.monotonic()
            outputs = llm.generate([prompt], params, use_tqdm=False)
            elapsed = time.monotonic() - started
        finally:
            if label == "npu":
                llm.stop_profile()
            elif label == "cpu":
                llm.collective_rpc("qfn_cpu_profile_stop", args=(str(output / "cpu-profile"),))
        receipts = [benchmark.output_receipt(item) for item in outputs]
        benchmark.verify_completion_budget(receipts, 1, tokens)
        after = llm.collective_rpc("npu_collective_status")
        result["measurements"].append(
            dict(
                label=label,
                profiled=label in ("npu", "cpu"),
                generated_tokens=tokens,
                elapsed_seconds=elapsed,
                wall_tokens_per_second=tokens / elapsed,
                outputs=receipts,
                collective_coverage=benchmark.coverage_deltas(before, after, custom=True, require_calls=False),
                graph_replay_coverage=benchmark.replay_deltas(before, after, required=True),
            )
        )
        (output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
    result["closed_workers"] = llm.collective_rpc("npu_collective_close")
    benchmark.verify_workers(result["closed_workers"], custom=True, closed=True)
    llm.llm_engine.engine_core.shutdown()
    result["ownership_released"] = True
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--collective-config", required=True, type=Path)
    parser.add_argument("--throughput-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args.output, json.loads(args.collective_config.read_text()), collect_diagnostics=not args.throughput_only)
    except BaseException:
        traceback.print_exc()
        while True:
            signal.pause()


if __name__ == "__main__":
    main()
