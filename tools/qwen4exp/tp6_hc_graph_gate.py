# SPDX-License-Identifier: Apache-2.0
"""Private real-weight HC mix graphs at TP6 mixed-batch shapes; no binding."""

import inspect
import statistics
import time
from pathlib import Path

import torch

from tools.qwen4exp import tp6_finalize_gate

MIN_FREE_BYTES = 1024**3


def exact(actual, reference):
    if isinstance(actual, (tuple, list)):
        return len(actual) == len(reference) and all(exact(a, b) for a, b in zip(actual, reference))
    return torch.equal(actual.cpu().contiguous().view(torch.uint8), reference.cpu().contiguous().view(torch.uint8))


def timing(functions):
    samples = {name: [] for name in functions}
    for iteration in range(6):
        names = list(functions)[:: 1 if iteration % 2 == 0 else -1]
        for name in names:
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                functions[name]()
            torch.npu.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000 / 5)
    return {name: dict(median_ms=statistics.median(values), samples_ms=values) for name, values in samples.items()}


def run_gate(worker, owner, resources, snapshot, early_injection=False):
    from vllm_ascend.models.qwen4_exp.model import _GatedResidual, _linear

    if getattr(worker.model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("worker is not drained")
    if torch.npu.mem_get_info()[0] < MIN_FREE_BYTES:
        raise RuntimeError("insufficient private graph headroom")
    native = inspect.getclosurevars(_GatedResidual.combine).nonlocals["op"] if early_injection else None
    before = snapshot(worker)
    modules = list(
        {
            id(module): module
            for wrapper in worker._resident_wrappers()
            for module in wrapper.runnable.modules()
            if isinstance(module, _GatedResidual) and module.use_combine
        }.values()
    )
    owner["records"] = records = []
    owner["results"] = results = []
    stream = torch.npu.Stream(device=worker.model_runner.device)
    owner["stream"] = stream
    stream.wait_stream(torch.npu.current_stream())
    generator = torch.Generator().manual_seed(191016 + worker.rank)
    allocated_before = torch.npu.memory_allocated()
    reserved_before = torch.npu.memory_reserved()
    with torch.inference_mode(), torch.npu.stream(stream):
        for index in (0, len(modules) // 2, len(modules) - 1):
            module = modules[index]
            for rows in (131, 134, 259, 262) if index == 0 else (131, 134):
                value = (
                    torch.randn((rows, module.hyper_hidden), generator=generator).half().to(worker.model_runner.device)
                )
                block = torch.randn((rows, module.hidden_size), generator=generator).half().to(value.device)
                record = dict(input=value.clone(), stream=stream, module=module, failed=True)
                records.append(record)

                def captured(module=module, record=record):
                    mixed, residuals = module.mix(record["input"])
                    if early_injection:
                        injection = 2.0 * torch.sigmoid(
                            _linear(residuals[1], module.block_inject_weight, module.compute_dtype) / module.hc_count
                        )
                        return mixed, (record["input"], injection)
                    return mixed, residuals

                def combine(block, residuals, module=module):
                    return (
                        native(residuals[0], block, residuals[1])
                        if early_injection
                        else module.combine(block, residuals)
                    )

                # Warm up the exact live mix; preserve any existing native combine.
                for _ in range(3):
                    captured()
                stream.synchronize()
                record["graph"] = torch.npu.NPUGraph()
                record["pool"] = torch.npu.graph_pool_handle()
                memory_before = torch.npu.memory_allocated()
                reservation_before = torch.npu.memory_reserved()
                with torch.npu.graph(record["graph"], pool=record["pool"], stream=stream):
                    record["output"] = captured()
                record["graph"].replay()
                stream.synchronize()
                memory_delta = torch.npu.memory_allocated() - memory_before
                checks = []
                for scale in (1.0, 0.01, 0.0):
                    value.copy_(torch.randn(value.shape, generator=generator).half().to(value.device) * scale)
                    expected = module.mix(value)
                    actual = resources["replay"](value, record)
                    previous = tuple(t.clone() for t in (actual[0], actual[1][1]))
                    full_expected = module.combine(block, expected[1])
                    full_actual = combine(block, actual[1])
                    full_repeat = module.combine(block, module.mix(value)[1])
                    if early_injection:
                        expected = (
                            expected[0],
                            (
                                value,
                                2.0
                                * torch.sigmoid(
                                    _linear(expected[1][1], module.block_inject_weight, module.compute_dtype)
                                    / module.hc_count
                                ),
                            ),
                        )
                    resources["replay"](value + 0.125, record)
                    stream.synchronize()
                    checks.append(
                        dict(
                            scale=scale,
                            exact=exact(actual, expected),
                            complete_exact=exact(full_actual, full_expected),
                            baseline_repeat_exact=exact(full_repeat, full_expected),
                            complete_max_abs=float((full_actual.float() - full_expected.float()).abs().max().cpu()),
                            output_lifetime=exact((actual[0], actual[1][1]), previous),
                            residual_alias=actual[1][0] is value,
                        )
                    )
                passed = all(
                    all(
                        row[k]
                        for k in (
                            "exact",
                            "complete_exact",
                            "output_lifetime",
                            "residual_alias",
                            "baseline_repeat_exact",
                        )
                    )
                    for row in checks
                )
                record["failed"] = not passed
                record["qualified"] = passed
                row = dict(
                    module=index,
                    rows=rows,
                    passed=passed,
                    checks=checks,
                    allocated_delta=memory_delta,
                    reserved_delta=torch.npu.memory_reserved() - reservation_before,
                )
                results.append(row)
                if not passed:
                    # Numerical rejection keeps owned resources until inspection.
                    continue
                replay = resources["HCMixGraph"](module.mix, record)
                row["mix_timing"] = timing(
                    dict(
                        eager=lambda module=module, value=value: module.mix(value),
                        replay=lambda replay=replay, value=value: replay(value),
                    )
                )
                row["complete_timing"] = timing(
                    dict(
                        eager=lambda module=module, block=block, value=value: module.combine(
                            block, module.mix(value)[1]
                        ),
                        replay=lambda combine=combine, block=block, replay=replay, value=value: combine(
                            block, replay(value)[1]
                        ),
                    )
                )
                row["replays"] = replay.calls
        stream.synchronize()
    torch.npu.synchronize()
    after = snapshot(worker)
    owner["completed"] = True
    return dict(
        passed=before == after,
        numerical_passed=all(row["passed"] for row in results),
        rank=worker.rank,
        before=before,
        after=after,
        modules=len(modules),
        results=results,
        allocated_delta=torch.npu.memory_allocated() - allocated_before,
        reserved_delta=torch.npu.memory_reserved() - reserved_before,
        early_injection=early_injection,
    )


def retire(owner, runtime):
    """Release only completed private captures; uncertain owners stay intact."""
    if not owner.get("completed") or owner.get("failure") or owner.get("retired"):
        raise RuntimeError("only completed private graph owners can be retired once")
    runtime.synchronize()
    for record in owner["records"]:
        record["graph"].reset()
    for record in owner["records"]:
        record.clear()
    owner["records"].clear()
    owner["retired"] = True
    return dict(passed=True, retired=True)


ENGINE = """
def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    if not hasattr(worker, "_qfn_tp6_hc_graph_gates"):
        worker._qfn_tp6_hc_graph_gates = []
    if RETIRE:
        owner = worker._qfn_tp6_hc_graph_gates[-1]
        try:
            import torch
            namespace = {}
            exec(compile(SNAPSHOT, "<tp6-hc-graph-snapshot>", "exec"), namespace)
            before = namespace["worker_snapshot"](worker)
            result = owner["gate"]["retire"](owner, torch.npu)
            after = namespace["worker_snapshot"](worker)
            return dict(result, rank=worker.rank, unchanged=before == after)
        except Exception as error:
            owner["retire_failure"] = error
            return dict(passed=False, rank=worker.rank, error=repr(error))
    owner = {}
    worker._qfn_tp6_hc_graph_gates.append(owner)
    try:
        owner["gate"] = gate = {}
        owner["resources"] = resources = {}
        snapshot = {}
        exec(compile(GATE, "<tp6-hc-graph-gate>", "exec"), gate)
        exec(compile(RESOURCES, "<tp6-hc-graph-resources>", "exec"), resources)
        exec(compile(SNAPSHOT, "<tp6-hc-graph-snapshot>", "exec"), snapshot)
        return gate["run_gate"](worker, owner, resources, snapshot["worker_snapshot"], EARLY_INJECTION)
    except Exception as error:
        owner["failure"] = error
        return dict(passed=False, rank=worker.rank, error=repr(error))
class Candidate:
    def validate(self, engine):
        if not engine._pause_complete() or engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
            raise RuntimeError("drain first")
        if not hasattr(self, "result"):
            snapshot = {}
            exec(compile(SNAPSHOT, "<tp6-hc-graph-snapshot>", "exec"), snapshot)
            before = snapshot["scheduler_snapshot"](engine)
            workers = engine.model_executor.collective_rpc(operation)
            after = snapshot["scheduler_snapshot"](engine)
            self.result = dict(passed=len(workers)==6 and all(w["passed"] for w in workers) and before==after,
                               before=before, after=after, workers=workers, installed=False)
        return self.result
    def apply(self, engine):
        raise RuntimeError("private graph gate cannot install a candidate")
def prepare(engine):
    return Candidate()
"""


def engine_source(retire_graphs=False, early_injection=False):
    return (
        f"RETIRE={retire_graphs!r}\nEARLY_INJECTION={early_injection!r}\n"
        f"GATE={Path(__file__).read_text().replace('from tools.qwen4exp import tp6_finalize_gate', '')!r}\n"
        f"RESOURCES={Path(__file__).with_name('tp6_hc_graph.py').read_text()!r}\n"
        f"SNAPSHOT={Path(tp6_finalize_gate.__file__).read_text()!r}\n" + ENGINE
    )
