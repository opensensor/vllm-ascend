"""Read resident weights and benchmark private tensors while serving is drained."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper):
    import time
    import torch
    worker = getattr(wrapper, "worker", wrapper)
    result = dict(rank=worker.rank, passed=True)
    try:
        assert worker.model_runner.execute_model_state is None
        assert getattr(worker, "profiler", None) is None
        device = worker.model_runner.device
        free, total = torch.npu.mem_get_info(device)
        result["free_bytes"] = free
        assert free > 500 * 1024**2
        if MODE == "schedule":
            from tools.qwen4exp.resident_candidates.qfn_w4_prefill_probe import run_validation
            state = worker._qfn_w4_prefill_probe_v3
            assert state["loaded"] and not state.get("failed")
            result["projections"] = [run_validation(worker, state["resources"],
                dict(action="schedule", projection=projection, rows=[5150], repeats=5), torch)
                for projection in ("gate_up_proj", "down_proj")]
        else:
            generator = torch.Generator().manual_seed(817)
            records = []
            for rows, short in ((515, 3), (518, 6), (521, 9), (1923, 3)):
                for shuffled in (False, True):
                    order = (torch.randperm(rows, generator=generator) if shuffled else torch.arange(rows)).to(device)
                    left, right = order[:short], order[short:]
                    a = torch.randn(short, 12, 128, generator=generator, dtype=torch.float16).to(device)
                    b = torch.randn(rows-short, 12, 128, generator=generator, dtype=torch.float16).to(device)
                    expected = torch.empty(rows, 12, 128, dtype=torch.float16, device=device)
                    expected[left], expected[right] = a, b
                    output = torch.empty_like(expected)
                    inverse = torch.empty_like(order)
                    inverse[left] = torch.arange(short, device=device)
                    inverse[right] = torch.arange(short, rows, device=device)
                    def indexed():
                        output[left], output[right] = a, b
                    def scattered():
                        output.scatter_(0, left[:, None, None].expand_as(a), a)
                        output.scatter_(0, right[:, None, None].expand_as(b), b)
                    def gathered():
                        return torch.cat((a, b), dim=0).index_select(0, inverse)
                    case = dict(rows=rows, short=short, shuffled=shuffled, timings={})
                    alternative = ("scatter", scattered) if MODE == "scatter" else ("gather", gathered)
                    for name, fn in (("indexed", indexed), alternative):
                        actual = fn()
                        torch.npu.synchronize()
                        actual = output if actual is None else actual
                        case[name + "_exact"] = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                        times = []
                        for repeat in range(5):
                            start = time.perf_counter()
                            for i in range(20):
                                fn()
                            torch.npu.synchronize()
                            times.append((time.perf_counter() - start) * 1000 / 20)
                        case["timings"][name] = times
                    records.append(case)
            result["scatter"] = records
        torch.npu.synchronize()
    except Exception as error:
        result["passed"] = False
        result["error"] = type(error).__name__ + ": " + str(error)
    return result

class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=all(r["passed"] for r in self.rows), workers=self.rows)
    def apply(self, engine):
        return self.validate(engine)
def prepare(engine): return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("schedule", "scatter", "gather"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    result = apply(client, f"MODE={args.mode!r}\n" + SOURCE, args.output)
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
