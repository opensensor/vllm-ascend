"""Private graph qualification using the resident WY math and worker-owned buffers."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper):
    import importlib
    import time
    from types import SimpleNamespace
    import torch
    worker = getattr(wrapper, "worker", wrapper)
    result = dict(rank=worker.rank, passed=False)
    try:
        assert worker.model_runner.execute_model_state is None
        assert getattr(worker, "profiler", None) is None
        assert getattr(worker, "_qfn_wy_graph_probe", None) is None
        module = importlib.import_module("vllm_ascend._310p.ops.fla.chunk_gated_delta_rule")
        original = module._compute_kernel_inputs_from_torch_wy
        device = worker.model_runner.device
        free, total = torch.npu.mem_get_info(device)
        assert free > 600 * 1024**2
        result["free_before"] = free
        generator = torch.Generator().manual_seed(859)
        def inputs():
            q, k = [torch.nn.functional.normalize(
                torch.randn(1, 512, 4, 128, generator=generator), dim=-1).half().to(device) for _ in range(2)]
            v = torch.randn(1, 512, 12, 128, generator=generator).half().to(device)
            g = -torch.rand(1, 512, 12, generator=generator).to(device)
            beta = torch.rand(1, 512, 12, generator=generator).half().to(device)
            return q, k, v, g, beta
        owner = SimpleNamespace(inputs=inputs(), graph=torch.npu.NPUGraph(),
                                stream=torch.npu.Stream(device=device), outputs=None, failed=True,
                                original=original, replay_stream=torch.npu.current_stream())
        worker._qfn_wy_graph_probe = owner
        owner.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(owner.stream):
            for _ in range(3):
                warm = original(*owner.inputs, 64)
            torch.npu.synchronize()
            del warm
            with torch.npu.graph(owner.graph, stream=owner.stream):
                owner.outputs = original(*owner.inputs, 64)
            torch.npu.synchronize()
        torch.npu.current_stream().wait_stream(owner.stream)
        torch.npu.synchronize()
        def replay(values):
            for dst, src in zip(owner.inputs, values): dst.copy_(src)
            owner.graph.replay()
            return tuple(value.clone() for value in owner.outputs)
        checks = []
        for _ in range(3):
            values = inputs()
            expected = original(*values, 64)
            actual = replay(values)
            checks.append([torch.equal(a.view(torch.int16), b.view(torch.int16)) for a,b in zip(actual, expected)])
        result["exact"] = checks
        result["timings"] = {}
        for name, fn in (("eager", lambda: original(*values, 64)), ("graph", lambda: replay(values))):
            elapsed = []
            for _ in range(5):
                torch.npu.synchronize()
                start = time.perf_counter()
                for repeat in range(10): output = fn()
                torch.npu.synchronize()
                elapsed.append((time.perf_counter()-start)*1000/10)
            result["timings"][name] = elapsed
        result["free_after"] = torch.npu.mem_get_info(device)[0]
        result["passed"] = all(all(c) for c in checks)
        owner.failed = not result["passed"]
        owner.receipt = result
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    return result

class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "rows"): self.rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(self.rows)==4 and all(r["passed"] for r in self.rows),workers=self.rows)
    def apply(self, engine): return self.validate(engine)
def prepare(engine): return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    result = apply(client, SOURCE, args.output)
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
