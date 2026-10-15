"""Check complete GDN outputs and recurrent state using private retained inputs."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper):
    import torch
    from types import FunctionType
    from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_310
    worker = getattr(wrapper, "worker", wrapper)
    result = dict(rank=worker.rank, passed=False)
    try:
        assert worker.model_runner.execute_model_state is None
        owner = worker._qfn_wy_graph
        assert owner.enabled and not owner.verify and not owner.failed
        values = tuple(value.clone() for value in owner.record.inputs)
        generator = torch.Generator().manual_seed(856)
        state = (torch.randn(1, 12, 128, 128, generator=generator) * 0.01).to(values[0].device)
        reference = FunctionType(chunk_gated_delta_rule_310.__code__,
            dict(chunk_gated_delta_rule_310.__globals__, _compute_kernel_inputs_from_torch_wy=owner.original),
            chunk_gated_delta_rule_310.__name__, chunk_gated_delta_rule_310.__defaults__,
            chunk_gated_delta_rule_310.__closure__)
        reference.__kwdefaults__ = chunk_gated_delta_rule_310.__kwdefaults__
        expected = reference(*values, initial_state=state.clone(), output_final_state=True)
        actual = chunk_gated_delta_rule_310(*values, initial_state=state.clone(),
            output_final_state=True)
        result["exact_output_and_state"] = [torch.equal(a.contiguous().view(torch.int16),
                                                       b.contiguous().view(torch.int16))
                                            for a,b in zip(actual, expected)]
        result["state_float32"] = actual[1].dtype == torch.float32
        result["passed"] = all(result["exact_output_and_state"]) and result["state_float32"]
        result["static_tensor_bytes"] = owner.status()["static_tensor_bytes"]
        result["allocated_bytes"] = torch.npu.memory_allocated(values[0].device)
        result["reserved_bytes"] = torch.npu.memory_reserved(values[0].device)
        result["free_bytes"] = torch.npu.mem_get_info(values[0].device)[0]
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    import json
    from pathlib import Path
    Path(OUTPUT, f"rank{worker.rank}-gate.json").write_text(json.dumps(result, indent=2))
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
    result = apply(client, f"OUTPUT={str(args.output)!r}\n" + SOURCE, args.output)
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
