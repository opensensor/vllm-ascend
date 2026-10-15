"""Private downstream equivalence diagnosis; public serving stays on eager WY."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper):
    import torch
    import torch_npu
    from types import FunctionType
    from tools.qwen4exp.wy_graph import WYGraph
    from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_310
    worker = getattr(wrapper, "worker", wrapper)
    result = dict(rank=worker.rank, passed=True)
    try:
        assert worker.model_runner.execute_model_state is None
        owner = worker._qfn_wy_graph
        assert not owner.enabled and not owner.failed
        candidate = WYGraph(owner.record)
        candidate.verify = False
        values = tuple(value.clone() for value in owner.record.inputs)
        def function(callback):
            fn = FunctionType(chunk_gated_delta_rule_310.__code__,
                dict(chunk_gated_delta_rule_310.__globals__, _compute_kernel_inputs_from_torch_wy=callback),
                chunk_gated_delta_rule_310.__name__, chunk_gated_delta_rule_310.__defaults__,
                chunk_gated_delta_rule_310.__closure__)
            fn.__kwdefaults__ = chunk_gated_delta_rule_310.__kwdefaults__
            return fn
        def info(tensor):
            return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                        format=torch_npu.get_npu_format(tensor), offset=tensor.storage_offset())
        base_inputs = owner.original(*values, 64)
        replay_inputs = candidate(*values, 64)
        result["inputs"] = [dict(reference=info(a), candidate=info(b), exact=torch.equal(a,b))
                            for a,b in zip(base_inputs, replay_inputs)]
        state = torch.zeros(1, 12, 128, 128, device=values[0].device)
        def compare(a, b):
            return dict(exact=torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)),
                        nan_a=torch.isnan(a).sum().item(), nan_b=torch.isnan(b).sum().item(),
                        max_abs=(a.float()-b.float()).abs().max().item(),
                        mismatches=(a!=b).sum().item())
        cases = []
        for varlen in (False, True):
            cu = torch.tensor([0, 512], device=values[0].device, dtype=torch.int32) if varlen else None
            outputs = []
            for callback in (owner.original, owner.original, candidate):
                pair = function(callback)(*values, initial_state=state.clone(),
                    output_final_state=True, cu_seqlens=cu)
                outputs.append(tuple(value.cpu() for value in pair))
            cases.append(dict(varlen=varlen,
                repeat=[compare(a,b) for a,b in zip(outputs[0],outputs[1])],
                graph=[compare(a,b) for a,b in zip(outputs[1],outputs[2])]))
        result["cases"] = cases
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        result["passed"] = False
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
