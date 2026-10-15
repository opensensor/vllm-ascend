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
        native = worker._qfn_gdn_output_adapter
        device = native.device
        generator = torch.Generator().manual_seed(53177)
        cases = []
        for tokens,zero_gate in ((64,True),(512,True),(512,False),(2048,False)):
            def normal(shape):
                return (torch.randn(shape,generator=generator)*0.1).half().to(device)
            q,k = [normal((1,4,tokens,128)) for _ in range(2)]
            v = normal((1,12,tokens,128))
            h = normal((1,12,tokens//64,128,128))
            g = (torch.zeros(1,12,tokens) if zero_gate else
                 -torch.rand(1,12,tokens//64,64,generator=generator).cumsum(-1).reshape(1,12,tokens)*0.02).to(device)
            outputs = [native(q,k,v,h,128**-0.5,g=g).cpu() for _ in range(4)]
            qc,kc,vc,hc,gc = [t.cpu().float() for t in (q,k,v,h,g)]
            expected = torch.empty_like(vc)
            for head in range(12):
                for chunk in range(tokens//64):
                    sl = slice(chunk*64,(chunk+1)*64)
                    qi,ki,vi = qc[0,head//3,sl],kc[0,head//3,sl],vc[0,head,sl]
                    gi = gc[0,head,sl]
                    mask = torch.ones(64,64,dtype=torch.bool).tril()
                    weights = (qi@ki.T)*(gi[:,None]-gi[None,:]).clamp_max(0).exp()
                    weights = weights.masked_fill(~mask,0).half().float()
                    expected[0,head,sl] = ((weights@vi)+gi.exp()[:,None]*(qi@hc[0,head,chunk]))*128**-0.5
            cases.append(dict(tokens=tokens,zero_gate=zero_gate,
                repeat_exact=all(torch.equal(outputs[0].view(torch.int16),o.view(torch.int16)) for o in outputs[1:]),
                reference_allclose=torch.allclose(expected.half(),outputs[0],atol=0.002,rtol=0.02),
                max_abs=(expected.half().float()-outputs[0].float()).abs().max().item()))
        result.update(cases=cases,status=native.status(),passed=all(
            row["repeat_exact"] and row["reference_allclose"] for row in cases))
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
