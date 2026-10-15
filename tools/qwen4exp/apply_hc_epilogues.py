# SPDX-License-Identifier: Apache-2.0
"""Drain, qualify, and attach private eager HC epilogues without replacing graphs."""

import argparse
import hashlib
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper, check_only=False):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import hashlib
        from pathlib import Path
        import torch
        from vllm_ascend.models.qwen4_exp.model import _GatedResidual
        state = getattr(worker, "_qfn_hc_epilogues", None)
        if state is None:
            assert ACTION == "install"
            for name, digest in DIGESTS.items():
                assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
            namespace = {"__name__": "qfn_hc_epilogues_private"}
            exec(compile(Path(MODULE).read_text(), MODULE, "exec"), namespace)
            torch.ops.load_library(BRIDGE)
            # Retain each native object immediately, including on gate failures.
            state = {"namespace": namespace, "kernels": [], "original": _GatedResidual.mix}
            worker._qfn_hc_epilogues = state
            for entry in ("qwen_hc_affine_v1", "qwen_hc_mean_epilogue_v1"):
                state["kernels"].append(torch.classes.qwen_hc_combine_v1.Kernel(BINARY, entry))
            owner = namespace["HCEpilogues"](*state["kernels"], torch.ops.qwen_hc_combine_v1.launch)
            state["owner"] = owner
            state["adapter"] = namespace["HCEpilogueAdapter"](owner, state["original"])
            cases = []
            generator = torch.Generator().manual_seed(15892)
            with torch.no_grad():
                for rows in (3, 259, 515, 1024, 2048):
                    for repeat in range(2):
                        x = torch.randn(rows, 10240, generator=generator).to(worker.device)
                        weight = torch.randn(10240, generator=generator).to(worker.device)
                        gate = torch.sigmoid(torch.randn(rows, 10240, generator=generator)).half().to(worker.device)
                        if repeat:
                            x.fill_({3: -0.0, 259: 2**-24, 515: 1e20,
                                     1024: float("nan"), 2048: -float("inf")}[rows])
                            weight.fill_(1.0)
                            gate.fill_(1.0)
                        expected_full = x * weight
                        full, narrow = owner.affine(x, weight)
                        actual_mean = owner.mean(x, gate)
                        expected_mean = (x.view(rows, 4, 2560) * gate.view(rows, 4, 2560)).mean(1).half()
                        passed = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a,b in (
                            (expected_full, full), (expected_full.half(), narrow), (expected_mean, actual_mean)))
                        cases.append(dict(rows=rows, repeat=repeat, passed=passed))
            state["gate"] = dict(passed=all(row["passed"] for row in cases), cases=cases)
        assert state["gate"]["passed"] and not state["owner"].failed
        adapter = state["adapter"]
        assert worker.model_runner.execute_model_state is None
        if ACTION == "install":
            assert _GatedResidual.mix is state["original"]
            if not check_only:
                _GatedResidual.mix = adapter.replacement
        else:
            assert _GatedResidual.mix is adapter.replacement
            if ACTION == "on":
                assert adapter.checked >= 48 and adapter.mismatches == 0
            if not check_only:
                if ACTION == "on": adapter.enabled, adapter.verify = True, False
                if ACTION == "off": adapter.enabled = False
                if ACTION == "verify": adapter.enabled, adapter.verify = True, True
        return dict(rank=worker.rank, passed=True, hc=adapter.status(), gate=state["gate"])
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=repr(error))

def admission(worker):
    return operation(worker, check_only=True)

class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(admission)
        return dict(passed=len(self.rows) == 4 and all(row["passed"] for row in self.rows), workers=self.rows)
    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(rows) == 4 and all(row["passed"] for row in rows), workers=rows)

def prepare(engine):
    return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "on", "off", "verify", "status"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    args = parser.parse_args()
    paths = dict(MODULE=args.module, BINARY=args.binary, BRIDGE=args.bridge)
    header = f"ACTION={args.action!r}\n"
    header += "".join(f"{name}={str(path.resolve(strict=True))!r}\n" for name, path in paths.items())
    digests = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}
    header += f"DIGESTS={digests!r}\n"
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    result = apply(client, header + SOURCE, args.output)
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
