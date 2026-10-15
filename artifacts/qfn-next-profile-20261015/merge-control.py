"""Drained, reversible admission of exact mixed GDN output assembly."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper, check_only=False):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        from tools.qwen4exp.mixed_gdn_merge import MixedGDNMerge, build_forward
        from vllm_ascend.models.qwen4_exp.model import _GDNAttention
        assert worker.model_runner.execute_model_state is None
        owner = getattr(worker, "_qfn_mixed_gdn_merge", None)
        if ACTION == "install":
            assert owner is None
            owner = MixedGDNMerge()
            owner.original = _GDNAttention._native_mixed_attention
            owner.replacement = build_forward(owner.original, owner)
            if not check_only:
                worker._qfn_mixed_gdn_merge = owner
                _GDNAttention._native_mixed_attention = owner.replacement
        else:
            assert owner is not None
            assert _GDNAttention._native_mixed_attention is owner.replacement
            if ACTION == "on":
                assert owner.checked >= 36 and not owner.failed
            if not check_only:
                if ACTION == "on": owner.verify, owner.enabled = False, True
                if ACTION == "off": owner.enabled = False
                if ACTION == "verify": owner.verify, owner.enabled = True, True
                if ACTION == "restore":
                    _GDNAttention._native_mixed_attention = owner.original
                    worker._qfn_mixed_gdn_merge = None
        return dict(rank=worker.rank, passed=True, merge=owner.status())
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=type(error).__name__ + ": " + str(error))
def admission(worker): return operation(worker, check_only=True)
class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "admitted"):
            self.admitted = engine.model_executor.collective_rpc(admission)
        return dict(passed=len(self.admitted)==4 and all(r["passed"] for r in self.admitted),workers=self.admitted)
    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=all(r["passed"] for r in rows), workers=rows)
def prepare(engine): return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "on", "off", "verify", "status", "restore"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    result = apply(client, f"ACTION={args.action!r}\n" + SOURCE, args.output)
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
