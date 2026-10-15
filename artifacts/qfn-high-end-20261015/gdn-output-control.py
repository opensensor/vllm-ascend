"""Drained ownership transitions for the qualified private WY graph."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper, check_only=False):
    import importlib

    worker = getattr(wrapper, "worker", wrapper)
    try:
        module = importlib.import_module("vllm_ascend._310p.ops.fla.chunk_gated_delta_rule")
        assert worker.model_runner.execute_model_state is None
        owner = worker._qfn_gdn_output_adapter
        assert (
            owner.receipt["passed"]
            and owner.receipt["source_sha256"] == "58e360fc40849c01f326232d181dd902e4d56439b4c2664d9fc40333967470da"
        )
        if ACTION == "install":
            assert module.chunk_gated_delta_rule_310 is owner.original_function
            if not check_only:
                module.chunk_gated_delta_rule_310 = owner.replacement
        else:
            assert module.chunk_gated_delta_rule_310 is owner.replacement
            if not check_only:
                if ACTION == "off":
                    owner.enabled = False
                if ACTION == "on":
                    owner.enabled = True
                if ACTION == "restore":
                    module.chunk_gated_delta_rule_310 = owner.original_function
        return dict(rank=worker.rank, passed=True, gdn_output=owner.status())
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=type(error).__name__ + ": " + str(error))


def admission(worker):
    return operation(worker, check_only=True)


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "admitted"):
            self.admitted = engine.model_executor.collective_rpc(admission)
        return dict(passed=len(self.admitted) == 4 and all(r["passed"] for r in self.admitted), workers=self.admitted)

    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=all(r["passed"] for r in rows), workers=rows)


def prepare(engine):
    return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "on", "off", "status", "restore"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    if args.action in ("install", "on", "off", "restore"):
        client.request("/pause?mode=wait&clear_cache=true")
    result = apply(client, f"ACTION={args.action!r}\n" + SOURCE, args.output)
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
