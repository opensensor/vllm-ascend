"""Compose the qualified paired epilogue without replacing resident owners."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-local-epilogue-hardware-20261016")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v16")
SOURCE = """def operation(wrapper, check_only=False):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        from pathlib import Path
        from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE

        assert worker.model_runner.execute_model_state is None
        state = worker._qfn_local_epilogue_private_v2
        prior = worker._qfn_pack_serving
        assert not prior["owner"].failed and not prior["owner"].mismatches
        assert prior["owner"].enabled and not prior["owner"].verify and prior["owner"].local_routes
        for phase in ("projection", "packed"):
            assert state["records"][phase] and all(r["exact"] for r in state["records"][phase])
        assert all(r["exact"] for r in state["records"]["epilogue"] if r["mode"] == "sequential")
        adapter = state["adapters"]["sequential"]
        assert not adapter.failed and not adapter.mismatches
        serving = state.get("serving")
        if serving is None:
            assert ACTION == "verify"
            assert W4SparseMoE._forward_grouped is prior["replacement"]
        else:
            assert W4SparseMoE._forward_grouped in (serving["previous"], serving["replacement"])
        if ACTION == "on":
            assert adapter.checked >= 48
        if not check_only:
            if serving is None:
                namespace = {"__name__": "qfn_private_paired_binding"}
                path = Path("/srv/ai/storage/artifacts/qfn-local-epilogue-hardware-20261016/qfn_route_finalize.py")
                exec(compile(path.read_text(), str(path), "exec"), namespace)
                replacement = namespace["build_forward"](
                    prior["original"], prior["finalizer"], eager_only=True, paired_down=adapter
                )
                serving = state["serving"] = dict(
                    previous=prior["replacement"], replacement=replacement, namespace=namespace
                )
            if ACTION in ("on", "verify", "off"):
                adapter.enabled = ACTION != "off"
                adapter.verify = ACTION == "verify"
                replacement = serving["previous"] if ACTION == "off" else serving["replacement"]
                W4SparseMoE._forward_grouped = replacement
                prior["replacement"] = replacement
                worker._qfn_activation_serving["replacement"] = replacement
        return dict(
            rank=worker.rank,
            passed=True,
            enabled=adapter.enabled,
            verify=adapter.verify,
            failed=adapter.failed,
            calls=adapter.calls,
            checked=adapter.checked,
            mismatches=adapter.mismatches,
            pair_calls=adapter.pair.calls,
            packer=adapter.packer.status(),
        )
    except Exception as error:
        import traceback

        return dict(rank=worker.rank, passed=False, error=repr(error), traceback=traceback.format_exc())


def admission(worker):
    return operation(worker, check_only=True)


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(admission)
        return dict(passed=len(self.rows) == 4 and all(r["passed"] for r in self.rows), workers=self.rows)

    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(rows) == 4 and all(r["passed"] for r in rows), workers=rows)


def prepare(engine):
    return Candidate()
"""

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("verify", "on", "off", "status"))
    p.add_argument("label")
    a = p.parse_args()
    assert Path(a.label).name == a.label
    client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", timeout=900)
    result = apply(client, "ACTION=" + repr(a.action) + "\n" + SOURCE, ROOT / a.label)
    receipt = result["applied"]["applied"]
    (ROOT / (a.label + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
