"""Inspect and clear only the identified failed read-only audit transaction."""

import gc
import json
import sys
from pathlib import Path


def run():
    from vllm.v1.engine.core import EngineCore

    engines = [x for x in gc.get_objects() if isinstance(x, EngineCore)]
    for frame in sys._current_frames().values():
        while frame is not None:
            obj = frame.f_locals.get("self")
            if isinstance(obj, EngineCore) and all(obj is not e for e in engines):
                engines.append(obj)
            frame = frame.f_back
    assert len(engines) == 1
    engine = engines[0]
    assert engine._pause_complete() and not engine.batch_queue
    assert not engine.scheduler.get_num_unfinished_requests()
    session = engine._qfn_engine_control_session
    assert session.failed and session.pending is not None
    assert session.pending[:2] == (
        "native_ple_a08de093194f4e3d8f8fca0a8c617b48",
        "c115afd10eeec5f0b0fcde7febed6baac07c99f3d01c5a900d85f47d882c9b65",
    )
    root = Path("/srv/ai/storage/artifacts/qfn-hc-queue-20261015")

    def inspect_private(wrapper):
        worker = getattr(wrapper, "worker", wrapper)
        import torch

        from vllm_ascend.models.qwen4_exp.model import _GatedResidual

        state = getattr(worker, "_qfn_hc_queue_candidate", None)
        active = worker._qfn_hc_epilogues
        owner = None if state is None else state.get("owner")
        result = dict(
            rank=worker.rank,
            allocated=state is not None,
            keys=[] if state is None else list(state),
            kernels=0 if state is None else len(state["kernels"]),
            owner_failed=None if owner is None else owner.failed,
            failure=None if owner is None else repr(owner.failure),
            configs=[] if owner is None else [str(k) for k in owner.configs],
            gate=None if state is None else state.get("gate"),
        )
        assert _GatedResidual.mix is active["adapter"].replacement
        assert active["adapter"].owner is active["owner"] and active["adapter"].enabled
        assert owner is not active["owner"]
        torch.npu.synchronize()
        return result

    result = dict(recovered=False, private=engine.model_executor.collective_rpc(inspect_private))
    (root / "recovery.json").write_text(json.dumps(result, indent=2))
    namespace = {}
    exec(
        compile(
            Path("/srv/ai/storage/artifacts/qfn-hc-epilogues-20261015/final-audit-source.py").read_text(),
            "<corrected-readonly-audit>",
            "exec",
        ),
        namespace,
    )
    corrected = namespace["prepare"](engine).apply(engine)
    result["corrected_audit"] = corrected
    (root / "recovery.json").write_text(json.dumps(result, indent=2))
    assert corrected["passed"]
    # Private allocations never replaced the serving owner. Keep all resources.
    session.pending = None
    session.failed = False
    result.update(recovered=True, status=session.status(engine))
    (root / "recovery.json").write_text(json.dumps(result, indent=2))


run()
