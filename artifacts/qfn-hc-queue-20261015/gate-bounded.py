"""Private chained-allocation gate for the existing non-draining owned bridge."""


def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import hashlib
        import json
        from pathlib import Path

        import torch

        from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing

        root = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/checkpoint-native-build-v1")
        manifest = json.loads((root / "provenance.json").read_text())
        assert manifest["flush_host_queue"] is False
        for filename in ("native_owned_bridge.cpp", "shared_launch_arguments.h"):
            assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == manifest["sources"][filename]
        bridge = manifest["bridge"]
        assert hashlib.sha256(Path(bridge["path"]).read_bytes()).hexdigest() == bridge["sha256"]
        torch.ops.load_library(bridge["path"])
        current = worker._qfn_hc_epilogues
        state = worker._qfn_hc_queue_candidate
        assert not state["owner"].failed
        owner = state["owner"]
        records = []
        with torch.no_grad():
            # Deliberately release inputs and consume outputs before a final sync.
            # Alternate allocations and shapes so callback ownership matters.
            for rows in (3, 259, 515, 1024, 2048):
                pending = []
                for iteration in range(12):
                    x = torch.full((rows, 10240), (iteration - 6) / 16, device=worker.device)
                    weight = torch.full((10240,), 1.25, device=worker.device)
                    full, narrow = owner.affine(x, weight)
                    expected_full = x * weight
                    pending.extend(((full, expected_full), (narrow, expected_full.half())))
                    del x, weight
                    gate = torch.sigmoid(narrow)
                    actual = owner.mean(full, gate)
                    expected = (full.view(rows, 4, 2560) * gate.view(rows, 4, 2560)).mean(1).half()
                    del full, narrow, gate
                    pending.append((actual + 1, expected + 1))
                    if iteration % 3 == 2:
                        torch.npu.synchronize()
                        passed = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in pending)
                        records.append(dict(rows=rows, wave=iteration // 3, cases=len(pending), passed=passed))
                        assert passed, records[-1]
                        pending.clear()
                torch.npu.synchronize()
                passed = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in pending)
                records.append(dict(rows=rows, cases=len(pending), passed=passed))
                assert passed, records[-1]
                del pending
            for rows in (515, 2048):
                x = torch.randn(rows, 10240, device=worker.device)
                gate = torch.sigmoid(x).half()
                timings = paired_timing(
                    {
                        "draining": lambda x=x, gate=gate: current["owner"].mean(x, gate),
                        "queued": lambda x=x, gate=gate: owner.mean(x, gate),
                    },
                    20,
                    5,
                )
                records.append(dict(rows=rows, timing=timings, passed=True))
        state["gate"] = dict(passed=True, records=records)
        return dict(rank=worker.rank, passed=True, gate=state["gate"])
    except Exception as error:
        import traceback

        result = dict(rank=worker.rank, passed=False, error=repr(error), traceback=traceback.format_exc())
        worker._qfn_hc_queue_candidate["gate"] = result
        return result


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        return dict(passed=True)

    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=True, candidate_passed=len(rows) == 4 and all(row["passed"] for row in rows), workers=rows)


def prepare(engine):
    return Candidate()
