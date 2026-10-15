"""Refresh the Python candidate to release its submitted RMS input promptly."""


def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import hashlib
        from pathlib import Path

        path = Path(
            "/srv/ai/storage/artifacts/ple-host-gather-20261015/serving-source-v2/tools/qwen4exp/hc_epilogues.py"
        )
        source = path.read_text()
        namespace = {"__name__": "qfn_hc_epilogues_final"}
        exec(compile(source, str(path), "exec"), namespace)
        state = worker._qfn_hc_epilogues
        adapter = state["adapter"]
        assert not adapter.owner.failed and not adapter.mismatches
        adapter.candidate = namespace["make_mix"](state["owner"], state["original"])
        state["final_namespace"] = namespace
        return dict(rank=worker.rank, passed=True, source_sha256=hashlib.sha256(source.encode()).hexdigest())
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=repr(error))


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        return dict(passed=True)

    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(rows) == 4 and all(row["passed"] for row in rows), workers=rows)


def prepare(engine):
    return Candidate()
