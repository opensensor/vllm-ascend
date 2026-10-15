def operation(wrapper):
    from pathlib import Path

    worker = getattr(wrapper, "worker", wrapper)
    try:
        state = worker._qfn_qsa_table_probe
        old = state["owner"]
        assert state["scope"]["qsa_sparse_attention_310"] is old and not old.enabled
        ns = {}
        path = Path("/srv/ai/storage/artifacts/qfn-device-targets-20261015/qsa-table-v1/qsa_table_cache.py")
        exec(compile(path.read_text(), str(path), "exec"), ns)
        owner = ns["QSATableCache"](old.kernel, old.original, old.device)
        owner.verify = True
        state["old_adapter"] = old
        state["owner"] = owner
        state["scope"]["qsa_sparse_attention_310"] = owner
        return dict(rank=worker.rank, passed=True)
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=repr(error))


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=all(row["passed"] for row in self.rows), workers=self.rows)

    def apply(self, engine):
        return self.validate(engine)


def prepare(engine):
    return Candidate()
