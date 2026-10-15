def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        state = worker._qfn_qsa_table_probe
        assert not state["owner"].enabled
        assert state["scope"]["qsa_sparse_attention_310"] is state["layout_wrapper"]
        state["scope"]["qsa_sparse_attention_310"] = state["owner"]
        return dict(rank=worker.rank, passed=True, layouts=list(state["layouts"].values()))
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
