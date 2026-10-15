def operation(wrapper):
    from vllm_ascend.models.qwen4_exp.model import _QSAAttention

    worker = getattr(wrapper, "worker", wrapper)
    try:
        fn = _QSAAttention.forward
        scope = fn.__globals__
        original = scope["qsa_sparse_attention_310"]
        assert not hasattr(worker, "_qfn_qsa_geometry")
        state = dict(original=original, scope=scope, records={}, calls=0)

        def measured(query, key, value, selection, table, bounds, **kwargs):
            key_ = (tuple(query.shape), tuple(key.shape), tuple(table.shape), tuple(bounds.shape), str(query.dtype))
            state["calls"] += 1
            if key_ in state["records"] or len(state["records"]) < 64:
                state["records"][key_] = state["records"].get(key_, 0) + 1
            return original(query, key, value, selection, table, bounds, **kwargs)

        state["replacement"] = measured
        worker._qfn_qsa_geometry = state
        scope["qsa_sparse_attention_310"] = measured
        return dict(passed=True, rank=worker.rank, original=str(original), forward=str(fn))
    except Exception as error:
        return dict(passed=False, rank=worker.rank, error=repr(error))


class Candidate:
    def validate(self, engine):
        assert (
            engine._pause_complete() and not engine.batch_queue and not engine.scheduler.get_num_unfinished_requests()
        )
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=all(r["passed"] for r in self.rows), workers=self.rows)

    def apply(self, engine):
        return self.validate(engine)


def prepare(engine):
    return Candidate()
