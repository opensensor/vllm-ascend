def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        state = worker._qfn_qsa_table_probe
        assert not state["owner"].enabled
        state["layouts"] = {}

        def measured(q, k, v, selection, table, bounds, **kwargs):
            tensors = (
                q,
                k,
                v,
                selection.group_indices,
                selection.group_counts,
                selection.tail_starts,
                selection.tail_counts,
                table,
                bounds,
            )
            details = tuple(
                (tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device), t.is_contiguous()) for t in tensors
            )
            if len(state["layouts"]) < 16:
                state["layouts"][str(details)] = dict(tensors=details, kwargs=kwargs)
            return state["original"](q, k, v, selection, table, bounds, **kwargs)

        state["layout_wrapper"] = measured
        state["scope"]["qsa_sparse_attention_310"] = measured
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
