"""Narrow the qualified QSA adapter to large prefills; preserve short-batch dispatch."""


def operation(wrapper):
    from pathlib import Path

    worker = getattr(wrapper, "worker", wrapper)
    try:
        state = worker._qfn_qsa_table_probe
        owner = state["owner"]
        assert state["scope"]["qsa_sparse_attention_310"] is owner
        assert owner.checked == 546 and owner.enabled and not owner.failed and not owner.verify
        namespace = {}
        path = Path("/srv/ai/storage/artifacts/qfn-device-targets-20261015/qsa-table-v1/qsa_table_cache.py")
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        assert namespace["MIN_PREFILL_TOKENS"] == 128
        scope = type(owner).__call__.__globals__
        state["previous_admission"] = scope["tiling_values"]
        scope["tiling_values"] = namespace["tiling_values"]
        state["min_prefill_tokens"] = 128
        return dict(
            rank=worker.rank,
            passed=True,
            min_prefill_tokens=128,
            calls=owner.calls,
            prior_metadata=[list(values) for values in owner.metadata],
        )
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
