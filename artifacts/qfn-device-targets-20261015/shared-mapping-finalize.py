def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        state = worker._qfn_shared_ple_probe
        assert state["installed"] and state["shadow"]["checked"] == 96 and not state["shadow"]["failed"]
        manager = wrapper._qfn_native_ple_serving["manager"]
        assert not manager.failed and not manager.busy
        assert wrapper._qfn_native_ple_serving["namespace"]["Lookup"] is state["factory"]
        assert all(manager.lookups[key] is owner for key, owner in state["owners"].items())
        for owner in state["owners"].values():
            owner.fetch = owner._shared_original_fetch
            del owner._shared_original_fetch
        state["qualified"] = True
        return dict(rank=worker.rank, passed=True, shadow=state["shadow"], owners=len(state["owners"]))
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=repr(error))


class Candidate:
    def validate(self, engine):
        assert (
            engine._pause_complete() and not engine.batch_queue and not engine.scheduler.get_num_unfinished_requests()
        )
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=all(row["passed"] for row in self.rows), workers=self.rows)

    def apply(self, engine):
        return self.validate(engine)


def prepare(engine):
    return Candidate()
