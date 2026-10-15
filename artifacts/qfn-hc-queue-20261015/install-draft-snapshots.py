"""Refresh the drained queued bridge without replacing its retained owner."""


def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import hashlib
        from pathlib import Path

        resource = wrapper._resident_session().native_resources["qfn_mamba_commit_v4"].queued_schedule_v1
        assert resource.enabled and resource.device_drafts
        assert resource.expected_device_inputs is None and resource.expected_device_history is None
        assert resource.runner.execute_model_state is None and not resource.runner.use_async_scheduling
        path = Path("/srv/ai/storage/artifacts/qfn-hc-queue-20261015/queued_schedule.py")
        namespace = {"__name__": "qfn_queued_draft_snapshots"}
        source = path.read_text()
        exec(compile(source, str(path), "exec"), namespace)
        assert not hasattr(worker, "_qfn_draft_snapshots")
        worker._qfn_draft_snapshots = dict(namespace=namespace, original_class=type(resource), resource=resource)
        resource.draft_snapshots = {}
        resource.draft_cache_hits = 0
        resource.__class__ = namespace["QueuedScheduleBridge"]
        return dict(
            rank=worker.rank,
            passed=True,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            status=resource.status(),
        )
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
