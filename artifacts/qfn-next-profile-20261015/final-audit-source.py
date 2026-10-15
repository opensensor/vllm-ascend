def inspect_retained(worker):
    commit = worker._resident_session().native_resources["qfn_mamba_commit_v4"]
    names = (
        "device_transition_v1",
        "ple_history_v2",
        "output_observer_v1",
        "queued_schedule_v1",
        "compact_table_staging_v1",
        "decode_preparation_v1",
        "host_commit_metadata_v1",
        "checkpoint_preparation_v1",
        "device_commit_inputs_v1",
    )
    owners = {name: getattr(commit, name).status() for name in names}
    owners["gdn_metadata_graphs_v1"] = [owner.status() for owner in commit.gdn_metadata_graphs_v1]
    for name in names:
        if not owners[name].get("enabled") or owners[name].get("failed", False):
            raise RuntimeError("inactive or failed retained optimization: " + name)
    assert commit.enabled and not commit.verify
    assert all(owner["enabled"] and not owner.get("failed", False) for owner in owners["gdn_metadata_graphs_v1"])
    assert owners["checkpoint_preparation_v1"]["backend"] == "fused"
    assert owners["queued_schedule_v1"]["device_drafts"]
    sizes = list(worker.model_runner.compilation_config.cudagraph_capture_sizes)
    assert sorted(sizes) == [3, 6, 9, 12]
    import hashlib
    import inspect

    from tools.qwen4exp import iteration_transition_runtime

    fn = iteration_transition_runtime.admit_all_acceptances
    source = inspect.getsource(fn)
    assert "np.arange(1, 4" in source and "np.nonzero(copy)" in source
    assert worker.model_runner._qfn_acceptance_reuse
    assert owners["ple_history_v2"]["early_readback"]
    assert owners["checkpoint_preparation_v1"]["lazy_readback"]
    return dict(
        rank=worker.rank,
        passed=True,
        owners=owners,
        capture_sizes=sizes,
        acceptance_reuse=True,
        consolidated_admission=dict(
            file=inspect.getsourcefile(fn),
            sha256=hashlib.sha256(source.encode()).hexdigest(),
            names=list(fn.__code__.co_names),
        ),
    )


class Candidate:
    def validate(self, engine):
        assert (
            engine._pause_complete() and not engine.batch_queue and not engine.scheduler.get_num_unfinished_requests()
        )
        assert engine._qfn_mixed_budget.max_prefill_tokens == 512
        assert engine.scheduler.schedule is engine._qfn_mixed_budget.wrapper
        assert engine.scheduler.scheduler_config.long_prefill_token_threshold == 0
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(inspect_worker)
        return dict(passed=len(self.rows) == 4 and all(row["passed"] for row in self.rows), workers=self.rows)

    def apply(self, engine):
        self.validate(engine)
        rows = self.rows
        return dict(
            passed=len(rows) == 4 and all(r["passed"] for r in rows),
            workers=rows,
            engine_scheduler=type(engine.scheduler).__name__,
            async_scheduling=engine.async_scheduling,
            budget=engine._qfn_mixed_budget.status(),
        )


def prepare(engine):
    return Candidate()


def inspect_worker(worker):
    wrapper = worker
    worker = getattr(worker, "worker", worker)
    try:
        import inspect

        result = inspect_retained(wrapper)
        score = worker._qfn_mixed_qsa
        assert score["mode"] == "score" and not score["failed"]
        assert all(inspect.getattr_static(owner, name) is value for owner, name, value in score["replacements"])
        ple = wrapper._qfn_native_ple_serving["manager"].status()
        assert ple["enabled"] and not ple["verify"] and not ple["failed"] and ple["checked"] > 0
        result["score_mode"] = score["mode"]
        result["score_gate"] = worker._qfn_mixed_qsa_gate
        result["native_ple"] = ple
        merge = worker._qfn_mixed_gdn_merge
        from vllm_ascend.models.qwen4_exp.model import _GDNAttention

        assert _GDNAttention._native_mixed_attention is merge.replacement
        assert merge.enabled and not merge.verify and not merge.failed and merge.checked >= 36
        assert getattr(worker, "profiler", None) is None
        result["mixed_gdn_merge"] = merge.status()
        return result
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=str(error))
