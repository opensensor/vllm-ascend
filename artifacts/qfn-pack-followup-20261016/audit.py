import hashlib
import inspect
from pathlib import Path


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
        return dict(passed=True, workers=self.rows)

    def apply(self, engine):
        self.validate(engine)
        rows = self.rows
        return dict(
            passed=True,
            audit_passed=len(rows) == 4 and all(r["passed"] for r in rows),
            workers=rows,
            engine_scheduler=type(engine.scheduler).__name__,
            async_scheduling=engine.async_scheduling,
            budget=engine._qfn_mixed_budget.status(),
            budget_source_sha256=hashlib.sha256(
                Path(inspect.getsourcefile(type(engine._qfn_mixed_budget))).read_bytes()
            ).hexdigest(),
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
        assert getattr(worker, "_qfn_mixed_profile", None) is None
        result["mixed_gdn_merge"] = merge.status()
        import importlib

        module = importlib.import_module("vllm_ascend._310p.ops.fla.chunk_gated_delta_rule")
        output = worker._qfn_gdn_output_adapter
        assert output.enabled and output.receipt["passed"] and output.calls > 0 and output.fallbacks == 0
        assert module.chunk_gated_delta_rule_310 is output.replacement
        wy = getattr(worker, "_qfn_wy_graph", None)
        assert wy is None or (not wy.enabled and not wy.failed)
        result["gdn_output"] = output.status()
        result["wy_graph"] = dict(installed=False) if wy is None else wy.status()
        manager = wrapper._qfn_native_ple_serving["manager"]
        assert manager.mapping_parent is not None and manager.mapping_parent in manager.retained
        assert manager.status()["shared_mapping"]
        assert hasattr(manager.lib, "ple_host_create_shared")
        result["shared_mapping"] = dict(
            qualified=True, owners=len(manager.lookups), scope="fresh shared-mapping adapter; no migration pool"
        )
        qsa = worker._qfn_qsa_table_probe
        assert qsa["scope"]["qsa_sparse_attention_310"] is qsa["owner"]
        assert qsa["owner"].enabled and not qsa["owner"].verify and not qsa["owner"].failed
        assert qsa["owner"].checked >= 52 and qsa["owner"].calls > 0
        assert qsa["namespace"]["MIN_PREFILL_TOKENS"] == 128
        result["qsa_min_prefill_tokens"] = 128
        assert not hasattr(worker, "_qfn_qsa_geometry")
        result["qsa_table"] = qsa["owner"].status()
        from vllm_ascend.models.qwen4_exp.model import _GatedResidual

        hc = worker._qfn_hc_epilogues["adapter"]
        assert _GatedResidual.mix is hc.replacement
        assert hc.enabled and not hc.verify and not hc.owner.failed
        assert hc.checked >= 48 and hc.mismatches == 0
        result["hc_epilogues"] = hc.status()
        from collections import Counter

        layers = [m for m in worker.model_runner.get_model().modules() if type(m).__name__ == "W4SparseMoE"]
        result["w4_activation"] = dict(Counter(getattr(m, "grouped_activation", "absent") for m in layers))
        result["w4_finalize"] = dict(Counter(getattr(m, "grouped_finalize", "absent") for m in layers))
        result["w4_classes"] = len(layers)
        result["draft_source_sha256"] = hashlib.sha256(
            Path(
                inspect.getsourcefile(
                    type(wrapper._resident_session().native_resources["qfn_mamba_commit_v4"].queued_schedule_v1)
                )
            ).read_bytes()
        ).hexdigest()
        result["draft_cache"] = (
            wrapper._resident_session().native_resources["qfn_mamba_commit_v4"].queued_schedule_v1.status()
        )
        activation = getattr(worker, "_qfn_activation_serving", None)
        if activation is not None:
            from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE

            owner = activation["owner"]
            assert W4SparseMoE._forward_grouped is activation["replacement"]
            assert inspect.getclosurevars(activation["replacement"]).nonlocals["resource"] is activation["finalizer"]
            assert owner.enabled and not owner.verify and not owner.failed and owner.mismatches == 0
            assert owner.checked >= 48 and owner.calls > 0
            pack = getattr(worker, "_qfn_pack_serving", None)
            if pack is not None:
                packed_owner = pack["owner"]
                assert packed_owner.enabled and not packed_owner.verify
                assert not packed_owner.failed and packed_owner.mismatches == 0
                assert packed_owner.checked >= 48 and packed_owner.calls > 0
                assert packed_owner.activation is owner
                assert pack["finalizer"] is activation["finalizer"]
                assert inspect.getclosurevars(pack["replacement"]).nonlocals["resource"] is pack["finalizer"]
                result["activation_pack"] = packed_owner.status()
            result["activation_epilogue"] = owner.status()
            result["activation_source_sha256"] = hashlib.sha256(
                Path(inspect.getsourcefile(type(owner))).read_bytes()
            ).hexdigest()
        return result
    except Exception as error:
        import traceback

        return dict(rank=worker.rank, passed=False, error=str(error), traceback=traceback.format_exc())
