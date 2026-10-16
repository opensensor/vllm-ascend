"""Attach the privately qualified local-route kernel, preserving the fallback."""


def operation(wrapper, check_only=False):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import importlib
        import inspect

        import torch

        from tools.qwen4exp import w4_activation_pack
        from tools.qwen4exp.resident_candidates import qfn_route_finalize
        from vllm_ascend.models.qwen4_exp.w4_moe import PackedExpertBank, W4SparseMoE

        prior = worker._qfn_pack_serving
        assert W4SparseMoE._forward_grouped is prior["replacement"]
        assert not prior["owner"].failed and prior["owner"].mismatches == 0
        private = worker._qfn_local_pack_test
        assert private["receipt"]["passed"]
        assert PackedExpertBank.native_linear is private["consumer"]
        assert worker.model_runner.execute_model_state is None
        if not check_only:
            importlib.reload(w4_activation_pack)
            assert w4_activation_pack.MIN_ROWS == 2560
            fallback = worker._qfn_activation_serving
            consumer = private["consumer"]
            binding = inspect.getclosurevars(consumer).nonlocals
            guard = w4_activation_pack.local_projection_guard(
                consumer, private["consumer_resource"], binding["native_backend"]
            )
            owner = w4_activation_pack.W4ActivationPack(
                prior["owner"].kernel,
                torch.ops.qwen_hc_combine_v1.launch,
                fallback["owner"],
                torch.ops._C_ascend.npu_qwen_w4_a8_pack_310,
                local_kernel=private["kernel"],
                local_projection=guard,
            )
            closure = inspect.getclosurevars(prior["replacement"]).nonlocals
            replacement = qfn_route_finalize.build_forward(
                prior["original"],
                prior["finalizer"],
                eager_only=closure["eager_only"],
                packed_down=owner,
            )
            state = dict(
                owner=owner,
                original=prior["original"],
                finalizer=prior["finalizer"],
                replacement=replacement,
                previous=prior,
            )
            worker._qfn_pack_serving = state
            fallback["replacement"] = replacement
            W4SparseMoE._forward_grouped = replacement
        return dict(rank=worker.rank, passed=True, status=worker._qfn_pack_serving["owner"].status())
    except Exception as error:
        import traceback

        return dict(rank=worker.rank, passed=False, error=repr(error), traceback=traceback.format_exc())


def admission(worker):
    return operation(worker, check_only=True)


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(admission)
        return dict(passed=len(self.rows) == 4 and all(r["passed"] for r in self.rows), workers=self.rows)

    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(rows) == 4 and all(r["passed"] for r in rows), workers=rows)


def prepare(engine):
    return Candidate()
