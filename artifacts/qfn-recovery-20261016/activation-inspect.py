"""Read the live finalizer binding before installing activation composition."""


def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import inspect

        from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE

        fn = W4SparseMoE._forward_grouped
        closure = inspect.getclosurevars(fn).nonlocals
        return dict(
            rank=worker.rank,
            passed=True,
            closure={k: type(v).__name__ for k, v in closure.items()},
            function=fn.__qualname__,
            attributes=list(vars(fn)),
            resource_names=list(wrapper._resident_session().native_resources),
            activation_state=list(getattr(worker, "_qfn_activation_serving", {})),
        )
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=repr(error))


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        return dict(passed=True)

    def apply(self, engine):
        return dict(passed=True, workers=engine.model_executor.collective_rpc(operation))


def prepare(engine):
    return Candidate()
