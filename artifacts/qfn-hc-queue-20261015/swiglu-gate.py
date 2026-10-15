"""Check FP32 built-in SwiGLU against the pinned W4 activation expression."""


def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import torch
        import torch.nn.functional as F
        import torch_npu

        from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing

        records = []
        with torch.no_grad():
            for rows in (30, 2590, 5150, 20480):
                for scale in (0.5, 1.5, 4.0):
                    source = (torch.randn(rows, 1280, device=worker.device) * scale).half()

                    def baseline(source=source):
                        gate, up = source.float().chunk(2, -1)
                        return (F.silu(gate) * up).half()

                    def candidate(source=source):
                        return torch_npu.npu_swiglu(source.float(), dim=-1).half()

                    actual, expected = candidate(), baseline()
                    mismatches = int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item())
                    records.append(
                        dict(
                            rows=rows,
                            scale=scale,
                            passed=mismatches == 0,
                            mismatches=mismatches,
                            max_abs=float((actual.float() - expected.float()).abs().max().item()),
                        )
                    )
                    del actual, expected
                    if mismatches:
                        return dict(rank=worker.rank, passed=False, records=records)
                timing = paired_timing({"baseline": baseline, "fused_fp32": candidate}, 5, 5)
                records.append(dict(rows=rows, passed=True, timing=timing))
        return dict(rank=worker.rank, passed=True, records=records)
    except Exception as error:
        import traceback

        return dict(rank=worker.rank, passed=False, error=repr(error), traceback=traceback.format_exc())


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        return dict(passed=True)

    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=True, candidate_passed=len(rows) == 4 and all(row["passed"] for row in rows), workers=rows)


def prepare(engine):
    return Candidate()
