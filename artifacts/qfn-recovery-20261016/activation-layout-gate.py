"""Compare exact W4 activation layouts without changing serving methods."""


def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import torch
        import torch.nn.functional as F

        from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing

        generator = torch.Generator().manual_seed(97203 + worker.rank)
        records = []
        with torch.no_grad():
            for rows in (30, 2590, 5150, 20480):
                source = (torch.randn(rows, 1280, generator=generator) * 1.5).half().to(worker.device)

                def baseline(source=source):
                    gate, up = source.float().chunk(2, -1)
                    return (F.silu(gate) * up).half()

                def split_first(source=source):
                    gate, up = source.chunk(2, -1)
                    return (F.silu(gate.float()) * up.float()).half()

                def inplace_gate(source=source):
                    gate, up = source.chunk(2, -1)
                    gate = F.silu(gate.float(), inplace=True)
                    return gate.mul_(up.float()).half()

                expected = baseline()
                functions = {"baseline": baseline}
                cases = []
                for name, fn in (("split_first", split_first), ("inplace_gate", inplace_gate)):
                    actual = fn()
                    mismatches = int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item())
                    cases.append(dict(name=name, mismatches=mismatches, passed=mismatches == 0))
                    if mismatches == 0:
                        functions[name] = fn
                    del actual
                del expected
                timings = paired_timing(functions, 5, 5)
                records.append(dict(rows=rows, cases=cases, timings=timings))
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
        return dict(passed=True, audit_passed=len(rows) == 4 and all(row["passed"] for row in rows), workers=rows)


def prepare(engine):
    return Candidate()
