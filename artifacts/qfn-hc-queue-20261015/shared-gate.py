"""Gate shared projection operands against complete real-weight mix/combine."""


def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        from pathlib import Path

        import torch

        from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing

        state = worker._qfn_hc_epilogues
        path = Path("/srv/ai/storage/artifacts/qfn-hc-queue-20261015/hc_epilogues.py")
        namespace = {"__name__": "qfn_shared_hc_operand_private"}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        assert not hasattr(worker, "_qfn_shared_hc_operand")
        trial = {"namespace": namespace, "gate": None}
        worker._qfn_shared_hc_operand = trial
        candidate = namespace["make_mix"](state["owner"], state["original"], share_operand=True)
        retained = state["adapter"].candidate
        modules = [
            (name, module)
            for name, module in worker.model_runner.get_model().named_modules()
            if type(module).__name__ == "_GatedResidual" and module.use_combine
        ]
        selected = [modules[index] for index in (0, 1, -3, -1)]
        records = []
        with torch.no_grad():
            for name, module in selected:
                for rows in (515, 2048):
                    values = torch.randn(rows, 10240, dtype=torch.float16, device=worker.device)
                    block = torch.randn(rows, 2560, dtype=torch.float16, device=worker.device)
                    for step in range(3):
                        values.mul_(0.75)
                        actual, saved = candidate(module, values)
                        expected, reference = retained(module, values)
                        output = module.combine(block, saved)
                        expected_output = module.combine(block, reference)
                        passed = all(
                            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                            for a, b in ((actual, expected), (saved[1], reference[1].half()), (output, expected_output))
                        )
                        records.append(dict(module=name, rows=rows, step=step, passed=passed))
                        assert passed, records[-1]
                        del actual, saved, expected, reference, output, expected_output

                    def combined(mix, module=module, values=values, block=block):
                        mixed, residual = mix(module, values)
                        return mixed, module.combine(block, residual)

                    timing = paired_timing(
                        {
                            "retained": lambda combined=combined: combined(retained),
                            "shared": lambda combined=combined: combined(candidate),
                        },
                        5,
                        5,
                    )
                    records.append(dict(module=name, rows=rows, passed=True, timing=timing))
                    del values, block
        trial["gate"] = dict(passed=True, records=records)
        return dict(rank=worker.rank, passed=True, gate=trial["gate"])
    except Exception as error:
        import traceback

        result = dict(rank=worker.rank, passed=False, error=repr(error), traceback=traceback.format_exc())
        if hasattr(worker, "_qfn_shared_hc_operand"):
            worker._qfn_shared_hc_operand["gate"] = result
        return result


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
