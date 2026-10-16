"""Private M16 schedule experiment against the retained real-weight consumer."""


def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import hashlib
        from pathlib import Path

        import torch

        from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing
        from tools.qwen4exp.native_prefill import NativeGroupedPrefill

        binary = Path("/srv/ai/storage/artifacts/qfn-local-pack-20261016/schedule-m16.bin")
        assert not hasattr(worker, "_qfn_m16_private")
        state = worker._qfn_m16_private = dict(retained=[])
        kernel = torch.classes.qwen_route_finalize_v3.Kernel(str(binary), "qwen_prefill_local_count_m16_v1")
        candidate = NativeGroupedPrefill(kernel, torch.ops.qwen_route_finalize_v3.launch)
        state["candidate"] = candidate
        reference = worker._qfn_local_pack_test["consumer_resource"]
        modules = [m for m in worker.model_runner.get_model().modules() if type(m).__name__ == "W4SparseMoE"]
        generator = torch.Generator().manual_seed(160172 + worker.rank)
        records = []
        with torch.no_grad():
            for name in ("gate_up_proj", "down_proj"):
                bank = modules[0].projections[name]
                for rows in (650, 5150, 20480):
                    source = (
                        (torch.randn(rows, bank.weight.shape[-1] * 2, generator=generator) * 1.5)
                        .half()
                        .to(worker.device)
                    )
                    packed = torch.ops._C_ascend.npu_qwen_w4_a8_pack_310(source)
                    state["retained"] = [source, packed]
                    for active, skew in ((0, False), (rows // 4, False), (rows // 4, True), (rows, False)):
                        counts = [active // 128 + (i < active % 128) for i in range(128)]
                        if skew:
                            # Half the routes in one expert, remaining routes over 127.
                            counts = [active // 2] + [0] * 127
                            for i in range(active - counts[0]):
                                counts[1 + i % 127] += 1
                        ends = torch.tensor(counts, device=worker.device, dtype=torch.int64).cumsum(0)
                        expected = reference(bank, packed, ends)
                        actual = candidate(bank, packed, ends)
                        exact = torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                        record = dict(projection=name, rows=rows, local_rows=active, skew=skew, exact=exact)
                        records.append(record)
                        if not exact:
                            state["receipt"] = dict(passed=False, records=records)
                            return dict(rank=worker.rank, **state["receipt"])
                        if active == rows // 4:
                            record["timing"] = paired_timing(
                                {
                                    "m32": lambda bank=bank, packed=packed, ends=ends: reference(bank, packed, ends),
                                    "m16": lambda bank=bank, packed=packed, ends=ends: candidate(bank, packed, ends),
                                },
                                2,
                                3,
                            )
                        del expected, actual
                    torch.npu.synchronize()
                    state["retained"] = []
        state["receipt"] = dict(
            passed=True, records=records, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()
        )
        return dict(rank=worker.rank, **state["receipt"])
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
        return dict(passed=True, candidate_passed=len(rows) == 4 and all(r["passed"] for r in rows), workers=rows)


def prepare(engine):
    return Candidate()
