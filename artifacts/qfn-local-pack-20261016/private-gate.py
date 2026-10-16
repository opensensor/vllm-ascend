"""Poison unused pack rows and verify the complete real-weight consumer."""


def operation(wrapper):
    """Check poisoned inactive storage through two real projection consumers."""
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import hashlib
        import inspect
        from pathlib import Path

        import torch
        import torch.nn.functional as F

        from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing
        from vllm_ascend.models.qwen4_exp.w4_moe import PackedExpertBank

        root = Path("/srv/ai/storage/artifacts/qfn-local-pack-20261016")
        binary = root / "native_activation_pack_local.bin"
        assert not hasattr(worker, "_qfn_local_pack_test")
        state = dict(configs={}, retained=[])
        worker._qfn_local_pack_test = state
        state["kernel"] = torch.classes.qwen_hc_combine_v1.Kernel(str(binary), "qwen_local_activation_pack_v1")
        state["consumer"] = PackedExpertBank.native_linear
        assert hasattr(state["consumer"], "_qfn_prefill_schedule_base")
        state["consumer_resource"] = inspect.getclosurevars(state["consumer"]).nonlocals["resource"]
        full_kernel = worker._qfn_pack_serving["owner"].kernel
        banks = [
            (n, m.projections["down_proj"])
            for n, m in worker.model_runner.get_model().named_modules()
            if type(m).__name__ == "W4SparseMoE"
        ]
        generator = torch.Generator().manual_seed(160135 + worker.rank)
        records = []
        with torch.no_grad():
            for rows in (2590, 5150, 20480):
                config = torch.tensor([rows], device=worker.device, dtype=torch.int64)
                state["configs"][rows] = config
                source = (torch.randn(rows, 1280, generator=generator) * 1.5).half().to(worker.device)
                for active in (0, 1, rows // 4, rows - 1, rows):
                    ends = torch.tensor([(i + 1) * active // 128 for i in range(128)], device=worker.device)

                    def packed(local, poison=False, source=source, rows=rows, ends=ends, config=config):
                        activated = F.silu(source[:, :640].float())
                        output = (
                            torch.empty((rows, 320), dtype=torch.int8, device=worker.device),
                            torch.empty((rows, 320), dtype=torch.int8, device=worker.device),
                            torch.empty((rows, 5, 8), dtype=torch.float32, device=worker.device),
                            torch.empty((rows, 5, 8), dtype=torch.float32, device=worker.device),
                        )
                        if poison:
                            for t in output[:2]:
                                t.fill_(101)
                            for t in output[2:]:
                                t.fill_(float("nan"))
                        args = [activated, source, *output]
                        if local:
                            args.append(ends)
                        args.append(config)
                        state["retained"] = args
                        torch.ops.qwen_hc_combine_v1.launch(state["kernel"] if local else full_kernel, args, 8)
                        return output

                    expected, actual = packed(False), packed(True, True)
                    prefix_exact = all(
                        torch.equal(a[:active].view(torch.uint8), b[:active].view(torch.uint8))
                        for a, b in zip(actual, expected)
                    )
                    tail_unwritten = all(torch.all(t[active:] == 101).item() for t in actual[:2])
                    tail_unwritten = tail_unwritten and all(torch.isnan(t[active:]).all().item() for t in actual[2:])
                    consumers = []
                    for name, bank in (banks[0], banks[-1]):
                        output = bank.native_linear(actual, ends)
                        reference = bank.native_linear(expected, ends)
                        exact = torch.equal(output.view(torch.uint8), reference.view(torch.uint8))
                        zero_tail = bool((output[active:] == 0).all().item())
                        consumers.append(dict(name=name, exact=exact, zero_tail=zero_tail))
                        del output, reference
                    row = dict(
                        rows=rows,
                        active=active,
                        prefix_exact=prefix_exact,
                        tail_unwritten=tail_unwritten,
                        consumers=consumers,
                    )
                    row["passed"] = (
                        prefix_exact and tail_unwritten and all(x["exact"] and x["zero_tail"] for x in consumers)
                    )
                    records.append(row)
                    if not row["passed"]:
                        state["receipt"] = dict(passed=False, records=records)
                        return dict(rank=worker.rank, **state["receipt"])
                    del expected, actual
                    if active == rows // 4:
                        timings = paired_timing({"full": lambda: packed(False), "local": lambda: packed(True)}, 5, 5)
                        records.append(dict(rows=rows, active=active, timings=timings))
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
