"""Private exactness and timing gate; never replace a serving activation here."""


def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import hashlib
        from pathlib import Path

        import torch
        import torch.nn.functional as F

        from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing

        root = Path("/srv/ai/storage/artifacts/qfn-pack-followup-20261016")
        binary = root / "native_activation_pack.bin"
        assert not hasattr(worker, "_qfn_activation_pack_test")
        state = dict(configs={}, retained=[])
        worker._qfn_activation_pack_test = state
        state["kernel"] = torch.classes.qwen_hc_combine_v1.Kernel(str(binary), "qwen_activation_pack_v1")
        generator = torch.Generator().manual_seed(34891 + worker.rank)
        records = []
        with torch.no_grad():
            for rows in (1, 3, 30, 2590, 5150, 20480):
                config = torch.tensor([rows], device=worker.device, dtype=torch.int64)
                state["configs"][rows] = config
                for pattern, scale in (("random", 1.5), ("wide", 64), ("exceptional", 1.5)):
                    host = (torch.randn(rows, 1280, generator=generator) * scale).half()
                    if pattern == "exceptional":
                        host[:, :8] = torch.tensor(
                            [0.0, -0.0, 2**-24, -(2**-24), 65504, float("inf"), -float("inf"), float("nan")]
                        ).half()
                    source = host.to(worker.device)

                    def baseline(source=source):
                        activation = worker._qfn_activation_serving["owner"](source, torch.float32, torch.float16)
                        return torch.ops._C_ascend.npu_qwen_w4_a8_pack_310(activation)

                    def candidate(source=source, config=config, rows=rows):
                        activated = F.silu(source[:, :640].float())
                        assert activated.is_contiguous() and source.is_contiguous()
                        output = (
                            torch.empty((rows, 320), dtype=torch.int8, device=worker.device),
                            torch.empty((rows, 320), dtype=torch.int8, device=worker.device),
                            torch.empty((rows, 5, 8), dtype=torch.float32, device=worker.device),
                            torch.empty((rows, 5, 8), dtype=torch.float32, device=worker.device),
                        )
                        state["retained"] = [activated, source, *output, config]
                        torch.ops.qwen_hc_combine_v1.launch(state["kernel"], state["retained"], 8)
                        return output

                    expected, actual = baseline(), candidate()
                    mismatches = [
                        int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item()) for a, b in zip(actual, expected)
                    ]
                    records.append(dict(rows=rows, pattern=pattern, mismatches=mismatches, passed=not any(mismatches)))
                    del expected, actual
                    if any(mismatches):
                        state["receipt"] = dict(passed=False, records=records)
                        return dict(rank=worker.rank, **state["receipt"])
                # Time finite values; exceptional inputs qualify semantics separately.
                source = (torch.randn(rows, 1280, generator=generator) * 1.5).half().to(worker.device)
                # Rebind defaults after replacing the source used by the parity cases.
                timings = paired_timing(
                    {
                        "baseline": lambda source=source: baseline(source),
                        "fused_pack": lambda source=source: candidate(source),
                    },
                    5,
                    5,
                )
                records.append(dict(rows=rows, timings=timings))
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
        return dict(passed=True, candidate_passed=len(rows) == 4 and all(row["passed"] for row in rows), workers=rows)


def prepare(engine):
    return Candidate()
