def operation(wrapper):
    import hashlib
    import json
    import time
    from pathlib import Path

    import torch

    worker = getattr(wrapper, "worker", wrapper)
    result = dict(rank=worker.rank, passed=False)
    root = Path("/srv/ai/storage/artifacts/qfn-device-targets-20261015/qsa-table-v1")
    try:
        from vllm_ascend.models.qwen4_exp.model import _QSAAttention
        from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection

        assert not hasattr(worker, "_qfn_qsa_table_probe")
        scope = _QSAAttention.forward.__globals__
        original = scope["qsa_sparse_attention_310"]
        ns = {}
        exec(compile((root / "qsa_table_cache.py").read_text(), str(root / "qsa_table_cache.py"), "exec"), ns)
        binary = root / "native-qsa-table.bin"
        kernel = torch.classes.qwen_checkpoint_preparation_v1.Kernel(str(binary), "qwen_qsa_table_v1")
        device = wrapper.model_runner.device
        owner = ns["QSATableCache"](kernel, original, device)
        worker._qfn_qsa_table_probe = dict(owner=owner, scope=scope, original=original, namespace=ns, receipt=result)
        generator = torch.Generator().manual_seed(702 + worker.rank)
        key = (torch.randn(512, 16, 64, 16, generator=generator, dtype=torch.float16) * 0.1).to(device)
        value = (torch.randn(512, 16, 64, 16, generator=generator, dtype=torch.float16) * 0.1).to(device)
        records = []
        for width, boundaries, mode in (
            (4096, [0, 3, 515], "sparse"),
            (4096, [0, 512, 515], "sparse"),
            (4096, [0, 13, 30, 230, 515], "sparse"),
            (4096, [0, 515], "sparse"),
            (2048, [0, 3, 515], "sparse"),
            (4096, [0, 3, 515], "dense"),
            (4096, [0, 13], "edge"),
        ):
            tokens = boundaries[-1]
            q = (torch.randn(tokens, 6, 256, generator=generator, dtype=torch.float16) * 0.1).to(device)
            table = torch.randint(0, 512, (len(boundaries) - 1, width), generator=generator, dtype=torch.int32).to(
                device
            )
            bounds = torch.tensor(boundaries, dtype=torch.int32, device=device)
            groups = torch.randint(0, 8191, (tokens, 512), generator=generator, dtype=torch.int32).to(device)
            counts = torch.full((tokens,), 512, dtype=torch.int32, device=device)
            starts = torch.full_like(counts, 32764)
            tails = torch.full_like(counts, 3)
            if mode == "dense":
                counts.fill_(-512)
            if mode == "edge":
                counts.copy_(
                    torch.tensor([0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 511, 512, 4], dtype=torch.int32, device=device)
                )
                tails.copy_(torch.tensor([0, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device=device))
            selection = QSAGroupSelection(groups, counts, starts, tails)
            args = (q, key, value, selection, table, bounds)
            expected = original(*args)
            before_calls = owner.calls
            actual = owner(*args)
            admitted = tokens >= ns.get("MIN_PREFILL_TOKENS", 13)
            assert owner.calls - before_calls == int(admitted)
            exact = torch.equal(expected.view(torch.int16), actual.view(torch.int16))
            record = dict(width=width, bounds=boundaries, mode=mode, exact=exact, admitted=admitted)
            if mode == "sparse" and boundaries == [0, 3, 515]:
                timings = {}
                for name, fn in (
                    ("original", original),
                    ("candidate", owner),
                    ("original_repeat", original),
                    ("candidate_repeat", owner),
                ):
                    times = []
                    for repeat in range(5):
                        torch.npu.synchronize()
                        begin = time.perf_counter()
                        for _ in range(5):
                            fn(*args)
                        torch.npu.synchronize()
                        times.append((time.perf_counter() - begin) * 1000 / 5)
                    timings[name] = times
                record["timings_ms"] = timings
            records.append(record)
            assert exact, record
        result.update(
            passed=True,
            cases=records,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            owner=owner.status(),
        )
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    Path(root.parent, f"qsa-gate-rank{worker.rank}.json").write_text(json.dumps(result, indent=2))
    return result


class Candidate:
    def validate(self, engine):
        assert (
            engine._pause_complete() and not engine.batch_queue and not engine.scheduler.get_num_unfinished_requests()
        )
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=all(row["passed"] for row in self.rows), workers=self.rows)

    def apply(self, engine):
        return self.validate(engine)


def prepare(engine):
    return Candidate()
