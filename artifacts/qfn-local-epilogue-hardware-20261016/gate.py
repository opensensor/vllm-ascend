"""Private resident gates; no model method changes or new serving bindings."""

# Controller prepends these immutable experiment selections.
PHASE = globals()["PHASE"]
VERSION = globals()["VERSION"]


def operation(wrapper):
    worker = getattr(wrapper, "worker", wrapper)
    try:
        import hashlib
        import inspect
        from pathlib import Path

        import torch

        from tools.qwen4exp.native_prefill import NativeGroupedPrefill
        from vllm_ascend.models.qwen4_exp.w4_moe import PackedExpertBank

        root = Path("/srv/ai/storage/artifacts/qfn-local-epilogue-hardware-20261016")
        attribute = "_qfn_local_epilogue_private" + ("_v2" if VERSION == 2 else "")
        build = root / ("build-v2" if VERSION == 2 else "build")
        state = getattr(worker, attribute, None)
        if state is None:
            state = dict(namespace={"__name__": "qfn_private_local_epilogue"}, retained=[], records={})
            setattr(worker, attribute, state)
            exec(
                compile((root / "local_route_epilogue.py").read_text(), str(root / "local_route_epilogue.py"), "exec"),
                state["namespace"],
            )
            ns = state["namespace"]
            constructor = torch.classes.qwen_route_finalize_v3.Kernel
            launch = torch.ops.qwen_route_finalize_v3.launch
            state["projection"] = NativeGroupedPrefill(
                constructor(str(build / "native_local_down_projection.bin"), "qwen_local_down_projection_v1"), launch
            )
            state["epilogues"] = {
                mode: ns["LocalRouteEpilogue"](
                    constructor(str(build / "native_local_route_epilogue.bin"), entry), launch, mode=mode
                )
                for mode, entry in zip(ns["MODES"], ns["ENTRY_POINTS"])
            }
            state["reference_projection"] = inspect.getclosurevars(PackedExpertBank.native_linear).nonlocals["resource"]
            state["reference_finalizer"] = worker._qfn_pack_serving["finalizer"]
            state["pairs"] = {
                mode: ns["LocalDownEpilogue"](
                    state["projection"],
                    epilogue,
                    native_backend=PackedExpertBank.native_linear._qfn_prefill_schedule_base.__globals__[
                        "NATIVE_INT4_BACKEND"
                    ],
                )
                for mode, epilogue in state["epilogues"].items()
            }
            state["hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in build.glob("*.bin")}
        generator = torch.Generator().manual_seed(160260 + worker.rank)
        records = []
        with torch.no_grad():
            if PHASE == "epilogue":
                for rows in (130, 650, 5150, 20480):
                    for active in (0, 1, rows // 4, rows):
                        source = torch.randn(rows, 2560, generator=generator, dtype=torch.float16)
                        if active:
                            source[0, :4] = torch.tensor([-0.0, 2**-24, -(2**-24), 65504], dtype=torch.float16)
                        source[active:] = 0
                        routed = source.to(worker.device)
                        inverse = torch.randperm(rows, generator=generator).to(worker.device)
                        weights = torch.rand(rows, 1, generator=generator).to(worker.device)
                        ends = torch.tensor(
                            [(i + 1) * active // 128 for i in range(128)], dtype=torch.int64, device=worker.device
                        )
                        expected = state["reference_finalizer"](routed, inverse, weights)
                        reduced = expected.reshape(-1, 10, 2560).sum(1)
                        # Poison the same buffer only after the reference has consumed it.
                        routed[active:] = float("nan")
                        state["retained"] = [routed, inverse, weights, ends, expected, reduced]
                        for mode, epilogue in state["epilogues"].items():
                            actual = epilogue(routed, inverse, weights, ends)
                            reference = expected if mode == "ordered" else reduced
                            exact = torch.equal(actual.view(torch.int32), reference.view(torch.int32))
                            records.append(
                                dict(
                                    rows=rows,
                                    active=active,
                                    mode=mode,
                                    exact=exact,
                                    max_abs_error=float((actual - reference).abs().max().cpu()),
                                )
                            )
                            del actual
                        torch.npu.synchronize()
                        state["retained"] = []
                        del routed, inverse, weights, ends, expected, reduced
                # Cancellation explicitly distinguishes the two proposed trees.
                rows = 130
                routed = torch.zeros(rows, 2560, dtype=torch.float16, device=worker.device)
                routed[:4] = torch.tensor([65504, 1, -65504, 1], dtype=torch.float16, device=worker.device).view(4, 1)
                weights = torch.ones(rows, 1, device=worker.device)
                weights[1] = weights[3] = 2**-10
                inverse = torch.arange(rows, device=worker.device)
                ends = torch.full((128,), rows, dtype=torch.int64, device=worker.device)
                reference = state["reference_finalizer"](routed, inverse, weights).reshape(-1, 10, 2560).sum(1)
                for mode in ("sequential", "pairwise"):
                    actual = state["epilogues"][mode](routed, inverse, weights, ends)
                    records.append(
                        dict(
                            case="cancellation",
                            mode=mode,
                            exact=torch.equal(actual.view(torch.int32), reference.view(torch.int32)),
                            reference=float(reference[0, 0].cpu()),
                            actual=float(actual[0, 0].cpu()),
                        )
                    )
            elif PHASE == "projection":
                assert state["records"].get("epilogue")
                assert all(r["exact"] for r in state["records"]["epilogue"] if r["mode"] == "ordered")
                banks = [
                    (n, m.projections["down_proj"])
                    for n, m in worker.model_runner.get_model().named_modules()
                    if type(m).__name__ == "W4SparseMoE"
                ]
                valid_modes = [
                    mode
                    for mode in state["epilogues"]
                    if all(r["exact"] for r in state["records"]["epilogue"] if r["mode"] == mode)
                ]
                for name, bank in (banks[0], banks[-1]):
                    for rows in (650, 5150, 20480):
                        source = torch.randn(rows, 640, generator=generator, dtype=torch.float16).to(worker.device)
                        prepared = torch.ops._C_ascend.npu_qwen_w4_a8_pack_310(source)
                        for active, skew in (
                            (0, False),
                            (1, False),
                            (rows // 4, False),
                            (rows // 4, True),
                            (rows, False),
                        ):
                            counts = (
                                [active // 128 + (i < active % 128) for i in range(128)]
                                if not skew
                                else [active] + [0] * 127
                            )
                            ends = torch.tensor(counts, dtype=torch.int64, device=worker.device).cumsum(0)
                            inverse = torch.randperm(rows, generator=generator).to(worker.device)
                            weights = torch.rand(rows, 1, generator=generator).to(worker.device)
                            full = state["reference_projection"](bank, prepared, ends)
                            reference = (
                                state["reference_finalizer"](full, inverse, weights).reshape(-1, 10, 2560).sum(1)
                            )
                            state["retained"] = [source, prepared, ends, inverse, weights, full, reference]
                            # The paired path must preserve the full real-bank result.
                            for mode in valid_modes:
                                actual = state["pairs"][mode](bank, prepared, ends, inverse, weights)
                                exact = torch.equal(actual.view(torch.int32), reference.view(torch.int32))
                                records.append(
                                    dict(
                                        bank=name,
                                        rows=rows,
                                        active=active,
                                        skew=skew,
                                        mode=mode,
                                        exact=exact,
                                        max_abs_error=float((actual - reference).abs().max().cpu()),
                                    )
                                )
                                del actual
                            torch.npu.synchronize()
                            state["retained"] = []
                            del full, reference
                        del source, prepared
            elif PHASE == "packed":
                from tools.qwen4exp.benchmark_shared_hc_operand_310 import paired_timing

                assert all(r["exact"] for r in state["records"]["projection"])
                state["adapters"] = {
                    mode: state["namespace"]["PackedLocalDownEpilogue"](
                        state["pairs"][mode], worker._qfn_pack_serving["owner"], state["reference_finalizer"]
                    )
                    for mode in ("ordered", "sequential")
                }
                for adapter in state["adapters"].values():
                    adapter.enabled = True
                    adapter.verify = False
                banks = [
                    (n, m.projections["down_proj"])
                    for n, m in worker.model_runner.get_model().named_modules()
                    if type(m).__name__ == "W4SparseMoE"
                ]
                for name, bank in (banks[0], banks[-1]):
                    for rows in (650, 2590, 5150, 20480):
                        projected = (torch.randn(rows, 1280, generator=generator) * 1.5).half().to(worker.device)
                        for active in (0, 1, rows // 4, rows):
                            ends = torch.tensor(
                                [(i + 1) * active // 128 for i in range(128)], dtype=torch.int64, device=worker.device
                            )
                            inverse = torch.randperm(rows, generator=generator).to(worker.device)
                            weights = torch.rand(rows, 1, generator=generator).to(worker.device)
                            args = (projected, bank, ends, inverse, weights, torch.float32, torch.float16)
                            state["retained"] = [args]
                            baseline = state["adapters"]["ordered"].baseline
                            reference = baseline(*args)
                            for mode, adapter in state["adapters"].items():
                                actual = adapter(*args)
                                exact = torch.equal(actual.view(torch.int32), reference.view(torch.int32))
                                record = dict(
                                    bank=name,
                                    rows=rows,
                                    active=active,
                                    mode=mode,
                                    exact=exact,
                                    max_abs_error=float((actual - reference).abs().max().cpu()),
                                )
                                records.append(record)
                                if not exact:
                                    torch.npu.synchronize()
                                    state["records"][PHASE] = records
                                    return dict(rank=worker.rank, safe=True, passed=False, records=records)
                            if active == rows // 4:
                                functions = {"retained": lambda baseline=baseline, args=args: baseline(*args)}
                                functions.update(
                                    {
                                        mode: lambda adapter=adapter, args=args: adapter(*args)
                                        for mode, adapter in state["adapters"].items()
                                    }
                                )
                                record["timing"] = paired_timing(functions, 2, 3)
                            torch.npu.synchronize()
                            state["retained"] = []
                            del reference, actual
            else:
                raise ValueError(PHASE)
            torch.npu.synchronize()
        state["records"][PHASE] = records
        return dict(
            rank=worker.rank,
            safe=True,
            passed=all(r["exact"] for r in records),
            records=records,
            hashes=state["hashes"],
        )
    except Exception as error:
        import traceback

        return dict(rank=worker.rank, safe=False, passed=False, error=repr(error), traceback=traceback.format_exc())


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        return dict(passed=True)

    def apply(self, engine):
        self.validate(engine)
        rows = engine.model_executor.collective_rpc(operation)
        return dict(
            passed=len(rows) == 4 and all(r["safe"] for r in rows),
            candidate_passed=all(r["passed"] for r in rows),
            workers=rows,
        )


def prepare(engine):
    return Candidate()
