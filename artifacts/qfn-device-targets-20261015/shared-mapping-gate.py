def expected_rows(tokens, history, starts, geometry):
    result = []
    for request, (start, end) in enumerate(zip(starts, starts[1:])):
        for position in range(start, end):
            local = position - start
            current = tokens[position]
            recent = tokens[position - 1] if local else history[request][1]
            older = tokens[position - 2] if local >= 2 else history[request][local]
            if recent == geometry.eos:
                older = geometry.eos
            first = current * geometry.multipliers[0] ^ recent * geometry.multipliers[1]
            result.extend(
                (first ^ (older * geometry.multipliers[2] if head >= 8 else 0)) % size + offset
                for head, (size, offset) in enumerate(zip(geometry.sizes, geometry.offsets))
            )
    return result


"""Qualify new PLE owners privately; never replace a live owner in this gate."""


def operation(wrapper):
    import ctypes
    import hashlib
    import json
    import os
    import time
    from pathlib import Path
    from types import SimpleNamespace

    import torch

    worker = getattr(wrapper, "worker", wrapper)
    result = dict(rank=worker.rank, passed=False)
    root = Path("/srv/ai/storage/artifacts/qfn-device-targets-20261015/shared-mapping-v1")
    try:
        geometry = worker._qfn_qsa_geometry
        assert geometry["scope"]["qsa_sparse_attention_310"] is geometry["replacement"]
        geometry["scope"]["qsa_sparse_attention_310"] = geometry["original"]
        result["qsa_geometry"] = [dict(shapes=key, calls=value) for key, value in geometry["records"].items()]
        result["qsa_calls"] = geometry["calls"]
        del worker._qfn_qsa_geometry
        live = wrapper._qfn_native_ple_serving
        manager = live["manager"]
        assert not manager.busy and not manager.failed
        assert not hasattr(worker, "_qfn_shared_ple_probe")
        ns = {}
        exec(compile((root / "ple_serving.py").read_text(), str(root / "ple_serving.py"), "exec"), ns)
        library = root / "libple_host_gather.so"
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        pool = SimpleNamespace(
            metadata=manager.metadata,
            fd=manager.fd,
            binary=manager.binary,
            lib=ns["load_library"](library, digest),
            retained=[],
            mapping_parent=None,
        )
        worker._qfn_shared_ple_probe = dict(pool=pool, namespace=ns, owners={}, receipt=result)
        state = worker._qfn_shared_ple_probe
        device = wrapper.model_runner.device
        torch.npu.synchronize()
        begin = time.perf_counter()
        baseline = live["namespace"]["Lookup"](manager, 11, 4, device)
        result["old_create_ms"] = (time.perf_counter() - begin) * 1000
        # No producer was attached and no transfer was submitted to this private
        # baseline. Close it now; existing serving owners are not touched.
        ns["checked"](manager.lib.ple_host_close(baseline.handle), "private baseline close")
        baseline.handle = ctypes.c_void_p()
        manager.retained.remove(baseline)
        records = []
        for count in range(1, 13):
            for requests in range(1, 5):
                begin = time.perf_counter()
                owner = ns["Lookup"](pool, count, requests, device)
                create_ms = (time.perf_counter() - begin) * 1000
                state["owners"][(count, requests, device.index)] = owner
                tokens = [(71 + i * 23) % owner.geometry.vocabulary for i in range(count)]
                histories = [[owner.geometry.eos, 113 + i] for i in range(requests)]
                starts = [i * count // requests for i in range(requests + 1)]
                inputs = (
                    torch.tensor(tokens, dtype=torch.int64, device=device),
                    torch.tensor(starts, dtype=torch.int32, device=device),
                    torch.tensor(histories, dtype=torch.int64, device=device),
                )
                rows = expected_rows(tokens, histories, starts, owner.geometry)
                expected = b"".join(os.pread(manager.fd, 320, row * 320) for row in rows)
                for iteration in range(2):
                    actual = owner.fetch(*inputs)
                    assert owner.rows.cpu().reshape(-1).tolist() == rows
                    assert actual.cpu().contiguous().view(torch.uint8).numpy().tobytes() == expected
                records.append(dict(tokens=count, requests=requests, create_ms=create_ms, exact=True))
        result.update(passed=True, library_sha256=digest, cases=records, owners=len(pool.retained))
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    Path(root.parent, f"shared-gate-rank{worker.rank}.json").write_text(json.dumps(result, indent=2))
    return result


class Candidate:
    def validate(self, engine):
        assert (
            engine._pause_complete() and not engine.batch_queue and not engine.scheduler.get_num_unfinished_requests()
        )
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(self.rows) == 4 and all(row["passed"] for row in self.rows), workers=self.rows)

    def apply(self, engine):
        return self.validate(engine)


def prepare(engine):
    return Candidate()
