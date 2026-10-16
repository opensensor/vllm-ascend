# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.qwen4exp.ple_shared_gather import SharedPLEGather


@pytest.fixture
def owner(tmp_path):
    path = tmp_path / "table"
    data = np.arange(8 * 160, dtype=np.float16).reshape(8, 160)
    path.write_bytes(data.tobytes())
    stat = path.stat()
    method = SimpleNamespace(
        embedding_dim=160,
        num_embeddings=8,
        dtype=torch.float16,
        _checkpoint_dir=tmp_path,
        gather_rows=lambda ids: "fallback",
    )
    metadata = dict(
        geometry=dict(table_rows=8),
        model=str(tmp_path),
        fd_path=str(path),
        bytes=stat.st_size,
        device=stat.st_dev,
        inode=stat.st_ino,
    )
    value = SharedPLEGather(method, metadata)
    yield value, data, method, metadata
    value.table = None
    value.mapping.close()
    import os

    os.close(value.fd)


def test_exact_order_duplicates_and_output_ownership(owner):
    gather, data, _, _ = owner
    ids = torch.tensor([[7, 0], [7, 2]])
    actual = gather.gather_rows(ids)
    assert torch.equal(actual, torch.from_numpy(data[[7, 0, 7, 2]]))
    actual.zero_()
    assert np.array_equal(gather.table, data)
    assert not gather.table.flags.writeable
    empty = gather.gather_rows(torch.empty(0, dtype=torch.int64))
    assert empty.shape == (0, 160)
    assert empty.view(torch.uint8).shape == (0, 320)


@pytest.mark.parametrize("ids", [[-1], [8], [0, 8]])
def test_range_check(owner, ids):
    with pytest.raises(IndexError):
        owner[0].gather_rows(torch.tensor(ids))


def test_non_cpu_keeps_existing_transport(owner):
    assert owner[0].gather_rows(SimpleNamespace(device=SimpleNamespace(type="npu"))) == "fallback"


@pytest.mark.parametrize("field,value", [("inode", 0), ("bytes", 1), ("model", "/different/model")])
def test_identity_and_geometry_fail_closed(owner, field, value):
    _, _, method, metadata = owner
    with pytest.raises(ValueError):
        SharedPLEGather(method, dict(metadata, **{field: value}))


def test_binding_shadow_failure_retains_reference_and_restores():
    from tools.qwen4exp.ple_shared_gather import SharedPLEBinding

    class Method:
        def gather_rows(self, ids):
            return torch.ones((ids.numel(), 160), dtype=torch.float16)

    method = Method()
    gather = SimpleNamespace(
        original=method.gather_rows, gather_rows=lambda ids: torch.zeros((ids.numel(), 160), dtype=torch.float16)
    )
    binding = SharedPLEBinding(method, gather)
    binding.install()
    ids = torch.tensor([0])
    assert torch.equal(method.gather_rows(ids), torch.ones((1, 160), dtype=torch.float16))
    assert binding.failed and binding.calls == 1
    method.gather_rows(ids)
    assert binding.calls == 1
    binding.restore()
    assert "gather_rows" not in vars(method)


def test_binding_success_and_ownership_checks(owner):
    from tools.qwen4exp.ple_shared_gather import SharedPLEBinding

    gather, data, method, _ = owner
    method.gather_rows = lambda ids: torch.from_numpy(data[ids.numpy()])
    gather.original = method.gather_rows
    binding = SharedPLEBinding(method, gather, verify_calls=2)
    original = method.gather_rows
    binding.install()
    for _ in range(3):
        assert torch.equal(method.gather_rows(torch.tensor([2])), torch.from_numpy(data[[2]]))
    assert binding.status() == dict(installed=True, failed=False, checked=2, remaining=0, calls=3)
    method.gather_rows = original
    with pytest.raises(RuntimeError, match="ownership changed"):
        binding.restore()
    method.gather_rows = binding
    binding.restore()
    assert method.gather_rows is original


def test_resident_control_source_is_bounded_and_reversible():
    from tools.qwen4exp.ple_shared_gather_control import engine_source

    for action in ("enable", "disable", "status"):
        source = engine_source(action)
        assert len(source.encode()) < 128 * 1024
        namespace = {}
        exec(compile(source, "<test-shared-ple-control>", "exec"), namespace)
        assert namespace["ACTION"] == action
        assert callable(namespace["prepare"])
    with pytest.raises(ValueError):
        engine_source("reset")


def test_control_preparation_does_not_bind_and_disable_restores():
    from tools.qwen4exp.ple_shared_gather_control import engine_source

    class Method:
        def gather_rows(self, ids):
            return torch.zeros((ids.numel(), 160), dtype=torch.float16)

    class Worker:
        def resident_status(self):
            return dict(rank=self.rank)

    workers = []
    originals = []
    for rank in range(6):
        method = Method()
        gather = SimpleNamespace(original=method.gather_rows, gather_rows=method.gather_rows)
        worker = Worker()
        worker.rank = rank
        worker._qfn_shared_gather_gate = dict(
            method=method, gather=gather, results=[dict(exact=True) for _ in range(16)]
        )
        workers.append(worker)
        originals.append(method.gather_rows)
    engine = SimpleNamespace(
        _pause_complete=lambda: True,
        batch_queue=[],
        scheduler=SimpleNamespace(get_num_unfinished_requests=lambda: 0),
        model_executor=SimpleNamespace(collective_rpc=lambda fn, args: [fn(w, *args) for w in workers]),
    )
    for action in ("enable", "disable"):
        namespace = {}
        exec(engine_source(action), namespace)
        namespace["SNAPSHOT_SOURCE"] = "def worker_snapshot(worker): return dict(rank=worker.rank)"
        candidate = namespace["prepare"](engine)
        assert candidate.validate(engine)["passed"]
        assert all(w._qfn_shared_prefill_binding.installed == (action == "disable") for w in workers)
        assert candidate.apply(engine)["passed"]
        assert all(w._qfn_shared_prefill_binding.installed == (action == "enable") for w in workers)
        assert all(w.resident_status()["shared_prefill_ple"]["installed"] == (action == "enable") for w in workers)
    assert all(w._qfn_shared_gather_gate["method"].gather_rows == original for w, original in zip(workers, originals))
