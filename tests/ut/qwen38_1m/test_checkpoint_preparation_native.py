# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.checkpoint_preparation_native import CheckpointPrecopyGroup, load, select_backend
from tools.qwen4exp.checkpoint_preparation_native_gate import oracle


@pytest.mark.parametrize("kind", ["shape", "dtype", "stride"])
def test_invalid_metadata_rejected_before_native_submission(kind):
    owner = CheckpointPrecopyGroup.__new__(CheckpointPrecopyGroup)
    owner.rows = 2
    owner.receipt = torch.empty((1, 2, 8), dtype=torch.int32)
    value = torch.zeros(2, dtype=torch.int32)
    invalid = {
        "shape": torch.zeros(3, dtype=torch.int32),
        "dtype": value.long(),
        "stride": torch.zeros(4, dtype=torch.int32)[::2],
    }[kind]
    with pytest.raises(ValueError, match="contiguous INT32"):
        owner(value, invalid, value)


def test_backend_switch_keeps_qualification_and_flushes_counts():
    calls = []
    owner = SimpleNamespace(
        failed=False,
        backend="tensor",
        backend_checks={"tensor": (10, 72), "fused": (0, 0)},
        checked=10,
        checked_blocks=72,
        materialize_counts=lambda: calls.append("flush"),
        tensor_records={1: "tensor"},
        fused_records={1: "fused"},
    )
    with pytest.raises(RuntimeError, match="live decision"):
        select_backend(owner, "fused", verify=False)
    assert not calls and owner.backend == "tensor"
    select_backend(owner, "fused", verify=True)
    assert owner.verify and owner.records is owner.fused_records
    owner.checked, owner.checked_blocks = 5, 144
    select_backend(owner, "fused", verify=False)
    assert not owner.verify and owner.backend_checks["fused"] == (5, 144)
    select_backend(owner, "tensor", verify=False)
    assert owner.checked == 10 and owner.checked_blocks == 72
    assert owner.records is owner.tensor_records and calls == ["flush"] * 3
    owner.failed = True
    with pytest.raises(ValueError, match="failed"):
        select_backend(owner, "fused", verify=True)


def test_native_load_rejects_binary_drift_before_loading(tmp_path, monkeypatch):
    source = tmp_path / "source.cpp"
    source.write_bytes(b"source")
    binary = tmp_path / "binary.bin"
    binary.write_bytes(b"changed binary")
    manifest = dict(
        namespace="qwen_checkpoint_preparation_v2",
        flush_host_queue=False,
        sources={source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        bridge=dict(path=str(binary), sha256=hashlib.sha256(b"original binary").hexdigest()),
        binaries={},
    )
    (tmp_path / "provenance.json").write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setattr(torch.ops, "load_library", calls.append)
    with pytest.raises(ValueError, match="binary provenance"):
        load(tmp_path)
    assert not calls


@pytest.mark.parametrize("conv", [False, True])
def test_native_gate_oracle_preserves_overlap_and_tail(conv):
    state = torch.arange(4 * 5 * 8, dtype=torch.int32).reshape(4, 5, 8)
    table = torch.tensor([[0, 0, 2, 3]], dtype=torch.int32)
    actual, receipt = oracle(state, table, torch.tensor([0]), torch.tensor([126]), torch.tensor([2]), conv)
    expected = state.clone()
    selected = state[0, 1:] if conv else state[0]
    expected[0].reshape(-1)[: selected.numel()].copy_(selected.reshape(-1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert receipt[0, 0] == 1 and not receipt[1].any()
