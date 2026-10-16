# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.qwen4exp.mamba_commit_runtime import ResidentMambaCommit, admit_commit
from tools.qwen4exp.resident_candidates.qfn_mamba_commit import MAMBA, RPC, make_replacements


def metadata():
    return [np.array(values, dtype=np.int32) for values in ([0, 1], [126, 255], [3, 3], [2, 2])]


def test_boundary_and_next_counts():
    tables = {1: np.array([[3, 2, 1, 0], [0, 1, 2, 3]], dtype=np.int32)}
    packed, counts, copies = admit_commit(metadata(), np.array([3, 1]), tables, block_size=128, slots=4)
    assert packed.shape == (5, 8) and packed.dtype == np.int32
    assert counts.tolist() == [1, 1]
    assert copies == [(0, 0, 0, 1)]


@pytest.mark.parametrize("bad", [-1, 64, 2000])
def test_global_or_invalid_ids_rejected_before_write(bad):
    table = np.zeros((2, 4), dtype=np.int32)
    table[0, 1] = bad
    with pytest.raises(ValueError, match="global or invalid"):
        admit_commit(metadata(), np.array([3, 1]), {1: table}, block_size=128, slots=64)


def test_no_copy_does_not_require_addressable_slots():
    columns = metadata()
    columns[1][:] = 10
    _, counts, copies = admit_commit(
        columns, np.array([3, 1]), {1: np.full((2, 4), -1, dtype=np.int32)}, block_size=128, slots=64
    )
    assert counts.tolist() == [3, 1] and not copies


@pytest.mark.parametrize("column,value", [(0, -1), (1, 1 << 31), (2, 0), (3, 4)])
def test_invalid_input_rejected(column, value):
    columns = [value.astype(np.int64) for value in metadata()]
    columns[column][0] = value
    with pytest.raises(ValueError):
        admit_commit(columns, np.array([3, 1]), {1: np.zeros((2, 4), dtype=np.int32)}, block_size=128, slots=64)


def test_changed_request_order_changes_copy_plan():
    columns = [value[::-1].copy() for value in metadata()]
    _, counts, copies = admit_commit(
        columns, np.array([1, 3]), {1: np.zeros((2, 4), dtype=np.int32)}, block_size=128, slots=64
    )
    assert counts.tolist() == [1, 1] and copies == [(1, 0, 0, 1)]


def test_wrapper_only_falls_back_before_submission():
    fallback = []
    resource = SimpleNamespace(enabled=True, execute=lambda kwargs: False)
    base = {MAMBA: lambda **kwargs: fallback.append(kwargs), RPC: lambda *args: "acceptance"}
    functions = make_replacements(base, resource)
    functions[MAMBA](key=1)
    assert fallback == [{"key": 1}]
    resource.execute = lambda kwargs: True
    functions[MAMBA](key=2)
    assert len(fallback) == 1

    def failed(kwargs):
        raise RuntimeError("submission uncertain")

    resource.execute = failed
    with pytest.raises(RuntimeError, match="uncertain"):
        functions[MAMBA](key=3)
    assert len(fallback) == 1


def test_control_preserves_acceptance_rpc_and_rejects_pending():
    runner = SimpleNamespace(execute_model_state=None, input_batch=SimpleNamespace(), _qfn_acceptance_reuse=True)
    worker = SimpleNamespace(rank=0, model_runner=runner)
    resource = SimpleNamespace(runner=runner, enabled=False, failed=False, status=lambda: {})
    functions = make_replacements({MAMBA: lambda **kwargs: None, RPC: lambda *args: "acceptance"}, resource)
    assert functions[RPC](worker, True) == "acceptance"
    payload = json.dumps(dict(enabled=True, verify=True, all_steps=True))
    assert functions[RPC](worker, payload)["passed"] and resource.enabled
    runner.execute_model_state = object()
    assert functions[RPC](worker, payload)["passed"] is False


def test_native_header_matches_admission_at_large_positions():
    # Native uses INT64 intermediates; input vectors remain INT32.
    columns = [np.array([value], dtype=np.int64) for value in (2046, 262142, 3, 2)]
    _, counts, copies = admit_commit(
        columns, np.array([3]), {1: np.zeros((1, 2050), dtype=np.int32)}, block_size=128, slots=64
    )
    assert copies == [(0, 2046, 2047, 1)] and counts.tolist() == [3]


def runtime_fixture():
    events = []
    resource = ResidentMambaCommit.__new__(ResidentMambaCommit)
    resource.failed = resource.verify = resource.all_steps = resource.staging_pending = False
    resource.slots = 64
    resource.groups, resource.block_size, resource.capacity = (1,), 128, {1: 4}
    resource.host_inputs = torch.zeros(5, 8, dtype=torch.int32)
    resource.inputs = torch.zeros_like(resource.host_inputs)
    resource.host_tables = {1: torch.zeros(4, 4, dtype=torch.int32)}
    resource.tables = {1: torch.zeros(4, 4, dtype=torch.int32)}
    resource.staging_done = SimpleNamespace(
        synchronize=lambda: events.append("fence"), record=lambda: events.append("record")
    )
    resource.records = {2: (None, SimpleNamespace(replay=lambda: events.append("replay")))}
    columns = metadata()
    ctx = SimpleNamespace(
        block_size=128,
        mamba_group_ids=[1],
        **{
            name: SimpleNamespace(np=value)
            for name, value in zip(
                ("mamba_state_idx_buf", "num_computed_tokens_buf", "num_scheduled_tokens_buf", "num_draft_tokens_buf"),
                columns,
            )
        },
    )
    batch = SimpleNamespace(
        _qfn_commit_counts=torch.tensor([3, 1]), _prefix_mamba_postprocess_tables={1: np.zeros((2, 4), dtype=np.int32)}
    )
    resource.runner = SimpleNamespace(input_batch=batch)
    kwargs = dict(
        input_batch=batch,
        num_reqs=2,
        bufs=SimpleNamespace(postprocess_align=ctx),
        num_accepted_tokens_cpu_tensor=torch.full((2,), 9, dtype=torch.int32),
    )
    return resource, kwargs, events


def test_staging_reuse_waits_before_second_replay_and_preserves_input_counts():
    resource, kwargs, events = runtime_fixture()
    assert resource.execute(kwargs)
    assert kwargs["num_accepted_tokens_cpu_tensor"].tolist() == [1, 1]
    assert kwargs["input_batch"]._qfn_commit_counts.tolist() == [3, 1]
    assert resource.execute(kwargs)
    assert events == ["record", "replay", "fence", "record", "replay"]


def test_idle_fast_path_submits_no_device_work():
    resource, kwargs, events = runtime_fixture()
    kwargs["bufs"].postprocess_align.num_computed_tokens_buf.np[:] = 10
    assert resource.execute(kwargs)
    assert events == []
    assert kwargs["num_accepted_tokens_cpu_tensor"].tolist() == [3, 1]


def test_submission_failure_keeps_owner_failed_and_does_not_publish_counts():
    resource, kwargs, events = runtime_fixture()

    def fail():
        raise RuntimeError("launch uncertain")

    resource.records[2][1].replay = fail
    with pytest.raises(RuntimeError, match="uncertain"):
        resource.execute(kwargs)
    assert resource.failed
    assert kwargs["num_accepted_tokens_cpu_tensor"].tolist() == [9, 9]
    with pytest.raises(RuntimeError, match="retain owners"):
        resource.execute(kwargs)
