# SPDX-License-Identifier: Apache-2.0
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates.tp6_state_commit import MAMBA, QUALIFIED_FALLBACK_SHA256, RPC, make_replacements


def fixture():
    counts = torch.tensor([1, 2, 3, 1, 2, 3])
    batch = SimpleNamespace(_mamba_accepted_counts_snapshot=counts)
    runner = SimpleNamespace(input_batch=batch, execute_model_state=None)
    resource = SimpleNamespace(
        runner=runner, enabled=True, verify=True, failed=False, checked_blocks=0, status=lambda: {}
    )
    seen = []

    def execute(kwargs):
        assert kwargs["input_batch"]._qfn_commit_counts is counts
        seen.append(kwargs["num_reqs"])
        return True

    resource.execute = execute
    calls = []
    replacements = make_replacements(lambda **kw: calls.append(kw), lambda *args: "previous", resource)
    return resource, replacements, seen, calls


def test_uses_existing_raw_acceptance_without_readback_or_leaking_step_scope():
    resource, functions, seen, calls = fixture()
    batch = resource.runner.input_batch
    functions[MAMBA](input_batch=batch, num_reqs=6)
    assert seen == [6] and not calls and resource.verified_shapes == {6}
    assert not hasattr(batch, "_qfn_commit_counts")
    assert batch._mamba_accepted_counts_snapshot.tolist() == [1, 2, 3, 1, 2, 3]


def test_declined_admission_falls_back_but_failed_submission_never_retries():
    resource, functions, seen, calls = fixture()
    batch = resource.runner.input_batch
    resource.execute = lambda kwargs: False
    functions[MAMBA](input_batch=batch, num_reqs=6)
    assert len(calls) == 1 and not resource.verified_shapes

    def fail(kwargs):
        raise RuntimeError("uncertain submission")

    resource.execute = fail
    with pytest.raises(RuntimeError, match="uncertain"):
        functions[MAMBA](input_batch=batch, num_reqs=6)
    assert len(calls) == 1 and not hasattr(batch, "_qfn_commit_counts")


def test_fast_mode_requires_real_six_row_boundary_verification_and_drained_snapshot():
    resource, functions, _, _ = fixture()
    worker = SimpleNamespace(rank=5, model_runner=resource.runner)
    control = functions[RPC]
    request = json.dumps(dict(tp6_commit=True, enabled=True, verify=False))
    assert not control(worker, request)["passed"]  # a snapshot is still live
    del resource.runner.input_batch._mamba_accepted_counts_snapshot
    assert not control(worker, request)["passed"]
    resource.verified_shapes.add(6)
    assert not control(worker, request)["passed"]
    resource.checked_blocks = 72
    assert control(worker, request)["passed"]
    assert resource.enabled and not resource.verify and not resource.all_steps
    resource.failed = True
    assert not control(worker, request)["passed"]
    assert control(worker, True) == "previous"


def test_reviewed_fallback_source_guard_matches_checkout():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/patch_mamba_utils.py"
    source = path.read_text()
    node = next(
        n for n in ast.parse(source).body if getattr(n, "name", None) == "_postprocess_mamba_align_gpu_cpu_fallback"
    )
    function = "".join(source.splitlines(True)[node.lineno - 1 : node.end_lineno])
    assert hashlib.sha256(function.encode()).hexdigest() == QUALIFIED_FALLBACK_SHA256
