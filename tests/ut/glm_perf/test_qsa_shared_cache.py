# SPDX-License-Identifier: Apache-2.0
"""Shared-cache source admission, launch ABI and reversible instance binding."""

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.instance_bindings import InstanceBindings
from tools.glm_perf.qsa_shared_binding import bind_attention, extend_replacements
from tools.glm_perf.qsa_shared_native import NativeQsaShared, tiling_header
from tools.glm_perf.stage_qsa_accumulate_rows import transform as accumulate_rows
from tools.glm_perf.stage_qsa_output_rows import transform as output_rows
from tools.glm_perf.stage_qsa_shared_cache import stage, transform

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "csrc/attention/qsa_sparse_attention_v310/op_kernel/qsa_cube_sparse_attention_v310.h"
)


def preprocess(source):
    source = "\n".join(line for line in source.splitlines() if not line.startswith("#include"))
    return subprocess.run(
        ["c++", "-E", "-P", "-x", "c++", "-"], input=source, text=True, capture_output=True, check=True
    ).stdout


def test_flag_off_preserves_complete_parent_code_and_does_not_change_allocations():
    parent = SOURCE.read_text()
    candidate = transform(parent)
    assert preprocess(candidate) == preprocess(parent)
    assert candidate.count("pipe_->InitBuffer(") == parent.count("pipe_->InitBuffer(")
    assert candidate.count("if (!sharedKeyValue_)") == 5
    assert "sharedKeyValue_ = keyCache == valueCache;" in candidate


def test_staging_rejects_parent_drift_and_cannot_overwrite_existing_candidate(tmp_path):
    parent = SOURCE
    candidate = tmp_path / "candidate.h"
    with pytest.raises(ValueError, match="differs"):
        stage(parent, candidate, "0" * 64)
    assert not candidate.exists()
    digest = hashlib.sha256(parent.read_bytes()).hexdigest()
    result = stage(parent, candidate, digest)
    assert not result["serving_evaluated"] and result["extra_ub_bytes"] == 0
    original = candidate.read_bytes()
    with pytest.raises(FileExistsError):
        stage(parent, candidate, digest)
    assert candidate.read_bytes() == original
    with pytest.raises(ValueError, match="already staged"):
        transform(candidate.read_text())
    with pytest.raises(ValueError, match="anchors changed"):
        transform(parent.read_text().replace("    TPipe *pipe_ = nullptr;", ""))


@pytest.mark.parametrize(
    "tokens,heads,dim,physical,logical",
    [(1, 16, 512, 1, 1), (640, 16, 512, 1, 1), (2, 32, 256, 2, 2), (17, 16, 512, 2, 1)],
)
def test_tiling_tasks_cover_each_token_and_head_exactly_once(tokens, heads, dim, physical, logical):
    header, blocks = tiling_header(
        (tokens, heads, dim), (14, physical * dim // 16, 640, 16), (tokens, 512), (2, 486), (3,), 256**-0.5, logical
    )
    assert len(header) == 14 and 0 < blocks <= 8
    per_task, tiles, tasks_per_core, count = header[3], header[4], header[11], header[12]
    coverage = set()
    for core in range(blocks):
        for offset in range(tasks_per_core):
            task = core * tasks_per_core + offset
            if task >= count:
                break
            row = task // (logical * tiles)
            head = (task % (logical * tiles)) // tiles
            start = (task % tiles) * per_task
            for query_head in range(start, min(heads // logical, start + per_task)):
                position = (row, head * (heads // logical) + query_head)
                assert position not in coverage
                coverage.add(position)
    assert coverage == {(row, head) for row in range(tokens) for head in range(heads)}
    assert header[-1] == 1048576


@pytest.mark.parametrize("scale,expected", [(0.5 / (1 << 24), 1), (-0.5 / (1 << 24), -1)])
def test_scale_matches_cpp_llround_half_away_from_zero(scale, expected):
    header, _ = tiling_header((1, 16, 512), (14, 32, 640, 16), (1, 512), (1, 8), (2,), scale)
    assert header[-1] == expected


def tensors():
    query = torch.empty(2, 16, 512, dtype=torch.float16)
    cache = torch.empty(2, 32, 640, 16, dtype=torch.float16)
    metadata = (
        torch.zeros(2, 512, dtype=torch.int32),
        *[torch.zeros(2, dtype=torch.int32) for _ in range(3)],
        torch.zeros(1, 8, dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
    )
    return query, cache, metadata


def test_launch_keeps_twelve_pointer_abi_and_reuses_config_without_device_reads(monkeypatch):
    calls = []
    native = NativeQsaShared(object(), lambda kernel, args, blocks: calls.append((args, blocks)), torch.device("cpu"))
    query, cache, metadata = tensors()
    first = native(query, cache, cache, *metadata, 0.0625)
    assert len(calls[0][0]) == 12 and calls[0][0][9] is first
    config = calls[0][0][-1]
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: pytest.fail("descriptor allocated again"))
    native(query, cache, cache, *metadata, 0.0625)
    assert calls[1][0][-1] is config and native.calls == native.shared_calls == 2
    with pytest.raises(ValueError, match="FP16"):
        native(query.float(), cache, cache, *metadata, 0.0625)
    assert len(calls) == 2


class Attention:
    glm_indexer = SimpleNamespace(topk_tokens=2048, index_kpool=4)
    num_heads, num_kv_heads, kv_lora_rank, scale = 16, 1, 512, 0.0625
    _forward_decode_fused = object()

    @staticmethod
    def _get_paged_latent_op():
        return "original operator"


def runner():
    owners = [Attention(), Attention()]
    cache = torch.empty(1, 32, 640, 16, dtype=torch.float16)
    roots = [SimpleNamespace(modules=lambda owner=owner: [SimpleNamespace(impl=owner)]) for owner in owners]
    return owners, SimpleNamespace(
        model=roots[0],
        drafter=SimpleNamespace(model=roots[1]),
        kv_caches=[[cache, cache]],
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(
                block_tables=[
                    SimpleNamespace(
                        is_mamba_group=False, block_size=32, block_table=SimpleNamespace(gpu=torch.empty(4, 9720))
                    )
                ]
            )
        ),
    )


def test_instance_binding_preserves_static_class_getter_and_restores_target_and_draft():
    owners, active_runner = runner()
    native = SimpleNamespace(device=torch.device("cpu"), configs={})
    bindings = InstanceBindings()
    assert bind_attention(bindings, active_runner, native, tiling_header) == 2
    assert all(owner._get_paged_latent_op() is native for owner in owners)
    assert Attention._get_paged_latent_op() == "original operator"
    assert len(native.configs) == 2560
    bindings.restore()
    assert all(owner._get_paged_latent_op() == "original operator" and not vars(owner) for owner in owners)


@pytest.mark.parametrize("fail", [False, True])
def test_capture_failure_and_successor_restore_bindings_before_transition(fail):
    owners, active_runner = runner()
    prefix = "vllm_ascend._310p.worker_310p:NPUWorker310."
    native = SimpleNamespace(device=torch.device("cpu"), configs={}, calls=0, shared_calls=0)
    worker = SimpleNamespace(model_runner=active_runner)
    worker._resident_error = lambda error: dict(error=str(error))
    worker._resident_session = lambda: SimpleNamespace(graphs_dirty=False)

    def capture(self):
        assert all(owner._get_paged_latent_op() is native for owner in owners)
        return dict(error="fixture failure") if fail else dict(captured=True)

    def apply(self, generation):
        assert all(owner._get_paged_latent_op() == "original operator" for owner in owners)
        return dict(generation=generation)

    changes = {
        prefix + "resident_capture": capture,
        prefix + "resident_apply": apply,
        prefix + "resident_status": lambda self: {},
    }
    candidate = extend_replacements(changes, native, tiling_header)
    result = candidate[prefix + "resident_capture"](worker)
    if fail:
        assert result["error"] and all(not vars(owner) for owner in owners)
    else:
        assert candidate[prefix + "resident_status"](worker)["shared_qsa_cache"]["attention_instances"] == 2
    assert candidate[prefix + "resident_apply"](worker, "next") == dict(generation="next")


def test_output_row_stage_keeps_flag_off_math_and_rejects_changed_loop():
    parent = SOURCE.read_text()
    batched = output_rows(transform(parent))
    assert preprocess(batched) == preprocess(parent)
    assert batched.count("GLM_QSA_OUTPUT_ROWS") == 1
    with pytest.raises(ValueError, match="already staged"):
        output_rows(batched)
    with pytest.raises(ValueError, match="loop changed"):
        output_rows(parent.replace("const float inverse = rowSum[head]", "const float reciprocal = rowSum[head]"))


def test_accumulator_row_stage_keeps_flag_off_order_and_rejects_changed_math():
    parent = SOURCE.read_text()
    batched = accumulate_rows(output_rows(transform(parent)))
    assert preprocess(batched) == preprocess(parent)
    with pytest.raises(ValueError, match="already staged"):
        accumulate_rows(batched)
    with pytest.raises(ValueError, match="arithmetic changed"):
        accumulate_rows(parent.replace("Muls(accumulator[offset]", "Adds(accumulator[offset]"))
