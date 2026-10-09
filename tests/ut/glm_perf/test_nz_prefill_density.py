# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the reachable M32 expert-batch density experiment."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf.fused_offset_tables import projection_descriptor
from tools.glm_perf.glm_fused_moe import NativeFusedMoE


@pytest.mark.parametrize("threshold", [-1, 1, 16, 32, 33, True, 31.0, "31"])
def test_reject_unreachable_or_invalid_threshold_before_build(tmp_path, threshold):
    with pytest.raises(ValueError, match="NZ prefill minimum rows"):
        builder.build(tmp_path / "build", tmp_path, tmp_path, nz_prefill_min_rows=threshold)
    assert not (tmp_path / "build").exists()


def test_threshold_requires_enabled_nz_schedule(tmp_path):
    with pytest.raises(ValueError, match="require the NZ accumulator"):
        builder.build(tmp_path / "build", tmp_path, tmp_path, nz_prefill_min_rows=31)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("threshold", [0, 17, 30, 31])
@pytest.mark.parametrize("cache_ends", [False, True])
def test_freeze_threshold_for_all_specialized_stages(tmp_path, monkeypatch, threshold, cache_ends):
    commands = []

    def fake_compile(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1] if "-o" in command else command[2]).write_bytes(b"offline-stub")

    monkeypatch.setattr(builder.subprocess, "run", fake_compile)
    monkeypatch.setattr(builder, "include_paths", lambda: [])
    monkeypatch.setattr(builder, "library_paths", lambda: [])
    monkeypatch.setattr(
        builder.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    root = builder.build(
        tmp_path / "build",
        tmp_path,
        tmp_path,
        version=1013,
        output_columns=128,
        tile_pipeline=True,
        all_bits=True,
        fused_moe=True,
        prepared_weight_layout=True,
        pair_scale_groups=True,
        compact_w4_scratch=True,
        vector_scale_products=True,
        prefill_rows_32=True,
        pair_prefill_scale_groups=True,
        nz_prefill_accumulator=True,
        nz_prefill_min_rows=threshold,
        cache_expert_ends=cache_ends,
    )
    options = json.loads((root / "provenance.json").read_text())["_build"]
    assert options["nz_prefill_min_rows"] == threshold
    assert options["cache_expert_ends"] is cache_ends
    stages = [cmd for cmd in commands if len(cmd) > 1 and Path(cmd[1]).name == "glm_fused_moe.cpp"]
    assert len(stages) == 4
    for command in stages:
        assert "-DGLM_NZ_PREFILL_ACCUMULATOR" in command
        assert ("-DGLM_CACHE_EXPERT_ENDS" in command) is cache_ends
        assert [arg for arg in command if arg.startswith("-DGLM_NZ_PREFILL_MIN_ROWS=")] == (
            [f"-DGLM_NZ_PREFILL_MIN_ROWS={threshold}"] if threshold else []
        )


@pytest.mark.parametrize(
    "options",
    [
        {"nz_prefill_min_rows": 32},
        {"nz_prefill_min_rows": True},
        {"nz_prefill_min_rows": 31},
        {"nz_prefill_min_rows": 31, "nz_prefill_accumulator": True, "prefill_rows_32": True},
    ],
)
def test_runtime_rejects_bad_contract_without_loading_kernels(tmp_path, options):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match="NZ prefill minimum rows"):
        NativeFusedMoE(tmp_path, namespace="must_not_load", activation_bits=4)


@pytest.mark.parametrize("gate_up", [False, True])
@pytest.mark.parametrize("tokens", [2, 640, 1280])
def test_density_selection_preserves_prepared_descriptor_abi(gate_up, tokens):
    options = {"nz_prefill_accumulator": True, "prefill_rows_32": True, "pair_prefill_scale_groups": True}
    header = (tokens * 8, 72, 4096, 2048, 3, 4, tokens, 8)
    old = projection_descriptor(header, gate_up=gate_up, options=options)
    new = projection_descriptor(header, gate_up=gate_up, options={**options, "nz_prefill_min_rows": 31})
    assert old == new


def test_cli_passes_threshold_as_keyword(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["build", "--build-dir", str(tmp_path), "--nz-prefill-accumulator", "--nz-prefill-min-rows", "31"]
    )
    calls = []
    monkeypatch.setattr(builder, "build", lambda *args, **kwargs: calls.append((args, kwargs)))
    builder.main()
    assert calls[0][1] == {
        "nz_prefill_min_rows": 31,
        "cache_expert_ends": False,
        "prefill_reduce_meta_cache": False,
        "direct_compact_down_scales": False,
    }
