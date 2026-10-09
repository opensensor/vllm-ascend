# SPDX-License-Identifier: Apache-2.0
"""Reject incompatible accumulation bundles before compilation or device access."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf.glm_fused_moe import NativeFusedMoE


@pytest.mark.parametrize("flag", [True, 1, "true"])
def test_invalid_accumulation_build_creates_no_artifacts(tmp_path, flag):
    with pytest.raises(ValueError, match="fused scale accumulation requires|must be boolean"):
        builder.build(tmp_path / "build", tmp_path, tmp_path, fused_scale_accumulation=flag)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"fused_scale_accumulation": 1},
        {"fused_scale_accumulation": True},
        {
            "fused_scale_accumulation": True,
            "fused_moe": True,
            "vector_scale_products": True,
            "nz_prefill_accumulator": True,
        },
    ],
)
def test_invalid_runtime_contract_loads_no_kernels(tmp_path, options):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match="scheduling flags|fused scale accumulation requires"):
        NativeFusedMoE(tmp_path, namespace="unused", activation_bits=4)


@pytest.mark.parametrize("enabled", [False, True])
def test_all_projection_entries_record_explicit_accumulation_contract(tmp_path, monkeypatch, enabled):
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
    output = builder.build(
        tmp_path / "build",
        tmp_path,
        tmp_path,
        version=1008,
        output_columns=128,
        tile_pipeline=True,
        all_bits=True,
        fused_moe=True,
        prepared_weight_layout=True,
        compact_w4_scratch=True,
        vector_scale_products=True,
        fused_scale_accumulation=enabled,
    )
    options = json.loads((output / "provenance.json").read_text())["_build"]
    assert options["fused_scale_accumulation"] is enabled
    entries = [command for command in commands if len(command) > 1 and Path(command[1]).name == "glm_fused_moe.cpp"]
    assert len(entries) == 4
    assert all(("-DGLM_FUSED_SCALE_ACCUMULATION" in command) is enabled for command in entries)
