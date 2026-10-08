# SPDX-License-Identifier: Apache-2.0
"""Pin diagnostic assets and require explicit device selection before imports."""

import json
from pathlib import Path

import pytest

from tools.glm_perf import build_kda_beta_scale_vector as builder
from tools.glm_perf.kda_beta_scale_vector_probe import run, verify


def test_device_gate_is_required_before_loading_assets_or_backend(tmp_path):
    with pytest.raises(ValueError, match="explicit device selection"):
        run(tmp_path / "missing", tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


def test_append_only_paired_build_hashes_every_asset(tmp_path, monkeypatch):
    compiler = tmp_path / "compiler"
    compiler.write_text("mock")
    cann = tmp_path / "cann"
    cann.mkdir()
    commands = []

    def compile_kernel(command, **kwargs):
        commands.append(command)
        Path(command[2]).write_bytes(b"mock kernel")

    monkeypatch.setattr(builder.subprocess, "run", compile_kernel)
    monkeypatch.setattr(builder, "compile_bridge", lambda output, namespace, cann: output.write_bytes(b"mock bridge"))
    output = tmp_path / "candidate"
    builder.build(output, compiler, "glm_kda_beta_scale_v977", 977, cann)
    provenance = verify(output)
    assert provenance["compile_only"] and not provenance["full_kda_evaluated"]
    assert len(commands) == 2 and all("--npu-arch=dav-2002" in command for command in commands)
    assert (output / "kda_beta_scalar.cpp").read_text().startswith("#define GLM_KDA_BETA_SCALAR_REFERENCE")
    assert not (output / "kda_beta_vector.cpp").read_text().startswith("#define GLM_KDA_BETA_SCALAR_REFERENCE")
    with pytest.raises(FileExistsError):
        builder.build(output, compiler, "glm_kda_beta_scale_v977", 977, cann)
    (output / "kda_beta_vector.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset changed"):
        verify(output)
    with pytest.raises(ValueError, match="fresh version namespace"):
        builder.build(tmp_path / "bad", compiler, "glm_kda_beta_scale_v976", 977, cann)
    assert not (tmp_path / "bad").exists()
    assert json.loads((output / "provenance.json").read_text()) == provenance
