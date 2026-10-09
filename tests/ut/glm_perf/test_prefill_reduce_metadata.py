# SPDX-License-Identifier: Apache-2.0
"""Run the actual reducer C++ with CPU semantics; device timing stays unqualified."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf.glm_fused_moe import NativeFusedMoE


@pytest.mark.parametrize("native_columns", [False, True])
def test_actual_kernel_output_order_coverage_peer_guards_and_metadata_reads(tmp_path, native_columns):
    stubs = Path(__file__).parent / "reduce_kernel_cpu_stubs"
    common = ["c++", "-std=c++17", "-O2", "-ffp-contract=off", f"-I{stubs}", f"-I{builder.HERE}"]
    flags = ["-DGLM_FP16_ROUTE_WORKSPACE"]
    if native_columns:
        flags += ["-DGLM_NATIVE_ROUTE_COLUMNS"]
    objects = []
    for name, enabled in (("legacy", False), ("cached", True)):
        output = tmp_path / (name + ".o")
        command = common + flags + [f"-Dglm_fused_reduce_half_v1={name}_reduce"]
        if enabled:
            command += ["-DGLM_PREFILL_REDUCE_META_CACHE"]
        subprocess.run(command + ["-c", str(builder.HERE / "glm_fused_reduce.cpp"), "-o", str(output)], check=True)
        objects.append(str(output))
    executable = tmp_path / "check-reducer"
    subprocess.run(common + flags + [str(stubs / "check_reduce.cpp"), *objects, "-o", str(executable)], check=True)
    result = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    assert "metadata read counts passed" in result.stdout


@pytest.mark.parametrize("flag", [True, 1, "true"])
def test_invalid_build_has_no_side_effects(tmp_path, flag):
    with pytest.raises(ValueError, match="cached reducer metadata|must be boolean"):
        builder.build(tmp_path / "build", tmp_path, tmp_path, prefill_reduce_meta_cache=flag)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_only_reducer_receives_new_define_and_schedule_is_identified(tmp_path, monkeypatch, enabled):
    commands = []

    def fake_compile(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1] if "-o" in command else command[2]).write_bytes(b"offline-stub")

    monkeypatch.setattr(builder.subprocess, "run", fake_compile)
    monkeypatch.setattr(builder, "include_paths", lambda: [])
    monkeypatch.setattr(builder, "library_paths", lambda: [])
    monkeypatch.setattr(
        builder.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "init.py"))
    )
    output = builder.build(
        tmp_path / "build",
        tmp_path,
        tmp_path,
        version=1014,
        output_columns=128,
        tile_pipeline=True,
        all_bits=True,
        fused_moe=True,
        fp16_route_workspace=True,
        prefill_reduce_meta_cache=enabled,
    )
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["_build"]["prefill_reduce_meta_cache"] is enabled
    assert (
        provenance["glm_fused_reduce_schedule.h"]["source_sha256"]
        == hashlib.sha256((builder.HERE / "glm_fused_reduce_schedule.h").read_bytes()).hexdigest()
    )
    for command in commands:
        if len(command) > 1:
            expected = enabled and Path(command[1]).name == "glm_fused_reduce.cpp"
            assert ("-DGLM_PREFILL_REDUCE_META_CACHE" in command) is expected


@pytest.mark.parametrize("options", [{"prefill_reduce_meta_cache": 1}, {"prefill_reduce_meta_cache": True}])
def test_invalid_runtime_contract_loads_no_kernels(tmp_path, options):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match="cached reducer metadata"):
        NativeFusedMoE(tmp_path, namespace="must_not_load", activation_bits=4)
