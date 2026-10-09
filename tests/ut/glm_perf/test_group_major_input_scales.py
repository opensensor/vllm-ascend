# SPDX-License-Identifier: Apache-2.0
"""Paired producer/consumer semantics, scratch bounds and frozen build contracts."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf.fused_offset_tables import projection_descriptor
from tools.glm_perf.glm_fused_moe import GROUP_MAJOR_INPUT_SCALE_LAYOUT, NativeFusedMoE


def test_actual_producer_and_consumer_helpers_preserve_bits_and_code_layout(tmp_path):
    stubs = Path(__file__).parent / "reduce_kernel_cpu_stubs"
    common = ["c++", "-std=c++17", "-O2", "-ffp-contract=off", f"-I{stubs}", f"-I{builder.HERE}"]
    objects = []
    for name, enabled in (("legacy", False), ("grouped", True)):
        output = tmp_path / (name + ".o")
        flags = ["-DGLM_RAW_INPUT_SCALES", f"-Dglm_fused_route_input_v1={name}_route"]
        if enabled:
            flags += ["-DGLM_GROUP_MAJOR_INPUT_SCALES"]
        subprocess.run(
            common + flags + ["-c", str(builder.HERE / "glm_fused_route_input.cpp"), "-o", str(output)], check=True
        )
        objects.append(str(output))
    executable = tmp_path / "check-input-layout"
    subprocess.run(common + [str(stubs / "check_route_input.cpp"), *objects, "-o", str(executable)], check=True)
    result = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    assert "paired consumers, changed metadata, UB bounds and packed codes passed" in result.stdout


@pytest.mark.parametrize("flag", [True, 1, "true"])
def test_invalid_build_creates_no_artifacts(tmp_path, flag):
    with pytest.raises(ValueError, match="group-major input scales|must be boolean"):
        builder.build(tmp_path / "build", tmp_path, tmp_path, group_major_input_scales=flag)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_only_paired_producer_and_gate_up_stages_receive_define(tmp_path, monkeypatch, enabled):
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
        version=1016,
        output_columns=128,
        tile_pipeline=True,
        all_bits=True,
        fused_moe=True,
        prepared_weight_layout=True,
        pair_scale_groups=True,
        pair_prefill_scale_groups=True,
        prefill_rows_32=True,
        vector_scale_products=True,
        compact_w4_scratch=True,
        specialize_w3=True,
        share_gate_up_input=True,
        cache_gate_up_activations=True,
        route_packed_input=True,
        raw_input_scales=True,
        group_major_input_scales=enabled,
    )
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["_build"]["group_major_input_scales"] is enabled
    assert provenance["_build"]["input_scale_layout"] == (GROUP_MAJOR_INPUT_SCALE_LAYOUT if enabled else "row_major_v1")
    for name in ("glm_fused_input_scales.h", "glm_route_input_layout.h"):
        assert provenance[name]["source_sha256"] == hashlib.sha256((builder.HERE / name).read_bytes()).hexdigest()
    paired = 0
    for command in commands:
        if len(command) > 1:
            paired_stage = Path(command[1]).name == "glm_fused_route_input.cpp" or (
                Path(command[1]).name == "glm_fused_moe.cpp" and "-DGLM_FUSED_GATE_UP" in command
            )
            paired += paired_stage
            assert ("-DGLM_GROUP_MAJOR_INPUT_SCALES" in command) is (enabled and paired_stage)
    assert paired == 4  # Ordinary, specialized W3/W4 gate/up plus producer.


@pytest.mark.parametrize(
    "options",
    [
        {"group_major_input_scales": 1},
        {"group_major_input_scales": True},
        {
            "group_major_input_scales": True,
            "raw_input_scales": True,
            "route_packed_input": True,
            "prefill_rows_32": True,
            "vector_scale_products": True,
            "prepared_weight_layout": True,
            "input_scale_layout": "row_major_v1",
        },
    ],
)
def test_runtime_rejects_invalid_or_mismatched_layout_before_loading(tmp_path, options):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match="group-major input scales"):
        NativeFusedMoE(
            tmp_path,
            namespace="must_not_load",
            activation_bits=4,
            prepared_weight_layout=options.get("prepared_weight_layout", False),
        )


@pytest.mark.parametrize("tokens", [2, 17, 640, 1280])
def test_prepared_descriptor_abi_is_unchanged(tokens):
    options = {"prefill_rows_32": True, "route_packed_input": True, "raw_input_scales": True}
    header = (tokens * 8, 72, 4096, 2048, 3, 4, tokens, 8)
    assert projection_descriptor(header, gate_up=True, options=options) == projection_descriptor(
        header, gate_up=True, options={**options, "group_major_input_scales": True}
    )


def test_cli_records_paired_layout_as_keyword(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["build", "--build-dir", str(tmp_path), "--group-major-input-scales"])
    calls = []
    monkeypatch.setattr(builder, "build", lambda *args, **kwargs: calls.append(kwargs))
    builder.main()
    assert calls[0]["group_major_input_scales"] is True
