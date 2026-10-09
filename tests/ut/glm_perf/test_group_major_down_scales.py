# SPDX-License-Identifier: Apache-2.0
"""Offline qualification for the paired hidden-scale store/down layout."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf.fused_offset_tables import projection_descriptor
from tools.glm_perf.glm_fused_moe import GROUP_MAJOR_DOWN_SCALE_LAYOUT, NativeFusedMoE


def test_actual_staging_and_consumer_helpers_preserve_bits_and_scratch_bounds(tmp_path):
    stubs = Path(__file__).parent / "reduce_kernel_cpu_stubs"
    executable = tmp_path / "check-down-scales"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            f"-I{stubs}",
            f"-I{builder.HERE}",
            str(stubs / "check_down_scales.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    assert "reused scratch, sparse/dense/sparse and consumer helpers passed" in result.stdout


@pytest.mark.parametrize("flag", [True, 1, "true"])
def test_invalid_build_creates_no_artifacts(tmp_path, flag):
    with pytest.raises(ValueError, match="group-major down scales|must be boolean"):
        builder.build(tmp_path / "build", tmp_path, tmp_path, group_major_down_scales=flag)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("input_layout", [False, True])
def test_only_paired_fused_stages_compile_down_layout(tmp_path, monkeypatch, enabled, input_layout):
    commands = []
    actual_run = subprocess.run

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
        version=1017,
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
        route_packed_down=True,
        route_compact_down_scales=True,
        quad_hidden_quant=True,
        raw_hidden_scales=True,
        raw_input_scales=True,
        group_major_input_scales=input_layout,
        group_major_down_scales=enabled,
    )
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["_build"]["group_major_down_scales"] is enabled
    assert provenance["_build"]["down_scale_layout"] == (GROUP_MAJOR_DOWN_SCALE_LAYOUT if enabled else "tile_row_v1")
    assert (
        provenance["glm_fused_down_scales.h"]["source_sha256"]
        == hashlib.sha256((builder.HERE / "glm_fused_down_scales.h").read_bytes()).hexdigest()
    )
    stages = 0
    for command in commands:
        if len(command) > 1:
            paired = Path(command[1]).name == "glm_fused_moe.cpp"
            stages += paired
            assert ("-DGLM_GROUP_MAJOR_DOWN_SCALES" in command) is (enabled and paired)
            if paired:
                # The down stage has no quantizer; the producer owns this define.
                assert ("-DGLM_QUAD_HIDDEN_QUANT" in command) is ("-DGLM_FUSED_GATE_UP" in command)
                # Exercise the actual stage's preprocessor guards with its frozen
                # defines; this is not Ascend compilation or Cube validation.
                stubs = Path(__file__).parent / "reduce_kernel_cpu_stubs"
                actual_run(
                    [
                        "c++",
                        "-E",
                        "-P",
                        f"-I{stubs}",
                        f"-I{builder.HERE}",
                        *(arg for arg in command if arg.startswith("-D")),
                        str(builder.HERE / "glm_fused_moe.cpp"),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
    assert stages == 6


@pytest.mark.parametrize(
    "options",
    [
        {"group_major_down_scales": 1},
        {"group_major_down_scales": True},
        {
            "group_major_down_scales": True,
            "raw_hidden_scales": True,
            "route_compact_down_scales": True,
            "quad_hidden_quant": True,
            "prefill_rows_32": True,
            "vector_scale_products": True,
            "prepared_weight_layout": True,
            "down_scale_layout": "tile_row_v1",
        },
        {
            "group_major_down_scales": True,
            "raw_hidden_scales": True,
            "route_compact_down_scales": True,
            "quad_hidden_quant": True,
            "prefill_rows_32": True,
            "vector_scale_products": True,
            "prepared_weight_layout": True,
            "direct_compact_down_scales": True,
        },
    ],
)
def test_runtime_rejects_bad_or_conflicting_layout_before_loading(tmp_path, options):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match="group-major down scales"):
        NativeFusedMoE(
            tmp_path,
            namespace="must_not_load",
            activation_bits=4,
            prepared_weight_layout=options.get("prepared_weight_layout", False),
        )


@pytest.mark.parametrize("gate_up", [False, True])
@pytest.mark.parametrize("tokens", [2, 17, 640, 1280])
def test_prepared_descriptor_abi_is_unchanged(gate_up, tokens):
    options = {"prefill_rows_32": True, "route_compact_down_scales": True, "raw_hidden_scales": True}
    header = (tokens * 8, 72, 4096, 2048, 3, 4, tokens, 8)
    assert projection_descriptor(header, gate_up=gate_up, options=options) == projection_descriptor(
        header, gate_up=gate_up, options={**options, "group_major_down_scales": True}
    )


def test_cli_records_paired_down_layout_as_keyword(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["build", "--build-dir", str(tmp_path), "--group-major-down-scales"])
    calls = []
    monkeypatch.setattr(builder, "build", lambda *args, **kwargs: calls.append(kwargs))
    builder.main()
    assert calls[0]["group_major_down_scales"] is True
