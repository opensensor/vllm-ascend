# SPDX-License-Identifier: Apache-2.0
"""Bound the populated Cube geometry and reject incompatible readback schedules."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.glm_perf import build_reconstruction as builder

SOURCE = Path(__file__).resolve().parents[3] / "tools/glm_perf/glm_fused_moe.cpp"


def test_compiled_row_selector_bounds_both_independent_products(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler")
    source = SOURCE.read_text()
    start = source.index("  __aicore__ inline uint32_t ActiveCubeRows(")
    end = source.index("\n#endif", start)
    method = source[start:end].replace("__aicore__", "")
    program = (
        "#include <cstdint>\n#include <cassert>\n"
        "constexpr uint32_t M=32, BULK_TOKENS=16, CUBE_ROW_TILE=16;\n"
        "constexpr bool PAIR_PREFILL_SCALE_GROUPS=true;\n"
        "struct Projection { uint32_t tokens_, activationBits_;\n" + method + "\n};\n"
        "int main() {\n"
        "for (uint32_t count=1; count<32; ++count) {\n"
        "Projection p{640,4}; uint32_t rows=p.ActiveCubeRows(count,true);\n"
        "assert(rows>=2*count && rows<=64 && rows%16==0);\n"
        "assert(rows-16<2*count);\n"
        "for (uint32_t part=0; part<2; ++part) {\n"
        "for (uint32_t m=0; m<count; ++m) {\n"
        "for (uint32_t channel=0; channel<128; ++channel) {\n"
        "uint32_t offset=(channel/16*rows+part*count+m)*16+channel%16;\n"
        "assert(offset<rows*128);\n"
        "assert((offset/16)%rows==part*count+m);\n"
        "assert(offset/(rows*16)==channel/16);\n"
        "}}}\n"
        "p.tokens_=16; assert(p.ActiveCubeRows(count,true)==(count>16?64:32));\n"
        "p.tokens_=640; p.activationBits_=8;\n"
        "assert(p.ActiveCubeRows(count,false)==(count<=15?32:64));\n"
        "}\n}\n"
    )
    path = tmp_path / "rows.cpp"
    path.write_text(program)
    binary = tmp_path / "rows"
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize(
    "options",
    [
        {"active_cube_rows": 1},
        {"direct_w4_l1": 1},
        {"direct_w4_l1": True},
        {"active_cube_rows": True},
        {"active_cube_rows": True, "prefill_rows_32": True, "pair_prefill_scale_groups": False},
        {"active_cube_rows": True, "nz_prefill_accumulator": True},
        {"active_cube_rows": True, "prefill_product_cast": True},
    ],
)
def test_bad_geometry_rejected_before_build_directory(tmp_path, options):
    with pytest.raises(ValueError):
        builder.build(tmp_path / "build", tmp_path, tmp_path, **options)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"prefill_rows_32": False}, "active Cube rows require paired M32"),
        ({"pair_prefill_scale_groups": False}, "active Cube rows require paired M32"),
        ({"nz_prefill_accumulator": True}, "active Cube rows require the qualified strided readback"),
        ({"prefill_product_cast": True}, "active Cube rows require the qualified strided readback"),
        ({"direct_w4_l1": True, "prefill_weight_cache": False}, "direct W4 L1 requires prepared projection cache"),
    ],
)
def test_incompatible_paths_rejected_after_valid_prerequisites(tmp_path, options, message):
    valid = dict(
        output_columns=128,
        tile_pipeline=True,
        all_bits=True,
        fused_moe=True,
        prepared_weight_layout=True,
        pair_scale_groups=True,
        vector_scale_products=True,
        prefill_rows_32=True,
        pair_prefill_scale_groups=True,
        active_cube_rows=True,
    )
    valid.update(options)
    with pytest.raises(ValueError, match=message):
        builder.build(tmp_path / "build", tmp_path, tmp_path, **valid)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_builder_records_and_compiles_both_projection_entries(tmp_path, monkeypatch, active, direct):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        target = command[command.index("-o") + 1] if "-o" in command else command[2]
        Path(target).write_bytes(b"host fixture only")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    monkeypatch.setattr(
        builder.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    monkeypatch.setattr(builder, "include_paths", lambda: [])
    monkeypatch.setattr(builder, "library_paths", lambda: [])
    output = builder.build(
        tmp_path / "build",
        tmp_path,
        tmp_path,
        version=979,
        output_columns=128,
        tile_pipeline=True,
        all_bits=True,
        fused_moe=True,
        prepared_weight_layout=True,
        pair_scale_groups=True,
        pair_prefill_scale_groups=True,
        vector_scale_products=True,
        prefill_rows_32=True,
        compact_w4_scratch=True,
        prefill_weight_cache=True,
        active_cube_rows=active,
        direct_w4_l1=direct,
    )
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["_build"]["active_cube_rows"] is active
    assert provenance["_build"]["direct_w4_l1"] is direct
    projections = [command for command in commands if len(command) > 1 and Path(command[1]).name == SOURCE.name]
    assert len(projections) == 4
    assert all(("-DGLM_ACTIVE_CUBE_ROWS" in command) is active for command in projections)
    assert all(("-DGLM_DIRECT_W4_L1" in command) is direct for command in projections)
