# SPDX-License-Identifier: Apache-2.0
"""Compare the actual compact index helper with the existing scale-layout ABI."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf.fused_offset_tables import projection_descriptor, projection_offset_tables
from tools.glm_perf.glm_fused_moe import NativeFusedMoE


@pytest.fixture(scope="module")
def actual_group_offsets(tmp_path_factory):
    root = tmp_path_factory.mktemp("compact-scales")
    source = root / "indices.cpp"
    source.write_text("""#include "glm_fused_compact_scales.h"
#include <iostream>
int main() {
  for (unsigned group = 0; group < 128; ++group)
    std::cout << GlmFusedCompactScales::GroupOffset(group) << " ";
}
""")
    executable = root / "indices"
    stubs = Path(__file__).parent / "reduce_kernel_cpu_stubs"
    subprocess.run(
        ["c++", "-std=c++17", f"-I{stubs}", f"-I{builder.HERE}", str(source), "-o", str(executable)], check=True
    )
    return torch.tensor([int(v) for v in subprocess.check_output([str(executable)], text=True).split()])


@pytest.mark.parametrize("groups", [8, 16, 64, 128])
@pytest.mark.parametrize("count", [5, 7, 16, 17, 30, 31])
def test_direct_columns_preserve_all_scale_bits_and_padded_rows(actual_group_offsets, groups, count):
    options = {"prefill_rows_32": True, "route_compact_down_scales": True, "raw_hidden_scales": True}
    header = (640 * 8, 72, 4096, groups * 32, 3, 4, 640, 8)
    activation_offsets = torch.tensor(projection_offset_tables(header, gate_up=False, options=options)[3][:groups]) // 4
    # Include signed zero, infinity and NaN bit patterns. Layout must copy bits,
    # including unused-row clamping; no floating arithmetic is performed here.
    packed = (torch.arange(groups * 32, dtype=torch.int64) * 2654435761).to(torch.int32)
    packed[:4] = torch.tensor([0, -2147483648, 2139095040, 2143289344], dtype=torch.int32)
    old_rows = torch.stack([packed[activation_offsets + row * 4] for row in range(count)])
    selected_rows = torch.arange(32).clamp(max=count - 1)
    selected_rows[count:] = 0
    expected = old_rows[selected_rows].T
    actual = packed[actual_group_offsets[:groups, None] + selected_rows[None, :] * 4]
    assert torch.equal(actual, expected)
    assert int((actual_group_offsets[:groups, None] + selected_rows[None, :] * 4).max()) < groups * 32
    before = projection_descriptor(header, gate_up=False, options=options)
    after = projection_descriptor(header, gate_up=False, options={**options, "direct_compact_down_scales": True})
    assert before == after


@pytest.mark.parametrize("flag", [True, 1, "true"])
def test_invalid_build_creates_no_artifacts(tmp_path, flag):
    with pytest.raises(ValueError, match="direct compact scales|must be boolean"):
        builder.build(tmp_path / "build", tmp_path, tmp_path, direct_compact_down_scales=flag)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_only_down_stages_compile_direct_layout(tmp_path, monkeypatch, enabled):
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
        version=1015,
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
        share_gate_up_input=True,
        cache_gate_up_activations=True,
        route_packed_input=True,
        route_packed_down=True,
        route_compact_down_scales=True,
        quad_hidden_quant=True,
        raw_hidden_scales=True,
        direct_compact_down_scales=enabled,
    )
    assert json.loads((output / "provenance.json").read_text())["_build"]["direct_compact_down_scales"] is enabled
    for command in commands:
        if len(command) > 1:
            is_down = Path(command[1]).name == "glm_fused_moe.cpp" and "-DGLM_FUSED_GATE_UP" not in command
            assert ("-DGLM_DIRECT_COMPACT_DOWN_SCALES" in command) is (enabled and is_down)


@pytest.mark.parametrize("options", [{"direct_compact_down_scales": 1}, {"direct_compact_down_scales": True}])
def test_invalid_runtime_loads_no_kernels(tmp_path, options):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match="direct compact scales"):
        NativeFusedMoE(tmp_path, namespace="must_not_load", activation_bits=4)
