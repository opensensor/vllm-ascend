# SPDX-License-Identifier: Apache-2.0
"""Compare prepared descriptors with the compiled legacy scalar table writer."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf.fused_offset_tables import HEADER_WORDS, LAYOUT_TAG, projection_descriptor, projection_offset_tables
from tools.glm_perf.glm_fused_moe import NativeFusedMoE

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("k", [256, 2048, 4096])
@pytest.mark.parametrize(
    "gate_up,options",
    [
        (True, {}),
        (True, {"prefill_rows_32": True, "active_cube_rows": True, "quad_hidden_quant": True}),
        (False, {"prefill_rows_32": True, "route_compact_down_scales": True, "raw_hidden_scales": True}),
        (False, {"native_route_columns": True, "nz_prefill_accumulator": True}),
    ],
)
def test_tables_equal_compiled_legacy_writer(tmp_path, gate_up, options, k):
    """Run the production scalar writer on host and compare every serialized index."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler")
    source = (ROOT / "tools/glm_perf/glm_fused_moe.cpp").read_text()
    start = source.index("  __aicore__ inline void PrepareOffsets() {")
    end = source.index("\n#ifdef GLM_WEIGHT_DECODE_LUT\n  __aicore__ inline void LookupPair", start)
    method = source[start:end].replace("__aicore__", "")
    quant_source = (ROOT / "tools/glm_perf/glm_fused_quantize.h").read_text()
    start = quant_source.index("__aicore__ inline void Prepare(")
    end = quant_source.index("__aicore__ inline void Run(", start)
    quant_method = quant_source[start:end].replace("__aicore__", "")
    m = 32 if options.get("prefill_rows_32") else 16
    layouts = 2 * m // 16 if options.get("active_cube_rows") else 2
    batch = 16 if options.get("quad_hidden_quant") else 8
    flags = []
    if gate_up:
        flags.append("GLM_FUSED_GATE_UP")
    flags += [
        define
        for name, define in (
            ("active_cube_rows", "GLM_ACTIVE_CUBE_ROWS"),
            ("native_route_columns", "GLM_NATIVE_ROUTE_COLUMNS"),
            ("nz_prefill_accumulator", "GLM_NZ_PREFILL_ACCUMULATOR"),
            ("route_compact_down_scales", "GLM_COMPACT_DOWN_SCALES"),
        )
        if options.get(name)
    ]
    header = (640, 1, 4096, k, 3, 4, 640, 1)
    tables = projection_offset_tables(header, gate_up=gate_up, options=options)
    program = """#include <cstdint>
#include <vector>
#include <iostream>
using half=uint16_t;
template<class T> struct LocalTensor {
    std::vector<uint32_t>* data; uint32_t base;
    void SetValue(uint32_t i, T value) { data->at(base+i)=value; }
    LocalTensor operator[](uint32_t i) const { return {data,base+i}; }
};
struct Buffer {
    std::vector<uint32_t> data=std::vector<uint32_t>(32768,0);
    template<class T> LocalTensor<T> Get() { return {&data,0}; }
};
constexpr int PIPE_ALL=0;
template<int> void PipeBarrier() {}
"""
    program += f"constexpr uint32_t M={m}, N=128, GROUP=32, NZ_N=16, LANES=8, MAX_K_GROUPS=128;\n"
    program += f"constexpr uint32_t CUBE_ROW_TILE=16, PRODUCT_LAYOUT_COUNT={layouts}, BULK_TOKENS=16;\n"
    lanes = 4 if options.get("raw_hidden_scales") else 8
    program += f"constexpr uint32_t SCALE_INDEX_OFFSET=3072, DOWN_SCALE_ROW_LANES={lanes}, NZ_SCALE_BLOCKS=4;\n"
    program += f"namespace GlmFusedQuant {{ constexpr uint32_t GROUP=32, BATCH={batch}, ELEMENTS=GROUP*BATCH;\n"
    program += quant_method + "}\n"
    program += f"struct Projection {{ int64_t k_={k},tokens_=640,activationBits_=4;\n"
    program += "Buffer outputOffsets_,productOffsets_,scaleCache_,activationScales_,quantOffsets_;\n"
    program += method + "};\nint main(){ Projection p; p.PrepareOffsets();\n"
    for table, buf, start in zip(
        tables,
        ("outputOffsets_", "productOffsets_", "scaleCache_", "activationScales_", "quantOffsets_"),
        (0, 0, 3072 // 4, (m + 8) * 128, 0),
    ):
        program += f'for(unsigned i=0;i<{len(table)};++i) std::cout<<p.{buf}.data.at({start}+i)<<" ";\n'
    program += "}\n"
    path = tmp_path / "legacy.cpp"
    path.write_text(program)
    binary = tmp_path / "legacy"
    subprocess.run(
        [compiler, "-std=c++17", *("-D" + f for f in flags), str(path), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    actual = tuple(map(int, subprocess.check_output([str(binary)], text=True).split()))
    assert actual == sum(tables, ())
    descriptor = torch.tensor(projection_descriptor(header, gate_up=gate_up, options=options), dtype=torch.int64)
    assert tuple(descriptor[:8].tolist()) == header
    assert descriptor[8:12].tolist() == [LAYOUT_TAG, len(actual), 0, 0]
    assert tuple(descriptor[HEADER_WORDS:].view(torch.uint32).tolist()) == actual
    assert descriptor.numel() * descriptor.element_size() % 32 == 0


@pytest.mark.parametrize(
    "header",
    [
        (1, 1, 128, 256, 3, 4, 2, 1),
        (1, 1, 128, 256, 3, 4, 1, True),
        (1, 1, 128, 255, 3, 4, 1, 1),
        (1, 1, 129, 256, 3, 4, 1, 1),
    ],
)
def test_bad_descriptors_rejected_on_host(header):
    with pytest.raises(ValueError):
        projection_descriptor(header, gate_up=True, options={})


@pytest.mark.parametrize("flag", [True, 1])
def test_builder_rejects_offsets_without_prepared_fused_layout(tmp_path, flag):
    with pytest.raises(ValueError):
        builder.build(tmp_path / "build", tmp_path, tmp_path, prepared_offset_tables=flag)
    assert not (tmp_path / "build").exists()


def test_native_configs_prepare_once_and_preserve_existing_launch_arguments():
    native = NativeFusedMoE.__new__(NativeFusedMoE)
    native.device, native.activation_bits = torch.device("cpu"), 4
    native.configs, native.scratch = {}, {}
    native.prepared_offset_tables = True
    native.offset_options = {}
    native.pack_kernel, native.gate_kernel, native.down_kernel, native.reduce_kernel = "pack", "gate", "down", "reduce"
    native.weight_lookup = None
    calls = []
    native.launch = lambda kernel, args, blocks: calls.append((kernel, args))
    x = torch.zeros(2, 256, dtype=torch.float16)
    gate = torch.zeros(1, 512, 128, dtype=torch.int8)
    down = torch.zeros(1, 256, 128, dtype=torch.int8)
    scales = torch.ones(1, 16, 8)
    down_scales = torch.ones(1, 8, 8)
    weights, ids = torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int64)
    native(x, gate, scales, down, down_scales, weights, ids)
    gate_config, down_config = calls[1][1][-1], calls[2][1][-1]
    assert gate_config[8].item() == down_config[8].item() == LAYOUT_TAG
    assert len(calls[1][1]) == 11 and len(calls[2][1]) == 10
    calls.clear()
    native(x * 2, gate, scales, down, down_scales, weights * 0.75, ids)
    assert calls[1][1][-1] is gate_config and calls[2][1][-1] is down_config


def test_builder_freezes_descriptor_helper_and_compiles_all_entries(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        target = command[command.index("-o") + 1] if "-o" in command else command[2]
        Path(target).write_bytes(b"host fixture")

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
        version=984,
        output_columns=128,
        tile_pipeline=True,
        all_bits=True,
        fused_moe=True,
        prepared_weight_layout=True,
        compact_w4_scratch=True,
        prepared_offset_tables=True,
    )
    record = json.loads((output / "provenance.json").read_text())
    assert record["_build"]["prepared_offset_tables"] is True
    assert "fused_offset_tables.py" in record["_helpers"]
    projections = [c for c in commands if len(c) > 1 and Path(c[1]).name == "glm_fused_moe.cpp"]
    assert len(projections) == 4 and all("-DGLM_PREPARED_OFFSET_TABLES" in c for c in projections)
