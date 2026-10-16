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
from tools.glm_perf import fused_offset_tables
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


@pytest.mark.parametrize("gate_up", [False, True])
@pytest.mark.parametrize("k", [256, 2048, 4096])
@pytest.mark.parametrize("tokens", [1, 17, 640, 1280])
@pytest.mark.parametrize("bits,activation_bits", [(2, 4), (3, 8), (4, 4), (4, 8)])
@pytest.mark.parametrize("active_cube_rows,nz_accumulator", [(False, False), (True, True)])
def test_streaming_descriptor_extends_exact_legacy_prefix(
    gate_up, k, tokens, bits, activation_bits, active_cube_rows, nz_accumulator
):
    header = (tokens * 8, 72, 4096, k, bits, activation_bits, tokens, 8)
    options = {
        "prefill_rows_32": True,
        "active_cube_rows": active_cube_rows,
        "nz_prefill_accumulator": nz_accumulator,
        "native_route_columns": not gate_up,
        "route_compact_down_scales": True,
        "raw_hidden_scales": True,
        "quad_hidden_quant": gate_up,
    }
    legacy = projection_offset_tables(header, gate_up=gate_up, options=options)
    streaming_options = {**options, "streaming_pipeline": True}
    streaming = projection_offset_tables(header, gate_up=gate_up, options=streaming_options)
    assert len(legacy) == 5 and len(streaming) == 6
    assert streaming[:5] == legacy
    assert streaming[-1] == tuple(row * 4 for row in range(32))
    old_descriptor = projection_descriptor(header, gate_up=gate_up, options=options)
    descriptor = projection_descriptor(header, gate_up=gate_up, options=streaming_options)
    assert descriptor[:9] == old_descriptor[:9]
    assert descriptor[9] == old_descriptor[9] + 32
    assert descriptor[10 : len(old_descriptor)] == old_descriptor[10:]
    assert descriptor[len(old_descriptor) :] == tuple(row * 4 | (row + 1) * 4 << 32 for row in range(0, 32, 2))
    assert (len(descriptor) - len(old_descriptor)) * 8 == 128
    assert len(descriptor) * 8 % 32 == 0
    assert (
        projection_descriptor(header, gate_up=gate_up, options={**options, "streaming_pipeline": False})
        == old_descriptor
    )


@pytest.mark.parametrize("mutation", ["wrong_suffix", "unaligned_suffix", "unaligned_prefix", "missing_suffix"])
def test_descriptor_rejects_corrupt_streaming_tables(monkeypatch, mutation):
    header = (640, 72, 4096, 2048, 4, 4, 640, 1)
    options = {"streaming_pipeline": True, "prefill_rows_32": True}
    tables = list(projection_offset_tables(header, gate_up=True, options=options))
    if mutation == "wrong_suffix":
        tables[-1] = tuple(row * 8 for row in range(32))
    elif mutation == "unaligned_suffix":
        tables[-1] = tables[-1][:-1]
    elif mutation == "unaligned_prefix":
        tables[0] = tables[0][:-1]
    else:
        tables.pop()
    monkeypatch.setattr(fused_offset_tables, "projection_offset_tables", lambda *args, **kwargs: tuple(tables))
    with pytest.raises(ValueError, match="suffix|unaligned|table count"):
        projection_descriptor(header, gate_up=True, options=options)


def _run_prepared_dma(tmp_path, *, gate_up, options, k, activation_bits, mutation=None):
    """Compile actual PrepareOffsets, enforcing DMA alignment, bounds and poison guards."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler")
    source = (ROOT / "tools/glm_perf/glm_fused_moe.cpp").read_text()
    start = source.index("  __aicore__ inline void PrepareOffsets() {")
    end = source.index("\n#ifdef GLM_WEIGHT_DECODE_LUT\n  __aicore__ inline void LookupPair", start)
    method = source[start:end].replace("__aicore__", "")
    if mutation == "missing_quant_advance":
        assert "offset += QUANT_OFFSET_BYTES / sizeof(uint32_t);" in method
        method = method.replace("offset += QUANT_OFFSET_BYTES / sizeof(uint32_t);", "")
    elif mutation == "wrong_mask_destination":
        method = method.replace("GlmStreamingScaleGather::GROUP_SOURCE_OFFSET / sizeof(uint32_t)", "0")
    elif mutation == "wrong_source":
        method = method.replace("offsetTables_[offset], M);", "offsetTables_[0], M);")
    elif mutation == "unaligned_copy":
        method = method.replace("offsetTables_[offset], M);", "offsetTables_[offset], M - 1);")
    m = 32
    layouts = 4 if options.get("active_cube_rows") else 2
    batch = 16 if options.get("quad_hidden_quant") else 8
    options = {**options, "streaming_pipeline": True, "prefill_rows_32": True}
    header = (1280 * 8, 72, 4096, k, 4, activation_bits, 1280, 8)
    descriptor = projection_descriptor(header, gate_up=gate_up, options=options)
    # Decode the actual serialized descriptor, rather than passing builder tables.
    payload = tuple(v for word in descriptor[HEADER_WORDS:] for v in (word & 0xFFFFFFFF, word >> 32))
    program = r"""#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
constexpr uint32_t POISON=0xdeadbeef;
struct Copy { std::string name; uint32_t source,destination,words; };
std::vector<Copy> copies;
uint32_t barriers=0;
template<class T> struct LocalTensor {
    std::vector<uint32_t>* data; uint32_t base; std::string name;
    LocalTensor operator[](uint32_t i) const { return {data,base+i,name}; }
};
template<class T> struct GlobalTensor {
    std::vector<uint32_t>* data; uint32_t base=0;
    GlobalTensor operator[](uint32_t i) const { return {data,base+i}; }
};
struct Buffer {
    std::string name; std::vector<uint32_t> data=std::vector<uint32_t>(8192,POISON);
    template<class T> LocalTensor<T> Get() { return {&data,0,name}; }
};
template<class T> void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src,uint32_t words) {
    if (barriers || words%8 || dst.base%8 || src.base%8)
        throw std::runtime_error("DMA alignment or final fence violated");
    if(dst.base+words>dst.data->size() || src.base+words>src.data->size())
        throw std::runtime_error("DMA bounds violated");
    copies.push_back({dst.name,src.base,dst.base,words});
    for(uint32_t i=0;i<words;++i) dst.data->at(dst.base+i)=src.data->at(src.base+i);
}
constexpr int PIPE_ALL=0;
template<int> void PipeBarrier() { ++barriers; }
namespace GlmStreamingScaleGather { constexpr uint32_t GROUP_SOURCE_OFFSET=28032; }
"""
    program += f"constexpr uint32_t M={m}, N=128, PRODUCT_LAYOUT_COUNT={layouts}, MAX_K_GROUPS=128, LANES=8;\n"
    program += f"constexpr uint32_t SCALE_INDEX_OFFSET=3072, QUANT_OFFSET_BYTES=({batch}*32*2+{batch})*4;\n"
    program += "struct Projection {\n"
    for name in ("outputOffsets_", "productOffsets_", "scaleCache_", "activationScales_", "quantOffsets_", "mask_"):
        program += f'Buffer {name}{{"{name}"}};\n'
    program += "GlobalTensor<uint32_t> offsetTables_;\n" + method + "};\n"
    program += r"""int main() { try {
    std::vector<uint32_t> input; uint64_t v;
    while(std::cin>>v) input.push_back(static_cast<uint32_t>(v));
    Projection p; p.offsetTables_={&input}; p.PrepareOffsets();
    if(barriers!=1) throw std::runtime_error("missing final DMA fence");
    for(auto c:copies) std::cout<<"DMA "<<c.name<<" "<<c.source<<" "<<c.destination<<" "<<c.words<<"\n";
"""
    # Emit the complete destination allocation, so untouched neighbors are checked.
    for name in ("outputOffsets_", "productOffsets_", "scaleCache_", "activationScales_", "quantOffsets_", "mask_"):
        program += (
            f'std::cout<<"BUFFER {name}"; for(auto value:p.{name}.data) std::cout<<" "<<value; std::cout<<"\\n";\n'
        )
    program += "} catch(const std::exception& e){std::cerr<<e.what();return 1;} }\n"
    path, binary = tmp_path / "prepared.cpp", tmp_path / "prepared"
    path.write_text(program)
    flags = ["GLM_PREPARED_OFFSET_TABLES", "GLM_STREAMING_PIPELINE"]
    if gate_up:
        flags.append("GLM_FUSED_GATE_UP")
    subprocess.run(
        [compiler, "-std=c++17", *("-D" + f for f in flags), str(path), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], input=" ".join(map(str, payload)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    records, buffers = [], {}
    for line in result.stdout.splitlines():
        label, name, *values = line.split()
        if label == "DMA":
            records.append((name, *map(int, values)))
        else:
            assert label == "BUFFER"
            buffers[name] = tuple(map(int, values))
    tables = projection_offset_tables(header, gate_up=gate_up, options=options)
    destinations = ("outputOffsets_", "productOffsets_", "scaleCache_", "activationScales_", "quantOffsets_", "mask_")
    starts = (0, 0, 3072 // 4, (m + 8) * 128, 0, 28032 // 4)
    expected_records, source_offset = [], 0
    for table, name, start in zip(tables, destinations, starts):
        expected = [0xDEADBEEF] * 8192
        expected[start : start + len(table)] = table
        assert buffers[name] == tuple(expected), f"{name} DMA payload or neighbor corrupted"
        if table:
            expected_records.append((name, source_offset, start, len(table)))
        source_offset += len(table)
    assert records == expected_records
    assert source_offset == descriptor[9] == len(payload)


@pytest.mark.parametrize("gate_up", [False, True])
@pytest.mark.parametrize("k,activation_bits", [(2048, 4), (4096, 8)])
@pytest.mark.parametrize("expanded", [False, True])
def test_production_streaming_prepared_dma_order(tmp_path, gate_up, k, activation_bits, expanded):
    _run_prepared_dma(
        tmp_path,
        gate_up=gate_up,
        k=k,
        activation_bits=activation_bits,
        options={
            "active_cube_rows": expanded,
            "quad_hidden_quant": expanded,
            "route_compact_down_scales": True,
            "raw_hidden_scales": True,
            "native_route_columns": True,
            "nz_prefill_accumulator": expanded,
        },
    )


@pytest.mark.parametrize(
    "mutation", ["missing_quant_advance", "wrong_mask_destination", "wrong_source", "unaligned_copy"]
)
def test_production_dma_faults_are_detected(tmp_path, mutation):
    with pytest.raises(AssertionError, match="DMA|mask_"):
        _run_prepared_dma(
            tmp_path, gate_up=True, k=4096, activation_bits=4, options={"quad_hidden_quant": True}, mutation=mutation
        )


def _compiled_entry_guard(tmp_path, *, gate_up, streaming, quant_batch, product_layouts, omit_streaming_length=False):
    """Execute the production entry through its geometry/descriptor admission guards."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires host C++ compiler")
    source = (ROOT / "tools/glm_perf/glm_fused_moe.cpp").read_text()
    start = source.index("  AscendC::InitSocState();")
    end = source.index("  Projection operation;", start)
    entry = source[start:end]
    if omit_streaming_length:
        extension = "+ GlmStreamingScaleGather::ROWS"
        assert extension in entry
        entry = entry.replace(extension, "")
    # Retain the production count declarations too; do not duplicate its entry formula.
    start = source.index("#ifdef GLM_ACTIVE_CUBE_ROWS\nconstexpr uint32_t CUBE_ROW_TILE")
    end = source.index("#ifdef GLM_STREAMING_PIPELINE", start)
    declarations = source[start:end]
    program = r"""#include <cstdint>
#include <iostream>
#include <vector>
#define __gm__
uint32_t initialized=0;
namespace AscendC { void InitSocState() { ++initialized; } }
namespace GlmStreamingScaleGather { constexpr uint32_t ROWS=32; }
"""
    program += "constexpr uint32_t M=32,N=128;\nnamespace GlmFusedQuant {\n"
    program += f"constexpr uint32_t BATCH={quant_batch},ELEMENTS=32*BATCH;\n}}\n"
    program += declarations
    program += "void Entry(void* tiling, bool& accepted) {\n" + entry + "accepted=true;\n}\n"
    program += r"""int main() {
    uint32_t size;
    while(std::cin>>size) {
        std::vector<int64_t> descriptor(size);
        for(auto& word:descriptor) if(!(std::cin>>word)) return 1;
        bool accepted=false;
        Entry(descriptor.data(),accepted);
        std::cout<<accepted<<" "<<initialized<<"\n";
    }
}
"""
    path, binary = tmp_path / "entry.cpp", tmp_path / "entry"
    path.write_text(program)
    flags = ["GLM_PREPARED_OFFSET_TABLES"]
    if gate_up:
        flags.append("GLM_FUSED_GATE_UP")
    if streaming:
        flags.append("GLM_STREAMING_PIPELINE")
    if product_layouts == 4:
        flags.append("GLM_ACTIVE_CUBE_ROWS")
    subprocess.run(
        [compiler, "-std=c++17", *("-D" + flag for flag in flags), str(path), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )

    def run(descriptors):
        # Every case has a complete config header. Physical GM payload bounds are
        # not represented by this ABI and are not inferred from entry acceptance.
        assert all(len(descriptor) >= HEADER_WORDS for descriptor in descriptors)
        serialized = "\n".join(f"{len(d)} " + " ".join(map(str, d)) for d in descriptors)
        result = subprocess.run([str(binary)], input=serialized, capture_output=True, text=True, check=True)
        answers = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
        assert len(answers) == len(descriptors)
        assert [initialized for _, initialized in answers] == list(range(1, len(descriptors) + 1))
        return [bool(accepted) for accepted, _ in answers]

    return run


@pytest.fixture(params=[(g, s, q, p) for g in (False, True) for s in (False, True) for q in (8, 16) for p in (2, 4)])
def entry_guard(request, tmp_path):
    gate_up, streaming, quant_batch, product_layouts = request.param
    options = {
        "streaming_pipeline": streaming,
        "prefill_rows_32": True,
        "quad_hidden_quant": quant_batch == 16,
        "active_cube_rows": product_layouts == 4,
        "native_route_columns": not gate_up,
        "route_compact_down_scales": True,
        "raw_hidden_scales": True,
    }
    run = _compiled_entry_guard(
        tmp_path, gate_up=gate_up, streaming=streaming, quant_batch=quant_batch, product_layouts=product_layouts
    )
    return run, gate_up, options


def test_production_entry_accepts_prepared_descriptors(entry_guard):
    run, gate_up, options = entry_guard
    descriptors = []
    for k, tokens in ((256, 1), (2048, 17), (4096, 640), (2048, 1280)):
        for bits in (2, 3, 4):
            for activation_bits in (4, 8):
                header = (tokens * 8, 72, 4096, k, bits, activation_bits, tokens, 8)
                descriptors.append(projection_descriptor(header, gate_up=gate_up, options=options))
    assert run(descriptors) == [True] * len(descriptors)


def test_production_entry_rejects_malformed_descriptor_headers(entry_guard):
    run, gate_up, options = entry_guard
    header = (1280 * 8, 72, 4096, 2048, 4, 4, 1280, 8)
    descriptor = projection_descriptor(header, gate_up=gate_up, options=options)
    corrupt = []
    # Actual entry reads tag, word count and both reserved fields.
    for index, value in ((8, LAYOUT_TAG + 1), (9, descriptor[9] - 32), (9, descriptor[9] + 32), (10, 1), (11, 1)):
        altered = list(descriptor)
        altered[index] = value
        corrupt.append(altered)
    for index, value in ((0, 0), (3, 255), (4, 1), (5, 16), (6, 1279)):
        altered = list(descriptor)
        altered[index] = value
        corrupt.append(altered)
    assert run([descriptor, *corrupt, descriptor]) == [True, *([False] * len(corrupt)), True]


def test_production_entry_rejects_opposite_descriptor_layout(entry_guard):
    run, gate_up, options = entry_guard
    header = (640 * 8, 72, 4096, 4096, 3, 8, 640, 8)
    descriptor = projection_descriptor(header, gate_up=gate_up, options=options)
    opposite = projection_descriptor(
        header, gate_up=gate_up, options={**options, "streaming_pipeline": not options["streaming_pipeline"]}
    )
    # In the streaming build, this explicitly exercises a legacy descriptor
    # without the new tail and with its original declared length.
    assert run([descriptor, opposite]) == [True, False]


@pytest.mark.parametrize("gate_up", [False, True])
def test_entry_regression_detects_omitted_streaming_length(tmp_path, gate_up):
    run = _compiled_entry_guard(
        tmp_path, gate_up=gate_up, streaming=True, quant_batch=16, product_layouts=4, omit_streaming_length=True
    )
    header = (640, 72, 4096, 2048, 4, 4, 640, 1)
    options = {"streaming_pipeline": True, "prefill_rows_32": True, "quad_hidden_quant": True, "active_cube_rows": True}
    descriptor = projection_descriptor(header, gate_up=gate_up, options=options)
    assert run([descriptor]) == [False]


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
