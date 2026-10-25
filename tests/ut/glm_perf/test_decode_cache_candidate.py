# SPDX-License-Identifier: Apache-2.0
"""Authenticate/freeze v984 and execute the actual transformed Run dispatch."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.glm_perf import decode_cache_candidate as candidate

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "artifacts/glm-perf-310p/streaming-upgrade/T22/inputs"
STUBS = Path(__file__).parent / "decode_cache_candidate_cpu_stubs"


def plan(tmp_path, inputs=INPUTS, **kwargs):
    toolchain = json.loads((INPUTS / "toolchain.json").read_text())
    return candidate.candidate_plan(inputs, tmp_path / "snapshot", tmp_path / "build", toolchain, **kwargs)


def test_dry_run_preserves_every_old_math_flag_and_records_all_commands(tmp_path):
    record, files = plan(tmp_path)
    baseline = json.loads((INPUTS / "reference-provenance.json").read_text())
    for name, value in baseline["_build"].items():
        if name not in ("namespace", "version", "helper_package"):
            assert record["_build"][name] == value
    assert record["_build"]["vector_scale_products"] is False
    assert not record["sdk_compiled"] and not record["native_qualified"] and not record["version_reserved"]
    assert not list(tmp_path.iterdir())
    assert len(record["compile_argv"]) == 6
    for command in record["compile_argv"][1:5]:
        assert command[-1] == "-DGLM_DECODE_END_CACHE"
        assert "-DGLM_PREFILL_ROWS_32" not in command and "-DGLM_VECTOR_SCALE_PRODUCTS" not in command
        assert "-DGLM_NATIVE_WIDE_CUBE_K=128" in command
        assert "-DGLM_FP16_WEIGHT_SCALES" in command
    assert len(record["reuse_assets"]) == 4
    assert candidate.digest(files["source/glm_decode_end_cache.h"]) == candidate.QUALIFIED_HELPER_SHA256
    transformed = files["source/glm_fused_moe.cpp"].decode()
    for old, new in (
        (candidate.INCLUDE_ANCHOR, candidate.INCLUDE_ANCHOR + candidate.INCLUDE_EXTENSION),
        (candidate.PREPARE_ANCHOR, candidate.PREPARE_ANCHOR + candidate.PREPARE_EXTENSION),
        (candidate.READ_ANCHOR, candidate.READ_REPLACEMENT),
    ):
        transformed = candidate.replace_once(transformed, new, old)
    assert transformed.encode() == (INPUTS / "source/glm_fused_moe.cpp").read_bytes()
    assert plan(tmp_path)[0] == record


def test_freeze_is_append_only_and_never_creates_future_build(tmp_path):
    record, files = plan(tmp_path)
    path = candidate.freeze_candidate(record, files)
    assert json.loads(path.read_text()) == record
    assert not (tmp_path / "build").exists()
    for name, sha in record["input_sha256"].items():
        assert candidate.digest((path.parent / name).read_bytes()) == sha
    with pytest.raises(FileExistsError):
        candidate.freeze_candidate(record, files)


def test_planned_remote_argv_keeps_local_freeze_offline(tmp_path):
    remote = Path("/srv/ai/isolated-decode/frozen-inputs")
    record, files = plan(tmp_path, execution_snapshot_dir=remote)
    assert record["snapshot_dir"] == str(tmp_path / "snapshot")
    assert record["execution_snapshot_dir"] == str(remote)
    assert record["compile_argv"][1][1] == str(remote / "source/glm_fused_moe.cpp")
    assert all(item["source"].startswith(str(remote)) for item in record["assembly_files"])
    candidate.freeze_candidate(record, files)
    assert (tmp_path / "snapshot/candidate-plan.json").is_file()
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("target", ["source", "provenance", "builder", "scratch", "router", "helper", "asset", "extra"])
def test_corrupt_or_added_inputs_fail_closed_without_writes(tmp_path, target):
    inputs = tmp_path / "inputs"
    shutil.copytree(INPUTS, inputs)
    names = {
        "source": "source/glm_fused_moe.cpp",
        "provenance": "reference-provenance.json",
        "builder": "source/build_reconstruction.py",
        "scratch": "source/glm_fused_scratch.h",
        "router": "source/glm_route_input_layout.h",
        "helper": "helpers/glm_fused_moe.py",
        "asset": "assets/glm_fused_pack.bin",
        "extra": "source/kernel_operator.h",
    }
    p = inputs / names[target]
    p.write_bytes(p.read_bytes() + b"\n" if p.exists() else b"untrusted SDK override")
    with pytest.raises(ValueError):
        plan(tmp_path, inputs=inputs)
    assert not (tmp_path / "snapshot").exists() and not (tmp_path / "build").exists()


@pytest.mark.parametrize("version", [984, 0, True])
def test_invalid_version_or_scale_contract_rejected(tmp_path, version):
    with pytest.raises(ValueError):
        plan(tmp_path, version=version)


def test_wrong_scale_marker_rejected(tmp_path):
    with pytest.raises(ValueError, match="marker"):
        plan(tmp_path, scale_layout="fp16_rounded_fp32_v1")


@pytest.mark.parametrize(
    "field,value",
    [
        ("cxx11_abi", 0),
        ("cxx11_abi", True),
        ("bridge_libraries", []),
        ("bridge_includes", ["relative/include"]),
        ("cann_root", "relative/sdk"),
    ],
)
def test_incomplete_or_incompatible_toolchain_is_rejected(tmp_path, field, value):
    toolchain = json.loads((INPUTS / "toolchain.json").read_text())
    toolchain[field] = value
    with pytest.raises(ValueError):
        candidate.candidate_plan(INPUTS, tmp_path / "snapshot", tmp_path / "build", toolchain)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("count", [0, 2])
def test_exact_anchor_cardinality_is_required(count):
    with pytest.raises(ValueError, match="exactly one"):
        candidate.replace_once("anchor" * count, "anchor", "replacement")


@pytest.mark.parametrize("mutation", ["file", "argv"])
def test_freeze_identity_tampering_rejected_before_writes(tmp_path, mutation):
    record, files = plan(tmp_path)
    if mutation == "file":
        files["source/glm_fused_moe.cpp"] += b"\n"
    else:
        record["compile_argv"][1].append("-ffast-math")
    with pytest.raises(ValueError, match="identity changed"):
        candidate.freeze_candidate(record, files)
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize("key", ["sdk_compiled", "native_qualified", "admissible_native_candidate", "version_reserved"])
def test_freeze_cannot_claim_unexecuted_qualification(tmp_path, key):
    record, files = plan(tmp_path)
    record[key] = True
    with pytest.raises(ValueError, match="pending/offline"):
        candidate.freeze_candidate(record, files)
    assert not (tmp_path / "snapshot").exists()


RUN_MODEL = r"""
#include "glm_decode_end_cache.h"
#include <array>
#include <algorithm>
#include <iostream>
using namespace AscendC;
constexpr uint32_t M = 16, N = 128, MAX_ROWS = 15, BULK_TOKENS = 16;
struct Buffer {
  alignas(32) std::array<uint8_t, 32768> data{};
  template <class T>
  LocalTensor<T> Get() {
    return {data.data(), static_cast<uint32_t>(data.size())};
  }
};
struct Projection {
  int64_t rows_, experts_, n_, k_, bits_, activationBits_, tokens_, topK_, firstToken_;
  GlobalTensor<int8_t> low_, high_, inputLow_, inputHigh_;
  GlobalTensor<uint8_t> codes_;
  GlobalTensor<float> xs_, inputScales_, weights_, y_;
  GlobalTensor<half> scales_, routedHalf_;
  GlobalTensor<int64_t> ends_, order_;
  GlobalTensor<uint32_t> offsetTables_;
  Buffer mask_, reduction_, products_, outputOffsets_;
  bool cacheActive_ = false;
  void Allocate() { mask_.data.fill(0xa5); }
  void PrepareOffsets() {
    context->caching = false;
#ifdef GLM_DECODE_END_CACHE
    context->caching = tokens_ <= BULK_TOKENS;
#endif
  }
  void PrepareExpert(int64_t expert, int64_t tile) { Event(4, expert, tile, 0, 0); }
  void Event(int64_t kind, int64_t expert, int64_t tile, int64_t row, uint32_t count) {
    context->trace.insert(context->trace.end(), {kind, expert, tile, row, count});
    // Simulate the audited W2/W3 live writer bound, preserving all retained ends.
    for (uint32_t byte = 0; byte < 16384; ++byte) mask_.data[byte] ^= 0x3c;
  }
  void GateUp(int64_t e, int64_t t, int64_t r, uint32_t c) { Event(1, e, t, r, c); }
  void Accumulate(int64_t e, int64_t t, int64_t r, uint32_t c) { Event(2, e, t, r, c); }
  void Combine(int64_t t, int64_t r, uint32_t c) { Event(3, 0, t, r, c); }
  LocalTensor<float> LowProducts() { return products_.Get<float>(); }
  // ACTUAL_RUN
};
int main(int argc, char** argv) {
  try {
    if (argc != 6) return 2;
    const uint32_t experts = std::stoul(argv[1]), tokens = std::stoul(argv[2]), pattern = std::stoul(argv[3]),
                   bits = std::stoul(argv[4]), activationBits = std::stoul(argv[5]);
    alignas(32) std::array<int64_t, 288> ends{};
    std::vector<int64_t> order(tokens * 8);
    uint64_t hash = 1469598103934665603ULL, totalGM = 0, totalUB = 0, totalDMA = 0;
    for (uint32_t replay = 0; replay < 3; ++replay) {
      std::vector<std::pair<uint32_t, int64_t>> pairs;
      for (uint32_t r = 0; r < tokens * 8; ++r)
        pairs.push_back(
            {pattern == 2 ? experts : (pattern == 1 ? (replay % 2 ? experts - 1 : 0) : (r + replay) % experts), r});
      std::stable_sort(pairs.begin(), pairs.end());
      for (uint32_t r = 0; r < pairs.size(); ++r) order[r] = pairs[r].second;
      for (uint32_t e = 0; e < experts; ++e)
        ends[e] = std::count_if(pairs.begin(), pairs.end(), [&](auto pair) { return pair.first <= e; });
      const auto originalEnds = ends;
      for (uint32_t block = 0; block < 8; ++block) {
        Context state;
        state.ends = reinterpret_cast<uint8_t*>(ends.data());
        state.experts = experts;
        state.routes = tokens * 8;
        state.block = block;
        context = &state;
        Projection p;
        int64_t config[8] = {tokens * 8, experts, 4096, 2048, bits, activationBits, tokens, 8};
        std::array<uint32_t, 8192> offsets{};
        p.Run(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ends.data(), order.data(),
              nullptr, nullptr, config, offsets.data());
        for (int64_t event : state.trace) {
          hash ^= static_cast<uint64_t>(event);
          hash *= 1099511628211ULL;
        }
        totalGM += state.gmReads;
        totalUB += state.ubReads;
        totalDMA += state.dmas;
        Require(ends == originalEnds, "Run changed input ends");
        if (state.caching) {
          Require(state.cacheFences == 2 && !state.pendingBytes, "Run cache prologue not completed");
          for (uint32_t e = 0; e < experts; ++e) {
            int64_t value;
            std::memcpy(&value, p.mask_.data.data() + 24640 + e * 8, 8);
            Require(value == ends[e], "Run retained stale metadata");
          }
          for (uint32_t b = 16384; b < 32768; ++b)
            if (b < 24640 || b >= 24640 + experts * 8) Require(p.mask_.data[b] == 0xa5, "Run cache neighbor changed");
        }
      }
    }
    std::cout << hash << " " << totalGM << " " << totalUB << " " << totalDMA << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
"""


@pytest.fixture(scope="module")
def actual_run_programs(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires local host C++ compiler")
    directory = tmp_path_factory.mktemp("decode-cache-run")
    source = candidate.transform_source((INPUTS / "source/glm_fused_moe.cpp").read_bytes()).decode()
    start = source.index("  __aicore__ inline void Run(")
    end = source.index("\n private:", start)
    program = RUN_MODEL.replace("  // ACTUAL_RUN", source[start:end])
    cpp = directory / "run.cpp"
    cpp.write_text(program)
    paths = {}
    for gate in (False, True):
        for cache in (False, True):
            binary = directory / f"run-{int(gate)}-{int(cache)}"
            flags = [
                "GLM_PREPARED_OFFSET_TABLES",
                "GLM_PREFILL_WEIGHT_CACHE",
                "GLM_FP16_WEIGHT_SCALES",
                "GLM_FP16_ROUTE_WORKSPACE",
            ]
            if gate:
                flags.append("GLM_FUSED_GATE_UP")
            else:
                flags.append("GLM_NATIVE_ROUTE_COLUMNS")
            if cache:
                flags.append("GLM_DECODE_END_CACHE")
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    f"-I{STUBS}",
                    f"-I{ROOT / 'tools/glm_perf'}",
                    *("-D" + f for f in flags),
                    str(cpp),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            paths[gate, cache] = binary
    return paths


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("experts", [1, 3, 4, 5, 71, 72, 73, 287, 288])
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 15, 16, 17])
@pytest.mark.parametrize("pattern", [0, 1, 2])
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("activation_bits", [4, 8])
def test_actual_transformed_run_matches_no_cache_trace_across_replays(
    actual_run_programs, gate, experts, tokens, pattern, bits, activation_bits
):
    rows = []
    for cache in (False, True):
        result = subprocess.run(
            [
                str(actual_run_programs[gate, cache]),
                str(experts),
                str(tokens),
                str(pattern),
                str(bits),
                str(activation_bits),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rows.append(tuple(map(int, result.stdout.split())))
    reference, cached = rows
    assert (
        reference[0] == cached[0]
    )  # Exact original dispatch order/counts; arithmetic is preserved by source identity.
    if tokens <= 16:
        assert cached[1] == experts % 4 * 8 * 3
        assert cached[2] == reference[1] and reference[2:] == (0, 0)
        assert cached[3] == (8 * 3 if experts >= 4 else 0)
    else:
        assert cached == reference  # Accidental bulk use keeps the original GM path.
