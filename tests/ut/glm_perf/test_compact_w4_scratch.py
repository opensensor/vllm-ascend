# SPDX-License-Identifier: Apache-2.0
"""Compile the kernel's scratch geometry on CPU and guard W4-only dispatch."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf import build_reconstruction, fused_moe_profile, glm_fused_moe
from tools.glm_perf.build_reconstruction import build
from tools.glm_perf.glm_fused_moe import NativeFusedMoE
from tools.glm_perf.native_checkpoint import LAYOUT, NativeInt4MoEMethod, copy_kernel_bundle, file_digest
from tools.glm_perf.route_columns_probe import run
from vllm_ascend.models.glm5next_w2 import moe

HERE = Path(__file__).resolve().parents[3] / "tools/glm_perf"


def test_actual_cpp_scratch_sizes_retain_readback_and_wide_activation_capacity(tmp_path):
    source = tmp_path / "geometry.cpp"
    source.write_text("""#include "glm_fused_scratch.h"
#include <iostream>
template<unsigned Rows, unsigned WideK, bool Compact> void emit() {
  using S=GlmFusedScratch::Layout<Rows,WideK,Compact>;
  std::cout << Rows << " " << WideK << " " << Compact << " " << S::RAW_BYTES << " "
            << S::DECODED_BYTES << " " << S::GATHERED_BYTES << " " << S::SCALE_PRODUCTS_OFFSET << "\\n";
}
int main() {
 emit<16,64,false>(); emit<16,64,true>(); emit<16,128,false>(); emit<16,128,true>();
 emit<16,256,false>(); emit<16,256,true>(); emit<32,64,false>(); emit<32,64,true>();
}
""")
    binary = tmp_path / "geometry"
    subprocess.run(["c++", "-std=c++17", "-I" + str(HERE), str(source), "-o", str(binary)], check=True)
    rows = [list(map(int, row.split())) for row in subprocess.check_output([str(binary)], text=True).splitlines()]
    for old, new in zip(rows[::2], rows[1::2]):
        m, k = old[:2]
        assert old[3:] == [16448, 65536, 32768, 24576]
        assert new[3] >= m * max(k, 128) // 2  # ActivationWide's largest packing.
        assert new[4] == (65536 if m == 32 else 4096)
        assert new[5:] == [8192, 0]
        assert sum(old[3:6]) - sum(new[3:6]) == (38912 if m == 32 else 100352 if k == 256 else 101376)


@pytest.mark.parametrize(
    "options", [{}, {"output_columns": 128, "tile_pipeline": True, "all_bits": True, "fused_moe": True}]
)
def test_unprepared_scratch_specialization_rejected_before_artifacts(tmp_path, options):
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="prepared fused MoE"):
        build(output, tmp_path, tmp_path, compact_w4_scratch=True, **options)
    assert not output.exists()


@pytest.mark.parametrize("value", [1, "true"])
def test_specialization_requires_a_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="must be boolean"):
        build(tmp_path / "bad", tmp_path, tmp_path, compact_w4_scratch=value)


def test_specialization_rejects_live_decode_tables(tmp_path):
    with pytest.raises(ValueError, match="lookup tables"):
        build(
            tmp_path / "bad",
            tmp_path,
            tmp_path,
            compact_w4_scratch=True,
            weight_decode_lut=True,
            fused_moe=True,
            all_bits=True,
            tile_pipeline=True,
            output_columns=128,
            prepared_weight_layout=True,
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("stage", ["gate", "down"])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_each_stage_selects_only_its_matching_width(stage, bits):
    native = object.__new__(NativeFusedMoE)
    setattr(native, stage + "_kernel", "generic")
    setattr(native, stage + "_w3_kernel", "w3")
    setattr(native, stage + "_w4_kernel", "compact_w4")
    assert native.stage_kernel(stage, bits) == {2: "generic", 3: "w3", 4: "compact_w4"}[bits]


@pytest.mark.parametrize("flag,present", [(True, ()), (True, ("gate_up",)), (False, ("gate_up", "down"))])
def test_partial_or_undeclared_w4_binaries_fail_before_loading(tmp_path, flag, present):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": {"compact_w4_scratch": flag}}))
    for stage in present:
        (tmp_path / f"glm_fused_{stage}_w4.bin").write_bytes(b"partial")
    with pytest.raises(ValueError, match="declared paired"):
        NativeFusedMoE(
            tmp_path,
            namespace="unused",
            activation_bits=4,
            prepared_weight_layout=True,
            launch=lambda *args: None,
            kernel_factory=lambda *args: pytest.fail("partial load"),
        )


def test_helper_rejects_unprepared_layout_before_loading(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": {"compact_w4_scratch": True}}))
    with pytest.raises(ValueError, match="prepared weights"):
        NativeFusedMoE(
            tmp_path,
            namespace="unused",
            activation_bits=4,
            launch=lambda *args: None,
            kernel_factory=lambda *args: pytest.fail("unprepared load"),
        )


def test_paired_stage_files_are_packaged_and_verified(tmp_path):
    bundle, output = tmp_path / "bundle", tmp_path / "copied"
    bundle.mkdir()
    options = dict(fused_moe=True, prepared_weight_layout=True, version=958, compact_w4_scratch=True)
    provenance = {"_build": options, "_helpers": {}}
    files = [
        "glm_reconstruction_bridge_v958.so",
        "glm_fused_gate_up.bin",
        "glm_fused_down.bin",
        "glm_fused_pack.bin",
        "glm_fused_gate_up_w4.bin",
        "glm_fused_down_w4.bin",
    ]
    for name in files:
        (bundle / name).write_bytes(name.encode())
        key = "reconstruction_bridge.cpp" if name.endswith(".so") else name
        provenance[key] = dict(binary_sha256=file_digest(bundle / name))
    (bundle / "provenance.json").write_text(json.dumps(provenance))
    copy_kernel_bundle(bundle, output)
    assert all((output / name).read_bytes() == (bundle / name).read_bytes() for name in files)
    (bundle / "glm_fused_down_w4.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="binary differs"):
        copy_kernel_bundle(bundle, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_frozen_profile_checks_w4_before_loading_device_library(tmp_path, monkeypatch):
    package = "test_w4_frozen_helpers"
    directory = tmp_path / package
    directory.mkdir()
    (directory / "__init__.py").write_text("")
    options = dict(helper_package=package, compact_w4_scratch=True, version=958)
    provenance = {"_build": options, "_helpers": {"__init__.py": hashlib.sha256(b"").hexdigest()}}
    for name in (
        "glm_reconstruction_bridge_v958.so",
        "glm_fused_pack.bin",
        "glm_fused_gate_up.bin",
        "glm_fused_down.bin",
        "glm_fused_gate_up_w4.bin",
        "glm_fused_down_w4.bin",
    ):
        (tmp_path / name).write_bytes(name.encode())
        key = "reconstruction_bridge.cpp" if name.endswith(".so") else name
        provenance[key] = dict(binary_sha256=file_digest(tmp_path / name))
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    (tmp_path / "glm_fused_down_w4.bin").write_bytes(b"changed")
    monkeypatch.setattr(torch.ops, "load_library", lambda *a: pytest.fail("loaded tampered library"))
    with pytest.raises(ValueError, match="specialized kernel"):
        fused_moe_profile.frozen_helper(tmp_path)


def test_full_pipeline_probe_still_requires_explicit_device_gate(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        run(tmp_path / "missing", tmp_path / "missing", tmp_path / "result.json", feature="compact_w4_scratch")


@pytest.mark.parametrize("gate_bits,down_bits", [(4, 4), (3, 4), (4, 2)])
@pytest.mark.parametrize("tokens", [2, 17, 640])
@pytest.mark.parametrize("activation", [4, 8])
def test_complete_host_pipeline_dispatch_keeps_mixed_banks_and_original_geometries(
    tmp_path, monkeypatch, gate_bits, down_bits, tokens, activation
):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": {"compact_w4_scratch": True}}))
    for stage in ("gate_up", "down"):
        (tmp_path / f"glm_fused_{stage}_w4.bin").write_bytes(b"fixture")
    proxy = SimpleNamespace(**vars(torch))
    proxy.device = lambda *args: torch.device("cpu") if args[0] == "npu" else torch.device(*args)
    proxy.npu = SimpleNamespace(current_device=lambda: 0)
    monkeypatch.setattr(glm_fused_moe, "torch", proxy)
    calls = []
    native = NativeFusedMoE(
        tmp_path,
        namespace="fixture",
        activation_bits=activation,
        prepared_weight_layout=True,
        kernel_factory=lambda path, entry: entry,
        launch=lambda kernel, args, blocks: calls.append((kernel, args)),
    )
    native.device = torch.device("cpu")
    x = torch.ones(tokens, 256).half()
    gate, down = (
        torch.zeros(3, n, 256 * bits // 8, dtype=torch.uint8) for n, bits in ((512, gate_bits), (256, down_bits))
    )
    gs, ds = torch.ones(3, 16, 8), torch.ones(3, 8, 8)
    result = native(x, gate, gs, down, ds, torch.ones(tokens, 8), torch.zeros(tokens, 8, dtype=torch.int64))
    expected = [
        "glm_fused_pack_v1",
        "glm_fused_gate_up_w4_v1" if gate_bits == 4 else "glm_fused_gate_up_v1",
        "glm_fused_down_w4_v1" if down_bits == 4 else "glm_fused_down_v1",
    ]
    if tokens > 16:
        expected.append("glm_fused_reduce_v1")
    assert [kernel for kernel, _ in calls] == expected
    assert result.shape == (tokens, 256)
    assert calls[1][1][-1][4] == gate_bits and calls[2][1][-1][4] == down_bits


def test_builder_keeps_generic_and_w3_stages_with_the_new_w4_pair(tmp_path, monkeypatch):
    calls = []

    def fake_compile(command, **kwargs):
        calls.append(command)
        output = command[command.index("-o") + 1] if command[0] == "c++" else command[2]
        Path(output).write_bytes(str(command).encode())

    monkeypatch.setattr(build_reconstruction.subprocess, "run", fake_compile)
    monkeypatch.setattr(
        build_reconstruction.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(tmp_path / "torch_npu/__init__.py")),
    )
    output = tmp_path / "compiled"
    build(
        output,
        tmp_path,
        tmp_path,
        fused_moe=True,
        prepared_weight_layout=True,
        all_bits=True,
        tile_pipeline=True,
        output_columns=128,
        specialize_w3=True,
        compact_w4_scratch=True,
        version=958,
    )
    compiled = [
        command
        for command in calls
        if command[0].endswith("compile-reconstruction") and command[1].endswith("glm_fused_moe.cpp")
    ]
    assert len(compiled) == 6
    for command in compiled:
        filename = Path(command[2]).name
        if "_w4" in filename:
            assert "-DGLM_STATIC_WEIGHT_BITS=4" in command and "-DGLM_COMPACT_W4_SCRATCH" in command
        else:
            assert "-DGLM_COMPACT_W4_SCRATCH" not in command
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["_build"]["compact_w4_scratch"] is True
    assert provenance["glm_fused_scratch.h"]["source_sha256"] == file_digest(HERE / "glm_fused_scratch.h")


def test_permanent_moe_calls_the_loaded_method_without_resolving_legacy_fp64(monkeypatch):
    calls = []

    class Native:
        def __call__(self, *args):
            calls.append(args)
            return args[0].float()

    bank = SimpleNamespace(
        native_weight_layout=LAYOUT,
        local_expert_offset=0,
        gate_up_packed_bank=object(),
        gate_up_scale_bank=object(),
        down_packed_bank=object(),
        down_scale_bank=object(),
    )
    owner = moe.Glm5NextW2MoE(w2_experts=bank)
    # This is the loader's composition seam, consumed by the actual .method
    # property and routed forward, rather than a call to the native method alone.
    owner._method = NativeInt4MoEMethod(Native())
    monkeypatch.setattr(moe, "resolve_w2_moe_method", lambda: pytest.fail("legacy FP64 scheme selected"))
    monkeypatch.setattr(moe, "_ep_rank_size", lambda: (0, 1))
    x = torch.ones(2, 256).half()
    output = owner.routed_experts_forward(x, torch.zeros(2, 8, dtype=torch.int64), torch.ones(2, 8))
    assert output.dtype == torch.float32 and torch.equal(output, x.float())
    assert len(calls) == 1 and calls[0][1] is bank.gate_up_packed_bank
