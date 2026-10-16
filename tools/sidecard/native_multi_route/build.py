# SPDX-License-Identifier: Apache-2.0
"""Build the R1 fixed-route path from authenticated v1140 source bytes only.

The six source edits change admission/layout thresholds together. Arithmetic,
weight packing, clipping, and stable expert/slot reduction remain frozen.
"""

import argparse
import copy
import difflib
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .contract import workspace_bytes

ORIGINAL_PROVENANCE = "9c40e60ba04bbfcabb87cbdce21828dcb018c5ca0006ee60a8c543ec8ac7d690"
CLIPPED_SOURCE = "520cc1eb25e2e797c266257e6f91b6c9e3f364df470860fa4f78f00dcf5725a5"
HELPER_PACKAGE = "glm_sidecard_multi_route_v1_helpers"
PATCHES = {
    "native_sources/glm_fused_moe.cpp": ("constexpr uint32_t BULK_TOKENS = 16;", "constexpr uint32_t BULK_TOKENS = 0;"),
    "native_sources/glm_fused_pack.cpp": ("constexpr int64_t BULK_TOKENS = 16;", "constexpr int64_t BULK_TOKENS = 0;"),
    "native_sources/glm_fused_route_input.cpp": ("config[6] <= 16", "config[6] <= 0"),
    "native_sources/glm_fused_reduce.cpp": ("config[6] <= 16", "config[6] <= 0"),
    "glm_fused_moe.py": ("FUSED_REDUCTION_TOKENS = 16", "FUSED_REDUCTION_TOKENS = 0"),
    "fused_offset_tables.py": ("BULK_TOKENS = 16", "BULK_TOKENS = 0"),
}
COMMON_DEFINES = (
    "STREAMING_PIPELINE",
    "PREPARED_WEIGHT_LAYOUT",
    "FP16_WEIGHT_SCALES",
    "PAIR_SCALE_GROUPS",
    "PREFILL_WEIGHT_CACHE",
    "VECTOR_SCALE_PRODUCTS",
    "GROUP_MAJOR_DOWN_SCALES",
    "PREFILL_ROWS_32",
    "DIRECT_W4_L1",
    "PREPARED_OFFSET_TABLES",
    "PRODUCT_PIPE_EVENTS",
    "FP16_ROUTE_WORKSPACE",
    "PAIR_PREFILL_SCALE_GROUPS",
    "ROUTE_PACKED_DOWN",
    "COMPACT_DOWN_SCALES",
    "RAW_HIDDEN_SCALES",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("bundle path escapes root")
    return path


def patch_text(relative, source):
    before, after = PATCHES[relative]
    if source.count(before) != 1:
        raise ValueError("frozen patch anchor differs: " + relative)
    return source.replace(before, after)


def authenticate(bundle, clipped, native_sources=None):
    """Reject changes before source execution or compiler invocation."""
    bundle, clipped = Path(bundle), Path(clipped)
    if digest(bundle / "provenance.json") != ORIGINAL_PROVENANCE:
        raise ValueError("unsupported original native provenance")
    provenance = json.loads((bundle / "provenance.json").read_bytes())
    if digest(clipped) != CLIPPED_SOURCE:
        raise ValueError("clipped source provenance differs")
    options = provenance["_build"]
    if options["version"] != 1140 or options["namespace"] != "glm_reconstruction_v1140":
        raise ValueError("bridge ABI differs from qualified v1140")
    required = ("streaming_pipeline", "fp16_route_workspace", "native_route_columns", "prepared_weight_layout")
    if not all(options.get(name) is True for name in required):
        raise ValueError("incompatible native layout/route ABI")
    files = dict(provenance["_streaming_sources"])
    for name, checksum in provenance["_helpers"].items():
        files[options["helper_package"] + "/" + name] = checksum
    bridge = f"glm_reconstruction_bridge_v{options['version']}.so"
    files[bridge] = provenance["reconstruction_bridge.cpp"]["binary_sha256"]
    for name, checksum in files.items():
        path = (
            contained(Path(native_sources), Path(name).name)
            if native_sources is not None and name.startswith("native_sources/")
            else contained(bundle, name)
        )
        if digest(path) != checksum:
            raise ValueError("frozen source/helper/bridge changed: " + name)
    # Authenticate reused stage binaries as well; even though candidate stages
    # are rebuilt, a successful gate must have an immutable padded comparator.
    for name, record in provenance.items():
        if name.endswith(".bin") and digest(bundle / name) != record["binary_sha256"]:
            raise ValueError("frozen comparator binary changed: " + name)
    return provenance


def commands(output, cann):
    """Exact v1140 defines for paired mixed-W2/W3 and specialized W4 stages."""
    native = output / "native_sources"
    compiler = output / "compile-reconstruction"
    result = [
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(native / "compile_reconstruction.cpp"),
            "-I" + str(cann / "include"),
            "-L" + str(cann / "lib64"),
            "-lacl_rtc",
            "-lascendcl",
            "-Wl,-rpath," + str(cann / "lib64"),
            "-o",
            str(compiler),
        ]
    ]
    base = [
        str(compiler),
        "",
        "",
        "--npu-arch=dav-2002",
        "--sysroot=" + str(cann / "tools/hcc/sysroot"),
        "-isystem" + str(cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"),
        "-isystem" + str(cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu"),
        "-I" + str(native),
    ]
    stages = [(stage, bits) for bits in (None, 4) for stage in ("gate_up", "down")]
    for stage, bits in stages:
        suffix = "_w4" if bits else ""
        command = list(base)
        command[1:3] = [str(native / "glm_fused_moe.cpp"), str(output / f"glm_fused_{stage}{suffix}.bin")]
        command += ["-DGLM_" + name for name in COMMON_DEFINES]
        if stage == "gate_up":
            command += [
                "-DGLM_" + name
                for name in (
                    "FUSED_GATE_UP",
                    "ROUTE_PACKED_INPUT",
                    "DIRECT_HIDDEN_GATHER",
                    "SHARE_GATE_UP_INPUT",
                    "CACHE_GATE_UP_ACTIVATIONS",
                    "QUAD_HIDDEN_QUANT",
                    "HIDDEN_QUANT_BATCH=16",
                )
            ]
        else:
            command += ["-DGLM_BULK_ROUTE_STORE", "-DGLM_NATIVE_ROUTE_COLUMNS"]
        if bits:
            command += ["-DGLM_STATIC_WEIGHT_BITS=4", "-DGLM_COMPACT_W4_SCRATCH"]
        result.append(command)
    for stage, defines in (
        ("pack", ("RAW_INPUT_SCALES",)),
        ("route_input", ("RAW_INPUT_SCALES",)),
        ("reduce", ("FP16_ROUTE_WORKSPACE", "NATIVE_ROUTE_COLUMNS")),
    ):
        command = list(base)
        command[1:3] = [str(native / f"glm_fused_{stage}.cpp"), str(output / f"glm_fused_{stage}.bin")]
        command += ["-DGLM_" + name for name in defines]
        result.append(command)
    return result


def build(bundle, clipped, output, cann, native_sources=None, *, prepare_only=False):
    provenance = authenticate(bundle, clipped, native_sources)
    bundle, output, cann = Path(bundle).resolve(), Path(output).resolve(), Path(cann).resolve()
    output.mkdir(exist_ok=False)
    original_helpers = provenance["_build"]["helper_package"]
    package = output / HELPER_PACKAGE
    package.mkdir()
    native = output / "native_sources"
    native.mkdir()
    for relative in provenance["_streaming_sources"]:
        if Path(relative).parent != Path("native_sources"):
            raise ValueError("unexpected frozen native directory")
        source = bundle / relative if native_sources is None else Path(native_sources) / Path(relative).name
        shutil.copyfile(source, output / relative)
    for name in provenance["_helpers"]:
        shutil.copyfile(bundle / original_helpers / name, package / name)
    shutil.copyfile(clipped, native / "glm_fused_moe.cpp")
    audit = []
    for relative in PATCHES:
        target = output / relative if relative.startswith("native_sources/") else package / relative
        before = target.read_text()
        after = patch_text(relative, before)
        target.write_text(after)
        audit.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), relative, relative))
    (output / "candidate.patch").write_text("".join(audit))
    bridge = f"glm_reconstruction_bridge_v{provenance['_build']['version']}.so"
    shutil.copyfile(bundle / bridge, output / bridge)
    manifest = dict(
        schema_version=1,
        candidate="multi-route-r1-fixed-slots-v1",
        original_provenance_sha256=ORIGINAL_PROVENANCE,
        clipped_source_sha256=CLIPPED_SOURCE,
        bridge_binary_sha256=digest(output / bridge),
        bridge_namespace=provenance["_build"]["namespace"],
        swiglu_limit=10.0,
        activation_bits=4,
        weight_bits=[2, 3, 4],
        rows=1,
        top_k=8,
        workspace_bytes=workspace_bytes(),
        patch_sha256=digest(output / "candidate.patch"),
        hardware_validated=False,
        build_complete=False,
        source_sha256={str(p.relative_to(output)): digest(p) for root in (native, package) for p in root.iterdir()},
        commands=commands(output, cann),
    )
    receipt = output / "multi-route-build.json"
    receipt.write_text(json.dumps(manifest, indent=2) + "\n")
    if prepare_only:
        return manifest
    for command in manifest["commands"]:
        subprocess.run(command, check=True)
    candidate = copy.deepcopy(provenance)
    candidate["_build"]["helper_package"] = HELPER_PACKAGE
    candidate["_helpers"] = {p.name: digest(p) for p in package.iterdir()}
    candidate["_streaming_sources"] = {str(p.relative_to(output)): digest(p) for p in native.iterdir()}
    binaries = {}
    for path in output.glob("glm_fused_*.bin"):
        source = "glm_fused_moe.cpp" if any(stage in path.name for stage in ("gate_up", "down")) else path.stem + ".cpp"
        candidate[path.name] = dict(source_sha256=digest(native / source), binary_sha256=digest(path))
        binaries[path.name] = digest(path)
    (output / "provenance.json").write_text(json.dumps(candidate, indent=2) + "\n")
    manifest.update(build_complete=True, binaries=binaries, provenance_sha256=digest(output / "provenance.json"))
    receipt.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--clipped-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument(
        "--native-sources", type=Path, help="authenticated native_sources directory if stored separately"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="write authenticated sources/audit/commands without compiling"
    )
    args = parser.parse_args()
    build(args.bundle, args.clipped_source, args.output, args.cann, args.native_sources, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
