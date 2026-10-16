# SPDX-License-Identifier: Apache-2.0
"""Freeze an isolated v984 decode-cache candidate; never invoke a compiler/device."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

REFERENCE_SOURCE_SHA256 = "7cac009f982c14533b83c2c764295f7ed80a85293602a9e010cb27e5f8319921"
REFERENCE_PROVENANCE_SHA256 = "1efedd4c28095ff6c7040d5f9110210c24e0a92f2794f688e8cd5b6312405a7e"
REFERENCE_BUILDER_SHA256 = "cded6f5cd62088c6184b2f0ebea73ab389ef0d8ccd7e88293025b79872a41175"
QUALIFIED_HELPER_SHA256 = "257bc333d56e6f7a7c7b304cfc1d5666f5571c2b5cde04d423d94064179cad08"
QUALIFIED_ROUTE_CACHE_SHA256 = "4028d822e8a753a01056e28768c470b305cfd093949839d19adff894ed1cb0f3"
SOURCE_HASHES = (
    ("glm_fused_moe.cpp", REFERENCE_SOURCE_SHA256),
    ("build_reconstruction.py", REFERENCE_BUILDER_SHA256),
    ("compile_reconstruction.cpp", "e7df352d8e98426001f43dd17116443c72ce9eafa9d7a07930329278d0c3447e"),
    ("reconstruction_bridge.cpp", "08906a208fab23cc9a87e5a18e624d7a64e6458f84f022f7039acf6d6a5b57e2"),
    ("glm_fused_quantize.h", "35980321f8df37ec1f05c431bfd51c38c34682cb3435a91a1e240eee4f654162"),
    ("glm_fused_scratch.h", "e5dae080fc2736005c1b91927fee04bd4ab796f1b4eafaba497a58ee0fbbeac9"),
    ("glm_route_input_layout.h", "6c97df7f5f78ba60ce5da617c975491e6c8926a488516d1ca4f6f0bc59bd331f"),
)
FP16_SCALE_LAYOUT = "fp16_storage_fp32_compute_v1"
DEFAULT_VERSION = 1150
STAGES = (("gate_up", None), ("down", None), ("gate_up", 4), ("down", 4))
REUSE_ASSETS = ("glm_fused_pack.bin", "glm_fused_reduce.bin", "glm_w4a8_pack.bin", "glm_w4a8_matmul.bin")
REUSE_PROVENANCE_KEYS = (
    "glm_fused_pack.bin",
    "glm_fused_reduce.bin",
    "glm_w4a8_pack.cpp",
    "glm_lowbit_a8_pipeline.cpp",
)
INCLUDE_ANCHOR = '#include "glm_route_input_layout.h"\n'
INCLUDE_EXTENSION = """#ifdef GLM_DECODE_END_CACHE
  #if !defined(GLM_PREPARED_WEIGHT_LAYOUT) || defined(GLM_PREFILL_ROWS_32) || defined(GLM_WEIGHT_DECODE_LUT) || \\
      defined(GLM_VECTOR_SCALE_PRODUCTS)
    #error "decode end cache requires the frozen prepared M16/no-LUT scalar-scale schedule"
  #endif
  #include "glm_decode_end_cache.h"
#endif
"""
PREPARE_ANCHOR = "    PrepareOffsets();\n"
PREPARE_EXTENSION = """#ifdef GLM_DECODE_END_CACHE
    // Refresh every launch/replay; retain this existing mask subview across all tiles.
    const auto cachedEnds = mask_.Get<int64_t>()[GlmDecodeEndCache::END_OFFSET / sizeof(int64_t)];
    if (tokens_ <= BULK_TOKENS) {
      if (experts_ < 1 || experts_ > GlmDecodeEndCache::MAX_EXPERTS) return;
      GlmDecodeEndCache::Prepare(cachedEnds, ends_, experts_);
    }
#endif
"""
READ_ANCHOR = "          const int64_t end = ends_.GetValue(expert);\n"
READ_REPLACEMENT = """#ifdef GLM_DECODE_END_CACHE
          const int64_t end = tokens_ <= BULK_TOKENS
                                  ? GlmDecodeEndCache::Read(cachedEnds, expert)
                                  : ends_.GetValue(expert);
#else
          const int64_t end = ends_.GetValue(expert);
#endif
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("source substitution requires exactly one qualified anchor")
    return source.replace(old, new, 1)


def transform_source(source):
    if digest(source) != REFERENCE_SOURCE_SHA256:
        raise ValueError("decode candidate requires the exact frozen v984 source SHA256")
    text = source.decode()
    text = replace_once(text, INCLUDE_ANCHOR, INCLUDE_ANCHOR + INCLUDE_EXTENSION)
    text = replace_once(text, PREPARE_ANCHOR, PREPARE_ANCHOR + PREPARE_EXTENSION)
    text = replace_once(text, READ_ANCHOR, READ_REPLACEMENT)
    # Reverse all three bounded edits to establish byte preservation everywhere else.
    original = replace_once(text, INCLUDE_ANCHOR + INCLUDE_EXTENSION, INCLUDE_ANCHOR)
    original = replace_once(original, PREPARE_ANCHOR + PREPARE_EXTENSION, PREPARE_ANCHOR)
    original = replace_once(original, READ_REPLACEMENT, READ_ANCHOR)
    if original.encode() != source:
        raise ValueError("candidate changed unrelated v984 data/math")
    return text.encode()


def validate_reference(source, provenance_bytes, scale_layout):
    if scale_layout != FP16_SCALE_LAYOUT:
        raise ValueError("candidate requires the permanent FP16 scale-storage marker")
    if digest(source) != REFERENCE_SOURCE_SHA256 or digest(provenance_bytes) != REFERENCE_PROVENANCE_SHA256:
        raise ValueError("frozen v984 source/provenance identity mismatch")
    provenance = json.loads(provenance_bytes)
    options = provenance["_build"]
    if not (
        options["version"] == 984
        and options["namespace"] == "glm_reconstruction_v984"
        and options["fp16_weight_scales"]
        and options["prepared_weight_layout"]
        and options["prepared_offset_tables"]
        and not options["vector_scale_products"]
        and not options["prefill_rows_32"]
        and not options["weight_decode_lut"]
        and options["wide_cube_k"] == 128
    ):
        raise ValueError("candidate requires the audited M16 prepared/no-LUT v984 layout")
    return provenance


def validate_toolchain(toolchain):
    if set(toolchain) != {"cann_root", "cxx", "bridge_includes", "bridge_libraries", "cxx11_abi"}:
        raise ValueError("toolchain must provide complete explicit compiler/include/library/ABI settings")
    if not isinstance(toolchain["cxx"], str) or not toolchain["cxx"] or type(toolchain["cxx11_abi"]) is not int:
        raise ValueError("invalid host compiler/ABI")
    if toolchain["cxx11_abi"] != 1:
        raise ValueError("host ABI must match the frozen __cxx11 NativeKernel signature")
    for name in ("cann_root", "bridge_includes", "bridge_libraries"):
        values = [toolchain[name]] if name == "cann_root" else toolchain[name]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) or not Path(v).is_absolute() for v in values)
        ):
            raise ValueError("toolchain paths must be explicit absolute paths")


def _eval_argv(node, bindings):
    # Only evaluate argument-list expressions from the hash-pinned old builder;
    # never import/execute its filesystem, subprocess, torch or device behavior.
    expression = ast.Expression(node)
    for item in ast.walk(expression):
        if isinstance(item, ast.Call) and not (isinstance(item.func, ast.Name) and item.func.id == "str"):
            raise ValueError("unexpected function in frozen compiler argument expression")
        if isinstance(item, ast.Attribute):
            raise ValueError("unexpected attribute in frozen compiler argument expression")
    return eval(compile(expression, "<frozen-v984-argv>", "eval"), {"__builtins__": {}, **bindings})


def compiler_commands(builder_bytes, options, snapshot_dir, build_dir, version, toolchain):
    """Derive actual four stage argv from the frozen old generator, without running it."""
    if digest(builder_bytes) != REFERENCE_BUILDER_SHA256:
        raise ValueError("compiler argv requires the qualified frozen v984 builder")
    validate_toolchain(toolchain)
    tree = ast.parse(builder_bytes.decode())
    lists = [
        n.args[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "subprocess"
        and n.func.attr == "run"
        and n.args
        and isinstance(n.args[0], ast.List)
    ]
    stage_nodes = [
        n
        for n in lists
        if any(isinstance(x, ast.Constant) and x.value == "-DGLM_PREPARED_WEIGHT_LAYOUT" for x in ast.walk(n))
    ]
    host_nodes = [n for n in lists if any(isinstance(x, ast.Constant) and x.value == "-lacl_rtc" for x in ast.walk(n))]
    option_nodes = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "options" for t in n.targets)
    ]
    if len(stage_nodes) != 1 or len(host_nodes) != 1 or len(option_nodes) != 1:
        raise ValueError("ambiguous compiler command construction in frozen builder")
    source_dir, build = Path(snapshot_dir) / "source", Path(build_dir)
    cann = Path(toolchain["cann_root"])
    bindings = {
        **options,
        "str": str,
        "HERE": source_dir,
        "cann_root": cann,
        "compiler": build / "compile-reconstruction",
    }
    bindings["target"] = cann / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    bindings["options"] = _eval_argv(option_nodes[0], bindings)
    host = _eval_argv(host_nodes[0], bindings)
    host[0] = toolchain["cxx"]
    commands = [host]
    for stage, precision in STAGES:
        suffix = f"_w{precision}" if precision else ""
        stage_bindings = {
            **bindings,
            "stage": stage,
            "precision": precision,
            "source": source_dir / "glm_fused_moe.cpp",
            "output": build / f"glm_fused_{stage}{suffix}.bin",
        }
        argv = _eval_argv(stage_nodes[0], stage_bindings)
        argv.append("-DGLM_DECODE_END_CACHE")
        commands.append(argv)
    # Bridge paths are supplied explicitly, without importing the serving torch
    # environment. Its ABI=1 is established by the frozen NativeKernel symbols.
    libraries = toolchain["bridge_libraries"]
    commands.append(
        [
            toolchain["cxx"],
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"-DGLM_RECONSTRUCTION_NAMESPACE=glm_reconstruction_v{version}",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            *[f"-I{p}" for p in toolchain["bridge_includes"]],
            str(source_dir / "reconstruction_bridge.cpp"),
            *[f"-L{p}" for p in libraries],
            *[f"-Wl,-rpath,{p}" for p in libraries],
            "-ltorch_npu",
            "-lascendcl",
            "-lc10",
            "-ltorch_cpu",
            "-ltorch",
            "-ltorch_python",
            "-o",
            str(build / f"glm_reconstruction_bridge_v{version}.so"),
        ]
    )
    return commands


def candidate_plan(
    inputs,
    snapshot_dir,
    build_dir,
    toolchain,
    *,
    version=DEFAULT_VERSION,
    scale_layout=FP16_SCALE_LAYOUT,
    execution_snapshot_dir=None,
):
    """Return hash-bound inputs and commands; default operation performs no writes."""
    if type(version) is not int or version <= 984:
        raise ValueError("candidate needs a distinct future version above 984; no version is reserved")
    inputs, snapshot, build = Path(inputs), Path(snapshot_dir).absolute(), Path(build_dir).absolute()
    execution_snapshot = snapshot if execution_snapshot_dir is None else Path(execution_snapshot_dir).absolute()
    if snapshot == build or snapshot in build.parents or build in snapshot.parents:
        raise ValueError("freeze and future build directories must be separate")
    if execution_snapshot == build or execution_snapshot in build.parents or build in execution_snapshot.parents:
        raise ValueError("staged input and future build directories must be separate")
    source = (inputs / "source/glm_fused_moe.cpp").read_bytes()
    raw_provenance = (inputs / "reference-provenance.json").read_bytes()
    reference = validate_reference(source, raw_provenance, scale_layout)
    files = {"reference-provenance.json": raw_provenance}
    if {p.name for p in (inputs / "source").iterdir()} != {name for name, _ in SOURCE_HASHES}:
        raise ValueError("unexpected frozen source dependencies")
    for name, expected in SOURCE_HASHES:
        if digest((inputs / "source" / name).read_bytes()) != expected:
            raise ValueError("frozen source dependency identity mismatch: " + name)
    for path in sorted((inputs / "source").iterdir()):
        if path.is_file():
            files["source/" + path.name] = path.read_bytes()
    for name, sha in reference["_helpers"].items():
        data = (inputs / "helpers" / name).read_bytes()
        if digest(data) != sha:
            raise ValueError("frozen Python helper identity mismatch: " + name)
        files[f"glm_reconstruction_v{version}_helpers/{name}"] = data
    for name in ("glm_fused_quantize.h", "glm_fused_scratch.h", "reconstruction_bridge.cpp"):
        if digest(files["source/" + name]) != reference[name]["source_sha256"]:
            raise ValueError("frozen source dependency identity mismatch: " + name)
    helper = (Path(__file__).parent / "glm_decode_end_cache.h").read_bytes()
    if digest(helper) != QUALIFIED_HELPER_SHA256:
        raise ValueError("decode cache helper differs from CPU-qualified T17 source")
    files["source/glm_decode_end_cache.h"] = helper
    route_cache = (Path(__file__).parent / "glm_fused_route_cache.h").read_bytes()
    if digest(route_cache) != QUALIFIED_ROUTE_CACHE_SHA256:
        raise ValueError("route cache layout differs from the audited T17 dependency")
    files["source/glm_fused_route_cache.h"] = route_cache
    files["source/glm_fused_moe.cpp"] = transform_source(source)
    reused = []
    for name, provenance_key in zip(REUSE_ASSETS, REUSE_PROVENANCE_KEYS):
        data = (inputs / "assets" / name).read_bytes()
        if digest(data) != reference[provenance_key]["binary_sha256"]:
            raise ValueError("reused baseline binary identity mismatch: " + name)
        files["reuse/" + name] = data
        reused.append(
            {
                "source": str(execution_snapshot / "reuse" / name),
                "destination": str(build / name),
                "sha256": digest(data),
            }
        )
    options = dict(
        reference["_build"],
        namespace=f"glm_reconstruction_v{version}",
        version=version,
        helper_package=f"glm_reconstruction_v{version}_helpers",
        decode_end_cache=True,
    )
    commands = compiler_commands(
        files["source/build_reconstruction.py"], reference["_build"], execution_snapshot, build, version, toolchain
    )
    record = {
        "schema_version": 1,
        "state": "offline_plan",
        "version_reserved": False,
        "sdk_compiled": False,
        "native_qualified": False,
        "admissible_native_candidate": False,
        "snapshot_dir": str(snapshot),
        "execution_snapshot_dir": str(execution_snapshot),
        "future_build_dir": str(build),
        "namespace": options["namespace"],
        "_build": options,
        "required_weight_scale_layout": scale_layout,
        "reference_provenance_sha256": REFERENCE_PROVENANCE_SHA256,
        "reference_source_sha256": REFERENCE_SOURCE_SHA256,
        "source_substitutions": {"include": 1, "launch_prepare": 1, "tile_reader": 1},
        "additional_ub_bytes": 0,
        "additional_hbm_bytes": 0,
        "toolchain": toolchain,
        "compile_argv": commands,
        "reuse_assets": reused,
        "assembly_files": [
            {"source": str(execution_snapshot / name), "destination": str(build / name), "sha256": digest(data)}
            for name, data in sorted(files.items())
            if name.startswith(options["helper_package"] + "/")
        ]
        + reused,
        "postcompile_requirements": [
            "Root creates a new unique build directory only when SDK compilation is authorized.",
            "Copy assembly_files without modifying authenticated bytes; execute the six recorded argv lists.",
            "Write full provenance.json with _build, helper/header hashes and actual new binary hashes.",
            "Keep SDK/native gates pending until run; this snapshot is not a loadable manifest.",
        ],
        "input_sha256": {name: digest(data) for name, data in sorted(files.items())},
    }
    record["plan_sha256"] = digest(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
    return record, files


def freeze_candidate(record, files):
    """Append a new immutable source snapshot; never reserve/build its binaries."""
    snapshot = Path(record["snapshot_dir"])
    if snapshot.exists():
        raise FileExistsError("candidate snapshot is append-only")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in files):
        raise ValueError("candidate files must remain within the new snapshot")
    if record.get("state") != "offline_plan" or any(
        record.get(k) is not False
        for k in ("version_reserved", "sdk_compiled", "native_qualified", "admissible_native_candidate")
    ):
        raise ValueError("freeze must retain pending/offline qualification state")
    if {name: digest(data) for name, data in sorted(files.items())} != record["input_sha256"]:
        raise ValueError("candidate file identity changed before freeze")
    unsigned = {k: v for k, v in record.items() if k != "plan_sha256"}
    if digest(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()) != record["plan_sha256"]:
        raise ValueError("candidate plan identity changed before freeze")
    snapshot.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o444)
    path = snapshot / "candidate-plan.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    path.chmod(0o444)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--execution-snapshot-dir", type=Path, help="planned remote staging path; no remote write")
    parser.add_argument("--toolchain", required=True, type=Path)
    parser.add_argument("--version", default=DEFAULT_VERSION, type=int)
    parser.add_argument("--scale-layout", default=FP16_SCALE_LAYOUT)
    parser.add_argument(
        "--freeze", action="store_true", help="write a new input snapshot; default is read-only dry-run"
    )
    args = parser.parse_args()
    record, files = candidate_plan(
        args.inputs,
        args.snapshot_dir,
        args.build_dir,
        json.loads(args.toolchain.read_text()),
        version=args.version,
        scale_layout=args.scale_layout,
        execution_snapshot_dir=args.execution_snapshot_dir,
    )
    if args.freeze:
        freeze_candidate(record, files)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
