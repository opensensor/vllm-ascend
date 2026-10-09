# SPDX-License-Identifier: Apache-2.0
"""Write a reviewable native manifest without loading libraries or kernels."""

import argparse
import ast
import json
from pathlib import Path

from tools.glm_perf.resident_native import NativeManifest, file_digest


def make_manifest(build: Path, runtime: Path):
    provenance = json.loads((build / "provenance.json").read_text())
    namespace = provenance["namespace"]
    wy_source = (runtime / "vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py").read_text()
    function = next(
        node
        for node in ast.parse(wy_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "chunk_gated_delta_rule_310"
    )
    if "wy_prepare" not in {arg.arg for arg in function.args.args + function.args.kwonlyargs}:
        raise ValueError("runtime lacks the WY preparation seam")
    for candidate in ("fused_wy", "local_routes"):
        source = (runtime / f"tools/qwen4exp/resident_candidates/{candidate}.py").read_text()
        resource = next(
            node.value.value
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "RESOURCE_NAME" for t in node.targets)
        )
        if resource != namespace:
            raise ValueError("resident candidate resource version differs from build")
    for declared in [provenance["bridge"], *provenance["binaries"].values()]:
        if file_digest(Path(declared["path"])) != declared["sha256"]:
            raise ValueError("build binary digest mismatch")
    paths = [build / "provenance.json", *[Path(entry["path"]) for entry in provenance["binaries"].values()]]
    paths += [
        build / name
        for name in (
            "native_wy.cpp",
            "native_route_gather.cpp",
            "native_local_swiglu.cpp",
            "reconstruction_bridge.cpp",
            "compile_reconstruction.cpp",
        )
    ]
    for frozen in paths[1 + len(provenance["binaries"]) :]:
        if file_digest(frozen) != provenance["sources"].get(frozen.name):
            raise ValueError("frozen source digest differs from compiled provenance")
    paths += [
        runtime / name
        for name in (
            "tools/qwen4exp/native_prefill.py",
            "tools/qwen4exp/benchmark_prefill_next_310.py",
            "tools/qwen4exp/resident_candidates/fused_wy.py",
            "tools/qwen4exp/resident_candidates/local_routes.py",
            "vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py",
            "vllm_ascend/models/qwen4_exp/w4_moe.py",
        )
    ]

    def entry(path):
        path = path.resolve(strict=True)
        return {"path": str(path), "sha256": file_digest(path)}

    value = {
        "name": namespace,
        "libraries": [provenance["bridge"]],
        "assets": [entry(path) for path in paths],
        "operators": [f"{namespace}::launch"],
        "validation_source": (
            "from pathlib import Path\n"
            "from tools.qwen4exp.benchmark_prefill_next_310 import load_resources, validate_resources\n"
            f"def prepare():\n    return load_resources(Path({str(build.resolve())!r}), load_library=False)\n"
            "def validate(resources):\n    return validate_resources(resources)\n"
        ),
    }
    NativeManifest(value).verify_files()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = make_manifest(args.build, args.runtime)
    with args.output.open("x") as output:
        output.write(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
