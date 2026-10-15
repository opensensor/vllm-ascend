# SPDX-License-Identifier: Apache-2.0
"""Generate a hash-checked native transaction without loading device code."""

import argparse
import json
from pathlib import Path

from tools.glm_perf.resident_native import NativeManifest, file_digest
from tools.qwen4exp.native_gate40_spec import BRIDGE_NAMESPACE, NATIVE_NAME, SYMBOL, binary_paths

RESOURCE_SOURCES = (
    "tools/qwen4exp/native_gate40_spec.py",
    "tools/qwen4exp/native_gate40_probe.py",
    "tools/qwen4exp/direct_gate_up.py",
    "tools/qwen4exp/benchmark_math_paths_310.py",
    "tools/qwen4exp/math_probe_cases.py",
    "vllm_ascend/models/qwen4_exp/w4a8_int4.py",
)


def make_manifest(library, binaries, runtime, native_name=NATIVE_NAME, namespace=BRIDGE_NAMESPACE):
    if set(binaries) != {"baseline", "tile40", "stream40"}:
        raise ValueError("native manifest needs baseline, tile40 and stream40 binaries")
    binaries = {name: str(Path(path).resolve(strict=True)) for name, path in binaries.items()}

    def entry(path):
        path = Path(path).resolve(strict=True)
        return {"path": str(path), "sha256": file_digest(path)}

    value = {
        "name": native_name,
        "libraries": [entry(library)],
        "assets": [entry(path) for path in binaries.values()]
        + [entry(Path(runtime) / path) for path in RESOURCE_SOURCES],
        "operators": [f"{namespace}::launch"],
        "validation_source": (
            "import torch\n"
            "from tools.qwen4exp.direct_gate_up import DirectGateUp\n"
            "from tools.qwen4exp.native_gate40_probe import validate_resources\n"
            f"def prepare():\n    paths = {binaries!r}\n"
            f"    kernel = getattr(torch.classes, {namespace!r}).Kernel\n"
            f"    launch = getattr(torch.ops, {namespace!r}).launch\n"
            f"    return {{name: DirectGateUp(kernel(path, {SYMBOL!r}), launch) for name, path in paths.items()}}\n"
            "def validate(resources):\n    return validate_resources(resources, small=True)\n"
        ),
    }
    manifest = NativeManifest(value)
    manifest.verify_files()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-name", default=NATIVE_NAME)
    parser.add_argument("--namespace", default=BRIDGE_NAMESPACE)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new manifest path; loaded versions must never be rewritten")
    value = make_manifest(args.library, binary_paths(), args.runtime, args.native_name, args.namespace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.output), "digest": NativeManifest(value).digest}), flush=True)


if __name__ == "__main__":
    main()
