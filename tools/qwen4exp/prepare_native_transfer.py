# SPDX-License-Identifier: Apache-2.0
"""Host-only native resource manifest preparation; no library/device loading."""

import argparse
import json
from pathlib import Path

from tools.glm_perf.resident_native import NativeManifest, file_digest

RUNTIME_FILES = (
    "tools/qwen4exp/native_prefill.py",
    "tools/qwen4exp/native_state_layout.py",
    "tools/qwen4exp/native_cached_metadata.py",
    "tools/qwen4exp/benchmark_transfer_next_310.py",
    "tools/qwen4exp/resident_candidates/native_state_layout.py",
    "tools/qwen4exp/resident_candidates/cached_w4_metadata.py",
    "vllm_ascend/models/qwen4_exp/model.py",
    "vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py",
)


def make_manifest(build, runtime):
    provenance = json.loads((build / "provenance.json").read_text())
    if provenance["namespace"] != "qwen_transfer_v1":
        raise ValueError("candidates require qwen_transfer_v1; update version bindings for a later build")
    paths = [build / "provenance.json"]
    paths += [Path(entry["path"]) for entry in provenance["binaries"].values()]
    for name, digest in provenance["sources"].items():
        path = build / name
        if file_digest(path) != digest:
            raise ValueError("compiled source fingerprint mismatch")
        paths.append(path)
    paths += [runtime / name for name in RUNTIME_FILES]
    entries = [{"path": str(path.resolve(strict=True)), "sha256": file_digest(path)} for path in paths]
    value = {
        "name": provenance["namespace"],
        "libraries": [provenance["bridge"]],
        "assets": entries,
        "operators": [f"{provenance['namespace']}::launch"],
        "validation_source": (
            "from pathlib import Path\n"
            "from tools.qwen4exp.benchmark_transfer_next_310 import load_resources, validate_resources\n"
            f"def prepare():\n    return load_resources(Path({str(build.resolve())!r}),load_library=False)\n"
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
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
