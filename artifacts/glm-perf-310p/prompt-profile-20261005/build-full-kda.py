# SPDX-License-Identifier: Apache-2.0
"""Build an isolated KDA package; never modifies installed runtime artifacts."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    original = Path("/srv/ai/src/glm-l1-wide-build-20261004")
    deployed = Path("/srv/ai/src/kda-persistent-scores-opp")
    source = root / "full-kda-source"
    build = root / "full-kda-build"
    package = root / "opp-score-cache"
    metadata = json.loads((root / "kda-score-cache-base.json").read_text())
    relative_header = Path(metadata["path"]).relative_to("csrc")
    vendor_relative = Path("vendors/custom_transformer")
    deployed_headers = deployed / vendor_relative / "op_impl/ai_core/tbe/custom_transformer_impl/ascendc/chunk_kda_fwd"
    assert not package.exists(), "isolated package already exists"
    hashes = {}
    for path in deployed_headers.iterdir():
        if path.suffix not in (".cpp", ".h"):
            continue
        source_path = original / relative_header.parent / path.name
        assert source_path.read_bytes() == path.read_bytes(), f"deployed source mismatch: {path.name}"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    if not source.exists():
        source.mkdir()
        for path in original.iterdir():
            if path.name.startswith(("build", "opp", "install", ".git")) or path.name in ("output", "third_party"):
                continue
            if path.is_dir():
                shutil.copytree(path, source / path.name)
            else:
                shutil.copy2(path, source / path.name)
        (source / "third_party").symlink_to(original / "third_party", target_is_directory=True)
        subprocess.run(
            ["git", "apply", "--check", "-p2", str(root / "kda-score-cache-columns.patch")], cwd=source, check=True
        )
        subprocess.run(["git", "apply", "-p2", str(root / "kda-score-cache-columns.patch")], cwd=source, check=True)
    assert hashlib.sha256((source / relative_header).read_bytes()).hexdigest() == metadata["candidate_sha256"]
    build.mkdir(exist_ok=True)
    build_environment = dict(os.environ)
    build_environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + build_environment["PATH"]
    commands = [
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DASCEND_COMPUTE_UNIT=ascend310p",
            "-DASCEND_OP_NAME=chunk_kda_fwd",
            "-DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0",
            "-DCANN_3RD_LIB_PATH=" + str(source / "third_party"),
            "-DBUILD_TYPE=Release",
            "-DVERSION=9.1.0",
            "-DENABLE_OPS_HOST=ON",
            "-DENABLE_OPS_KERNEL=ON",
            "-DOPS_COMPILE_OPTIONS=-DGLM_KDA_SCORE_CACHE_COLUMNS",
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DASCEND_PYTHON_EXECUTABLE=" + sys.executable,
            "-DHI_PYTHON=" + sys.executable,
        ],
        ["cmake", "--build", str(build), "--target", "ops_transformer_kernel", "-j2"],
    ]
    for command, label in zip(commands, ("configure", "build")):
        print(label, flush=True)
        with (build / f"{label}.log").open("w") as output:
            subprocess.run(
                command, cwd=source, env=build_environment, stdout=output, stderr=subprocess.STDOUT, check=True
            )

    objects = list((build / "binary/ascend310p/bin/chunk_kda_fwd").glob("*.o"))
    assert len(objects) == 1, objects
    shutil.copytree(deployed, package)
    destination = package / vendor_relative / "op_impl/ai_core/tbe/kernel/ascend310p/chunk_kda_fwd"
    for binary in objects:
        for path in (binary, binary.with_suffix(".json")):
            shutil.copy2(path, destination / path.name)
    shutil.copy2(
        source / relative_header,
        package
        / vendor_relative
        / "op_impl/ai_core/tbe/custom_transformer_impl/ascendc/chunk_kda_fwd"
        / relative_header.name,
    )
    record = {
        "package": str(package),
        "source": str(source),
        "compile_flag": metadata["compile_flag"],
        "deployed_source_hashes": hashes,
        "candidate_header_sha256": metadata["candidate_sha256"],
        "binary": str(destination / objects[0].name),
        "binary_sha256": hashlib.sha256(objects[0].read_bytes()).hexdigest(),
    }
    (root / "full-kda-build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
