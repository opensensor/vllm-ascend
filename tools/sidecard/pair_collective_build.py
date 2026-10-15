# SPDX-License-Identifier: Apache-2.0
"""Build the private pair-SUM shim without importing or initializing devices."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(sdk, output, compiler="g++", *, reference=False):
    sdk, output = Path(sdk).resolve(), Path(output).absolute()
    include, libraries = sdk / "include", sdk / "lib64"
    sources = Path(__file__).resolve().parents[2] / "csrc/sidecard_collectives"
    library_names = (
        ("libhccl.so", "libascendcl.so") if reference else ("libascendcl.so", "libopapi.so", "libnnopbase.so")
    )
    header_names = (
        ("hccl/hccl.h", "hccl/hccl_types.h", "acl/acl_rt.h")
        if reference
        else ("acl/acl_rt.h", "aclnnop/aclnn_add.h", "aclnn/acl_meta.h")
    )
    source_names = ("hccl_reference.cpp",) if reference else ("pair_sum.cpp", "pair_sum.h")
    native_libraries = {name: libraries / name for name in library_names}
    headers = {name: include / name for name in header_names}
    compiler = shutil.which(compiler)
    if compiler is None or any(not p.is_file() for p in (*native_libraries.values(), *headers.values())):
        raise ValueError("compiler or required CANN SDK inputs are missing")
    if output.exists():
        raise FileExistsError("private collective build output must be fresh")
    output.mkdir(parents=True, mode=0o700)
    source_dir = output / "source"
    source_dir.mkdir()
    for name in source_names:
        shutil.copyfile(sources / name, source_dir / name)
    binary = output / ("libsidecard_hccl_reference.so" if reference else "libsidecard_pair.so")
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(include),
        str(source_dir / source_names[0]),
        "-L",
        str(libraries),
        "-Wl,-z,defs",
        "-Wl,-rpath," + str(libraries),
        *("-l" + name.removeprefix("lib").removesuffix(".so") for name in library_names),
        "-o",
        str(binary),
    ]
    before = {str(p): digest(p) for p in (*native_libraries.values(), *headers.values())}
    with (output / "compiler.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    after = {str(p): digest(p) for p in (*native_libraries.values(), *headers.values())}
    receipt = dict(
        schema_version=1,
        abi=1 if reference else 2,
        role="hccl-reference" if reference else "custom",
        hardware_validated=False,
        build_complete=result.returncode == 0 and before == after,
        command=command,
        returncode=result.returncode,
        sdk_inputs_sha256=before,
        sdk_inputs_unchanged=before == after,
        compiler_sha256=digest(compiler),
        source_sha256={p.name: digest(p) for p in source_dir.iterdir()},
        compiler_log_sha256=digest(output / "compiler.log"),
        library_sha256=digest(binary) if binary.exists() else None,
    )
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["build_complete"]:
        raise RuntimeError("private build failed; inspect compiler.log and build.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path, help="CANN architecture root containing include/ and lib64/")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--hccl-reference", action="store_true", help="Build the isolated raw-HCCL comparison shim")
    args = parser.parse_args()
    print(json.dumps(build(args.sdk, args.output, args.compiler, reference=args.hccl_reference), indent=2))


if __name__ == "__main__":
    main()
