# SPDX-License-Identifier: Apache-2.0
"""Build private, fixed-row local-add binaries for the specialized pair SUM.

ACLRTC initializes ACL in a subprocess without selecting a device. This build
must run with a healthy driver. Nothing is installed into the shared OPP tree.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .pair_collective_build import digest

ROWS = (1, 4, 32)
SOURCES = ("pair_add_compile.cpp", "pair_add_runtime.cpp", "pair_add_runtime.h", "pair_add_kernel.cpp")


def commands(sdk, output, compiler):
    sdk, output = Path(sdk), Path(output)
    # Architecture headers/libraries live one level below the shared compiler
    # in CANN 9.1. Some older layouts also place tools in the architecture root.
    tools = sdk / "tools" if (sdk / "tools/hcc/sysroot").is_dir() else sdk.parent / "tools"
    source = output / "source"
    common = [compiler, "-std=c++17", "-O3", "-Wall", "-Wextra", "-Werror", "-I" + str(sdk / "include")]
    link = ["-L" + str(sdk / "lib64"), "-Wl,-rpath," + str(sdk / "lib64")]
    driver = output / "compile-local-add"
    result = [
        common + [str(source / "pair_add_compile.cpp"), *link, "-lacl_rtc", "-lascendcl", "-o", str(driver)],
        common
        + [
            "-fPIC",
            "-shared",
            str(source / "pair_add_runtime.cpp"),
            *link,
            "-Wl,-z,defs",
            "-lascendcl",
            "-o",
            str(output / "libsidecard_local_add.so"),
        ],
    ]
    for rows in ROWS:
        result.append(
            [
                str(driver),
                str(source / "pair_add_kernel.cpp"),
                str(output / f"local-add-r{rows}.bin"),
                "--npu-arch=dav-2002",
                "--sysroot=" + str(tools / "hcc/sysroot"),
                "-isystem" + str(tools / "hcc/aarch64-target-linux-gnu/include/c++/7.3.0"),
                "-isystem" + str(tools / "hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu"),
                "-I" + str(source),
                f"-DSIDECARD_PAIR_ROWS={rows}",
            ]
        )
    return result


def build(sdk, output, compiler="g++"):
    sdk, output = Path(sdk).resolve(), Path(output).absolute()
    compiler = shutil.which(compiler)
    inputs = [
        sdk / p
        for p in (
            "lib64/libascendcl.so",
            "lib64/libacl_rtc.so",
            "include/acl/acl.h",
            "include/acl/acl_rt.h",
            "include/acl/acl_rt_compile.h",
        )
    ]
    if compiler is None or any(not p.is_file() for p in inputs):
        raise ValueError("compiler or local-add SDK inputs are missing")
    if output.exists():
        raise FileExistsError("private local-add output must be fresh")
    output.mkdir(parents=True, mode=0o700)
    source = output / "source"
    source.mkdir()
    original = Path(__file__).resolve().parents[2] / "csrc/sidecard_collectives"
    for name in SOURCES:
        shutil.copyfile(original / name, source / name)
    before = {str(p): digest(p) for p in inputs}
    receipt = dict(
        schema_version=1,
        role="pair-local-add",
        abi=1,
        hardware_validated=False,
        build_complete=False,
        rows=list(ROWS),
        arch="dav-2002",
        compiler_sha256=digest(compiler),
        sdk_inputs_sha256=before,
        source_sha256={p.name: digest(p) for p in source.iterdir()},
        commands=commands(sdk, output, compiler),
        completed_commands=0,
    )
    path = output / "build.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "compiler.log").open("w") as log:
        for command in receipt["commands"]:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            receipt["returncode"] = result.returncode
            if result.returncode:
                break
            receipt["completed_commands"] += 1
    receipt["sdk_inputs_unchanged"] = before == {str(p): digest(p) for p in inputs}
    products = [output / "libsidecard_local_add.so", *(output / f"local-add-r{r}.bin" for r in ROWS)]
    receipt["products_sha256"] = {p.name: digest(p) for p in products if p.is_file()}
    receipt["compiler_log_sha256"] = digest(output / "compiler.log")
    receipt["build_complete"] = (
        receipt["completed_commands"] == len(receipt["commands"])
        and len(receipt["products_sha256"]) == len(products)
        and receipt["sdk_inputs_unchanged"]
        and all(p.read_bytes()[:4] == b"\x7fELF" for p in products)
    )
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["build_complete"]:
        raise RuntimeError("local-add build failed; inspect compiler.log and build.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    print(json.dumps(build(args.sdk, args.output, args.compiler), indent=2))


if __name__ == "__main__":
    main()
