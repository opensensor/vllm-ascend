# SPDX-License-Identifier: Apache-2.0
"""Build private general-N exchange/plan libraries and an optional Torch addon."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .pair_collective_build import digest

NATIVE_SOURCES = ("npu_exchange.cpp", "npu_exchange.h", "npu_group.cpp", "npu_group.h")


def build(
    sdk,
    output,
    compiler="g++",
    *,
    torch_root=None,
    torch_npu_root=None,
    python_include=None,
    cxx11_abi=None,
    reference=False,
):
    sdk, output = Path(sdk).resolve(), Path(output).absolute()
    compiler = shutil.which(compiler)
    torch_options = (torch_root, torch_npu_root, python_include, cxx11_abi)
    if any(x is not None for x in torch_options) and not all(x is not None for x in torch_options):
        raise ValueError("all Torch ABI paths and the exact C++ ABI flag are required together")
    addon = torch_root is not None
    if reference and addon:
        raise ValueError("the raw HCCL reference cannot include the Torch addon")
    if addon and (type(cxx11_abi) is not int or cxx11_abi not in (0, 1)):
        raise ValueError("Torch C++ ABI must be exactly 0 or 1")
    inputs = [
        sdk / p
        for p in (
            "include/acl/acl.h",
            "include/acl/acl_rt.h",
            "include/aclnnop/aclnn_add.h",
            "include/aclnn/acl_meta.h",
            "lib64/libascendcl.so",
            "lib64/libopapi.so",
            "lib64/libnnopbase.so",
        )
    ]
    if reference:
        inputs = [
            sdk / p
            for p in (
                "include/hccl/hccl.h",
                "include/hccl/hccl_types.h",
                "include/acl/acl_rt.h",
                "lib64/libhccl.so",
                "lib64/libascendcl.so",
            )
        ]
    if addon:
        torch_root, torch_npu_root, python_include = (
            Path(p).resolve() for p in (torch_root, torch_npu_root, python_include)
        )
        inputs += [
            torch_root / p
            for p in ("include/torch/library.h", "lib/libtorch.so", "lib/libtorch_cpu.so", "lib/libc10.so")
        ]
        inputs += [
            torch_npu_root / p
            for p in (
                "lib/libtorch_npu.so",
                "include/torch_npu/csrc/core/npu/NPUStream.h",
                "include/torch_npu/csrc/framework/OpCommand.h",
            )
        ]
        inputs.append(python_include / "Python.h")
    if compiler is None or any(not p.is_file() for p in inputs):
        raise ValueError("compiler or collective ABI inputs are missing")
    if output.exists():
        raise FileExistsError("private collective output must be fresh")
    source = output / "source"
    source.mkdir(mode=0o700, parents=True)
    original = Path(__file__).resolve().parents[2] / "csrc/sidecard_collectives"
    names = ("hccl_group_reference.cpp",) if reference else (*NATIVE_SOURCES, *(("npu_torch.cpp",) if addon else ()))
    for name in names:
        shutil.copyfile(original / name, source / name)
    common = [
        compiler,
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I" + str(sdk / "include"),
        "-I" + str(source),
    ]
    native = output / ("libsidecard_hccl_group_reference.so" if reference else "libsidecard_npu_collective.so")
    commands = [
        common
        + [str(source / p) for p in ("npu_exchange.cpp", "npu_group.cpp")]
        + [
            "-L" + str(sdk / "lib64"),
            "-Wl,-rpath," + str(sdk / "lib64"),
            "-Wl,-z,defs",
            "-lascendcl",
            "-lopapi",
            "-lnnopbase",
            "-o",
            str(native),
        ]
    ]
    if reference:
        commands = [
            common
            + [
                str(source / "hccl_group_reference.cpp"),
                "-L" + str(sdk / "lib64"),
                "-Wl,-rpath," + str(sdk / "lib64"),
                "-Wl,-z,defs",
                "-lhccl",
                "-lascendcl",
                "-o",
                str(native),
            ]
        ]
    products = [native]
    if addon:
        torch_library = output / "libsidecard_npu_torch.so"
        # Third-party headers are system includes; retain warnings on our code.
        include_paths = [
            torch_root / "include",
            torch_root / "include/torch/csrc/api/include",
            torch_npu_root / "include",
            python_include,
        ]
        commands.append(
            common
            + [arg for p in include_paths for arg in ("-isystem", str(p))]
            + [
                "-D_GLIBCXX_USE_CXX11_ABI=" + str(cxx11_abi),
                str(source / "npu_torch.cpp"),
                "-L" + str(output),
                "-L" + str(torch_root / "lib"),
                "-L" + str(torch_npu_root / "lib"),
                "-Wl,-rpath," + ":".join(map(str, (output, torch_root / "lib", torch_npu_root / "lib"))),
                "-Wl,-z,defs",
                "-lsidecard_npu_collective",
                "-ltorch_npu",
                "-ltorch",
                "-ltorch_cpu",
                "-lc10",
                "-o",
                str(torch_library),
            ]
        )
        products.append(torch_library)
    before = {str(p): digest(p) for p in inputs}
    receipt = dict(
        schema_version=1,
        role="hccl-group-reference" if reference else "npu-binomial-collective",
        exchange_abi=1,
        group_abi=1,
        hardware_validated=False,
        build_complete=False,
        torch_addon=addon,
        cxx11_abi=cxx11_abi,
        compiler_sha256=digest(compiler),
        source_sha256={p.name: digest(p) for p in source.iterdir()},
        abi_inputs_sha256=before,
        commands=commands,
        completed_commands=0,
    )
    path = output / "build.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "compiler.log").open("w") as log:
        for command in commands:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            except OSError as error:
                receipt["launch_error"] = dict(type=type(error).__name__, message=str(error))
                receipt["returncode"] = None
                break
            receipt["returncode"] = result.returncode
            if result.returncode:
                break
            receipt["completed_commands"] += 1
    receipt["abi_inputs_unchanged"] = before == {str(p): digest(p) for p in inputs}
    receipt["compiler_unchanged"] = receipt["compiler_sha256"] == digest(compiler)
    receipt["sources_unchanged"] = receipt["source_sha256"] == {p.name: digest(p) for p in source.iterdir()}
    receipt["products_sha256"] = {p.name: digest(p) for p in products if p.is_file()}
    receipt["compiler_log_sha256"] = digest(output / "compiler.log")
    receipt["build_complete"] = (
        receipt["completed_commands"] == len(commands)
        and receipt["abi_inputs_unchanged"]
        and receipt["compiler_unchanged"]
        and receipt["sources_unchanged"]
        and len(receipt["products_sha256"]) == len(products)
        and all(p.read_bytes()[:4] == b"\x7fELF" for p in products)
    )
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["build_complete"]:
        raise RuntimeError("collective build failed; inspect compiler.log/build.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--torch-root", type=Path)
    parser.add_argument("--torch-npu-root", type=Path)
    parser.add_argument("--python-include", type=Path)
    parser.add_argument("--cxx11-abi", type=int, choices=(0, 1))
    parser.add_argument("--hccl-reference", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.sdk,
                args.output,
                args.compiler,
                torch_root=args.torch_root,
                torch_npu_root=args.torch_npu_root,
                python_include=args.python_include,
                cxx11_abi=args.cxx11_abi,
                reference=args.hccl_reference,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
