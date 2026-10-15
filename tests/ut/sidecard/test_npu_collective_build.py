# SPDX-License-Identifier: Apache-2.0
"""Private general-N/addon build provenance tests; no SDK/device execution."""

import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard import npu_collective_build as module

SDK_INPUTS = (
    "include/acl/acl.h",
    "include/acl/acl_rt.h",
    "include/aclnnop/aclnn_add.h",
    "include/aclnn/acl_meta.h",
    "lib64/libascendcl.so",
    "lib64/libopapi.so",
    "lib64/libnnopbase.so",
)
TORCH_INPUTS = ("include/torch/library.h", "lib/libtorch.so", "lib/libtorch_cpu.so", "lib/libc10.so")
NPU_INPUTS = (
    "lib/libtorch_npu.so",
    "include/torch_npu/csrc/core/npu/NPUStream.h",
    "include/torch_npu/csrc/framework/OpCommand.h",
)
SOURCE_NAMES = frozenset({"npu_exchange.cpp", "npu_exchange.h", "npu_group.cpp", "npu_group.h"})


@pytest.mark.parametrize("failed", (False, True))
def test_separate_raw_hccl_reference_build(build_setup, failed):
    state = build_setup
    for relative in ("include/hccl/hccl.h", "include/hccl/hccl_types.h", "lib64/libhccl.so"):
        path = state.sdk / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"raw HCCL ABI input")
    state.fail_at = 1 if failed else None
    if failed:
        with pytest.raises(RuntimeError, match="build failed"):
            module.build(state.sdk, state.output, reference=True)
    else:
        result = module.build(state.sdk, state.output, reference=True)
        assert result["build_complete"] and result["role"] == "hccl-group-reference"
        assert set(result["source_sha256"]) == {"hccl_group_reference.cpp"}
        assert set(result["products_sha256"]) == {"libsidecard_hccl_group_reference.so"}
    assert len(state.calls) == 1 and "-lhccl" in state.calls[0]
    assert "-lopapi" not in state.calls[0] and "-ltorch" not in state.calls[0]


def elf_product():
    """ELF64 ET_DYN header stand-in, never loaded or described as compiled code."""
    identity = b"\x7fELF\x02\x01\x01" + bytes(9)
    return identity + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0)


@pytest.fixture
def build_setup(tmp_path, monkeypatch):
    roots = dict(
        sdk=tmp_path / "pinned SDK",
        torch_root=tmp_path / "pinned Torch",
        torch_npu_root=tmp_path / "pinned Torch NPU",
        python_include=tmp_path / "pinned Python",
    )
    inputs = []
    for key, names in (
        ("sdk", SDK_INPUTS),
        ("torch_root", TORCH_INPUTS),
        ("torch_npu_root", NPU_INPUTS),
        ("python_include", ("Python.h",)),
    ):
        for name in names:
            path = roots[key] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((key + ":" + name).encode())
            inputs.append(path)
    compiler = tmp_path / "pinned compiler"
    compiler.write_bytes(b"fake-host-compiler-input")
    state = SimpleNamespace(
        **roots,
        compiler=compiler,
        output=tmp_path / "private build",
        inputs=inputs,
        calls=[],
        copies=[],
        fail_at=None,
        raise_at=None,
        malformed_at=None,
        missing_product_at=None,
        mutate=None,
    )
    monkeypatch.setattr(module.shutil, "which", lambda _: str(compiler))
    original_copy = module.shutil.copyfile

    def copy(source, destination):
        assert Path(destination).is_relative_to(state.output)
        assert Path(source).parent.name == "sidecard_collectives"
        state.copies.append((Path(source), Path(destination)))
        return original_copy(source, destination)

    def run(command, **kwargs):
        state.calls.append(list(command))
        index = len(state.calls)
        assert kwargs["check"] is False and kwargs["stderr"] == subprocess.STDOUT
        assert not kwargs.get("shell", False)
        kwargs["stdout"].write(f"fake host compiler step {index}\n")
        if index == state.raise_at:
            raise OSError("injected host compiler launch error")
        if index == state.fail_at:
            return SimpleNamespace(returncode=23)
        assert command.count("-o") == 1
        product = Path(command[command.index("-o") + 1])
        assert product.parent == state.output
        if index != state.missing_product_at:
            product.write_bytes(b"not-an-ELF-product" if index == state.malformed_at else elf_product())
        if state.mutate is not None:
            target = {
                "sdk": state.sdk / SDK_INPUTS[0],
                "torch": state.torch_root / TORCH_INPUTS[1],
                "torch_npu": state.torch_npu_root / NPU_INPUTS[0],
                "python": state.python_include / "Python.h",
                "compiler": compiler,
                "source": state.output / "source/npu_exchange.cpp",
            }[state.mutate]
            target.write_bytes(target.read_bytes() + b"\nmutation")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.shutil, "copyfile", copy)
    monkeypatch.setattr(module.subprocess, "run", run)
    return state


def options(state, abi=1):
    return dict(
        torch_root=state.torch_root,
        torch_npu_root=state.torch_npu_root,
        python_include=state.python_include,
        cxx11_abi=abi,
    )


def read_receipt(state):
    return json.loads((state.output / "build.json").read_bytes())


def test_native_only_freezes_general_n_sources_and_pinned_sdk_command(build_setup):
    state = build_setup
    original_inputs = {str(path): module.digest(path) for path in state.inputs}
    receipt = module.build(state.sdk, state.output, compiler="explicit-host-compiler")
    assert receipt == read_receipt(state)
    assert receipt["build_complete"] and not receipt["hardware_validated"] and not receipt["torch_addon"]
    assert receipt["completed_commands"] == 1
    assert receipt["group_abi"] == receipt["exchange_abi"] == 1
    assert set(receipt["source_sha256"]) == SOURCE_NAMES
    assert set(receipt["products_sha256"]) == {"libsidecard_npu_collective.so"}
    command = state.calls[0]
    assert command[0] == str(state.compiler)
    assert all(flag in command for flag in ("-std=c++17", "-shared", "-fPIC", "-Werror", "-Wl,-z,defs"))
    assert [argument for argument in command if argument.endswith(".cpp")] == [
        str(state.output / "source/npu_exchange.cpp"),
        str(state.output / "source/npu_group.cpp"),
    ]
    assert "-I" + str(state.sdk / "include") in command
    assert "-L" + str(state.sdk / "lib64") in command
    assert "-Wl,-rpath," + str(state.sdk / "lib64") in command
    assert all(library in command for library in ("-lascendcl", "-lopapi", "-lnnopbase"))
    assert receipt["compiler_log_sha256"] == module.digest(state.output / "compiler.log")
    assert receipt["source_sha256"] == {name: module.digest(state.output / "source" / name) for name in SOURCE_NAMES}
    assert original_inputs == {str(path): module.digest(path) for path in state.inputs}
    assert not any(word in arg for arg in command for word in ("sudo", "--install", "--opp", "SetDevice"))
    assert all(destination.parent == state.output / "source" for _, destination in state.copies)


@pytest.mark.parametrize("abi", (0, 1))
def test_torch_addon_links_exact_frozen_private_core_and_explicit_abi(build_setup, abi):
    state = build_setup
    receipt = module.build(state.sdk, state.output, **options(state, abi))
    assert receipt["build_complete"] and receipt["torch_addon"] and receipt["cxx11_abi"] == abi
    assert receipt["completed_commands"] == 2
    assert set(receipt["source_sha256"]) == SOURCE_NAMES | {"npu_torch.cpp"}
    assert set(receipt["products_sha256"]) == {"libsidecard_npu_collective.so", "libsidecard_npu_torch.so"}
    native, addon = state.calls
    assert not any(argument.startswith("-D_GLIBCXX_USE_CXX11_ABI=") for argument in native)
    assert [argument for argument in addon if argument.startswith("-D_GLIBCXX_USE_CXX11_ABI=")] == [
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
    ]
    assert str(state.output / "source/npu_torch.cpp") in addon
    assert "-L" + str(state.output) in addon and "-lsidecard_npu_collective" in addon
    assert (
        "-Wl,-rpath," + ":".join(map(str, (state.output, state.torch_root / "lib", state.torch_npu_root / "lib")))
        in addon
    )
    assert all(library in addon for library in ("-ltorch_npu", "-ltorch", "-ltorch_cpu", "-lc10"))
    assert [addon[index + 1] for index, argument in enumerate(addon) if argument == "-isystem"] == list(
        map(
            str,
            (
                state.torch_root / "include",
                state.torch_root / "include/torch/csrc/api/include",
                state.torch_npu_root / "include",
                state.python_include,
            ),
        )
    )
    expected_inputs = {str(path): module.digest(path) for path in state.inputs}
    assert receipt["abi_inputs_sha256"] == expected_inputs
    assert all((state.output / product).read_bytes().startswith(b"\x7fELF") for product in receipt["products_sha256"])


def test_relative_torch_paths_are_normalized_in_command_receipt(build_setup, monkeypatch):
    state = build_setup
    monkeypatch.chdir(state.output.parent)
    relative = {name: Path(path.name) if name != "cxx11_abi" else path for name, path in options(state).items()}
    receipt = module.build(Path(state.sdk.name), state.output, **relative)
    assert all(Path(name).is_absolute() for name in receipt["abi_inputs_sha256"])
    assert "-L" + str(state.torch_root / "lib") in state.calls[1]


@pytest.mark.parametrize("mask", range(1, 15))
def test_torch_inputs_must_be_supplied_all_or_none_before_output_creation(build_setup, mask):
    state = build_setup
    selected = {key: value for index, (key, value) in enumerate(options(state).items()) if mask & (1 << index)}
    with pytest.raises(ValueError, match="required together"):
        module.build(state.sdk, state.output, **selected)
    assert not state.output.exists() and not state.calls and not state.copies


@pytest.mark.parametrize("abi", (False, True, -1, 2, 0.0, 1.0, "1"))
def test_torch_abi_rejects_bool_float_or_guessed_value(build_setup, abi):
    state = build_setup
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        module.build(state.sdk, state.output, **options(state, abi))
    assert not state.output.exists() and not state.calls


@pytest.mark.parametrize("missing", SDK_INPUTS)
def test_missing_pinned_sdk_input_fails_before_source_or_output_changes(build_setup, missing):
    state = build_setup
    (state.sdk / missing).unlink()
    with pytest.raises(ValueError, match="inputs are missing"):
        module.build(state.sdk, state.output)
    assert not state.output.exists() and not state.calls and not state.copies


@pytest.mark.parametrize(
    "root,name",
    (
        ("torch_root", TORCH_INPUTS[0]),
        ("torch_root", TORCH_INPUTS[2]),
        ("torch_npu_root", NPU_INPUTS[0]),
        ("torch_npu_root", NPU_INPUTS[1]),
        ("python_include", "Python.h"),
    ),
)
def test_missing_torch_or_python_abi_inputs_fail_before_build(build_setup, root, name):
    state = build_setup
    (getattr(state, root) / name).unlink()
    with pytest.raises(ValueError, match="inputs are missing"):
        module.build(state.sdk, state.output, **options(state))
    assert not state.output.exists() and not state.calls


def test_unknown_compiler_and_existing_output_do_not_overwrite_anything(build_setup, monkeypatch):
    state = build_setup
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="compiler"):
        module.build(state.sdk, state.output)
    assert not state.output.exists()
    monkeypatch.setattr(module.shutil, "which", lambda _: str(state.compiler))
    state.output.mkdir()
    marker = state.output / "keep"
    marker.write_bytes(b"unrelated-existing-artifact")
    with pytest.raises(FileExistsError):
        module.build(state.sdk, state.output)
    assert marker.read_bytes() == b"unrelated-existing-artifact" and not state.calls and not state.copies


@pytest.mark.parametrize("fail_at", (1, 2))
def test_each_compiler_failure_retains_frozen_sources_log_and_accurate_receipt(build_setup, fail_at):
    state = build_setup
    state.fail_at = fail_at
    with pytest.raises(RuntimeError, match="build failed"):
        module.build(state.sdk, state.output, **options(state))
    receipt = read_receipt(state)
    assert receipt["returncode"] == 23 and receipt["completed_commands"] == fail_at - 1
    assert not receipt["build_complete"] and not receipt["hardware_validated"]
    assert len(state.calls) == fail_at
    assert set(receipt["source_sha256"]) == SOURCE_NAMES | {"npu_torch.cpp"}
    assert (state.output / "compiler.log").read_text().endswith(f"fake host compiler step {fail_at}\n")
    assert receipt["compiler_log_sha256"] == module.digest(state.output / "compiler.log")
    assert receipt["compiler_unchanged"] and receipt["sources_unchanged"] and receipt["abi_inputs_unchanged"]
    assert len(receipt["products_sha256"]) == fail_at - 1


@pytest.mark.parametrize("raise_at", (1, 2))
def test_compiler_launch_exception_preserves_completed_step_provenance(build_setup, raise_at):
    state = build_setup
    state.raise_at = raise_at
    with pytest.raises((OSError, RuntimeError)):
        module.build(state.sdk, state.output, **options(state))
    receipt = read_receipt(state)
    assert not receipt["build_complete"] and not receipt["hardware_validated"]
    assert receipt["completed_commands"] == raise_at - 1
    assert receipt["returncode"] is None
    assert receipt["launch_error"]["type"] == "OSError"
    assert "injected host compiler launch error" in receipt["launch_error"]["message"]
    assert len(receipt["products_sha256"]) == raise_at - 1
    assert set(receipt["source_sha256"]) == SOURCE_NAMES | {"npu_torch.cpp"}
    assert len(state.calls) == raise_at
    assert (state.output / "compiler.log").read_text().endswith(f"fake host compiler step {raise_at}\n")
    assert (state.output / "source/npu_torch.cpp").is_file()


@pytest.mark.parametrize("mutation", ("sdk", "torch", "torch_npu", "python", "compiler", "source"))
def test_mutated_abi_compiler_or_frozen_source_cannot_receive_completed_receipt(build_setup, mutation):
    state = build_setup
    state.mutate = mutation
    with pytest.raises(RuntimeError, match="build failed"):
        module.build(state.sdk, state.output, **options(state))
    receipt = read_receipt(state)
    assert not receipt["build_complete"]
    field = (
        "compiler_unchanged"
        if mutation == "compiler"
        else "sources_unchanged"
        if mutation == "source"
        else "abi_inputs_unchanged"
    )
    assert receipt[field] is False
    assert (state.output / "source/npu_group.cpp").is_file()


@pytest.mark.parametrize("malformed,missing", ((1, None), (2, None), (None, 1), (None, 2)))
def test_non_elf_or_missing_native_and_addon_products_fail_with_provenance(build_setup, malformed, missing):
    state = build_setup
    state.malformed_at, state.missing_product_at = malformed, missing
    with pytest.raises(RuntimeError, match="build failed"):
        module.build(state.sdk, state.output, **options(state))
    receipt = read_receipt(state)
    assert receipt["completed_commands"] == 2 and not receipt["build_complete"]
    assert receipt["sources_unchanged"] and receipt["abi_inputs_unchanged"]
    assert (state.output / "compiler.log").is_file()


def test_import_and_cli_help_do_not_import_torch_or_npu():
    source = (
        "import sys; import tools.sidecard.npu_collective_build; "
        "assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules"
    )
    root = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, "-c", source], cwd=root, check=True)
    subprocess.run(
        [sys.executable, "-m", "tools.sidecard.npu_collective_build", "--help"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
