# SPDX-License-Identifier: Apache-2.0
"""Build orchestration/admission tests using synthetic files, never a compiler."""

import ast
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import regex as re
import torch

from tools.sidecard import execution_kernel_build as build


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents.encode() if isinstance(contents, str) else contents)
    return path


def json_file(path, value):
    return write(path, json.dumps(value, sort_keys=True) + "\n")


def elf(machine=None):
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", header, 16, 3, 62 if (machine or platform.machine()) == "x86_64" else 183)
    return bytes(header)


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "frozen"
    repo = Path(__file__).resolve().parents[3]
    for name in build.REQUIRED_SOURCES:
        path = root / name
        if name.startswith("csrc/sidecard_execution/") or name == "tools/sidecard/execution_kernel_build.py":
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / name, path)
        else:
            write(path, "authenticated frozen fixture " + name)
    for directory in build.DIRECTORIES:
        for relative in ("CMakeLists.txt", "op_host/CMakeLists.txt", "op_kernel/candidate.cpp"):
            write(root / "csrc" / directory / relative, "frozen candidate " + relative)
    for name in build.PREPARED_DEPENDENCIES:
        write(root / name, "authenticated prepared dependency")
    source_manifest = json_file(tmp_path / "sources.json", dict(schema_version=1, files=build.inventory(root)))
    roots = {name: tmp_path / name for name in ("cann_root", "torch_root", "torch_npu_root")}
    for path in roots.values():
        path.mkdir()
    paths = {name: str(path) for name, path in roots.items()}
    for name in ("python_executable", "cmake", "compiler", "c_compiler", "base_ops_library"):
        paths[name] = str(write(tmp_path / "pinned" / name, "pinned " + name))
    paths["cann_compiler"] = str(write(roots["cann_root"] / "compiler/bin/bisheng", "pinned bisheng"))
    paths["cann_version_file"] = str(write(roots["cann_root"] / "version.info", "version=9.1.0\n"))
    for name in ("share/cmake/Torch/TorchConfig.cmake", "lib/libtorch.so", "lib/libtorch_cpu.so", "lib/libc10.so"):
        write(roots["torch_root"] / name, "pinned " + name)
    write(roots["torch_npu_root"] / "lib/libtorch_npu.so", "pinned torch npu")
    write(roots["cann_root"] / "include/acl/acl.h", "pinned ACL header")
    write(roots["cann_root"] / "lib64/libascendcl.so", "pinned ACL library")
    abi = dict(
        schema_version=1,
        kind="sidecard-execution-abi-v1",
        soc="ascend310p",
        cann_version="9.1.0",
        torch_version="2.13.0",
        torch_npu_version="2.13.0.post1",
        cxx11_abi=1,
        cxx11_abi_source="parent CPU torch._C._GLIBCXX_USE_CXX11_ABI",
        machine=platform.machine(),
        python_soabi="cpython-312-x86_64-linux-gnu",
        paths=paths,
        compiler_version="pinned C++ version",
        c_compiler_version="pinned C version",
        cann_compiler_version="pinned CANN version",
        cmake_version="pinned cmake version",
        files={
            str(path): build.digest(path)
            for parent in (*roots.values(), tmp_path / "pinned")
            for path in parent.rglob("*")
            if path.is_file()
        },
    )
    abi_manifest = json_file(tmp_path / "abi.json", abi)
    return dict(
        source_root=root,
        source_manifest=source_manifest,
        source_sha256=build.digest(source_manifest),
        abi_manifest=abi_manifest,
        abi_sha256=build.digest(abi_manifest),
        output=tmp_path / "build",
        abi=abi,
    )


def prepare(inputs):
    return build.prepare(**{name: value for name, value in inputs.items() if name != "abi"})


def mocked_compiler(inputs, monkeypatch):
    """Simulate successful tool outputs; fixture ELF headers are not real kernels."""
    calls = []

    def execute(argv, *, cwd, env, stdout, stderr, check):
        root = Path(cwd)
        manifest = json.loads((root / build.MANIFEST).read_bytes())
        command = next(record for record in manifest["commands"] if record["argv"] == argv)
        name = command["name"]
        calls.append(name)
        version_names = dict(
            zip(
                ("host-cxx-version", "host-c-version", "cann-compiler-version", "cmake-version"),
                build.ABI_VERSIONS,
                strict=True,
            )
        )
        if name in version_names:
            stdout.write((inputs["abi"][version_names[name]] + "\n").encode())
        elif name == "python-package-abi":
            stdout.write(
                json.dumps(
                    {
                        key: inputs["abi"][key]
                        for key in ("torch_version", "torch_npu_version", "machine", "python_soabi")
                    }
                ).encode()
            )
        elif name == "configure-private-opp":
            write(root / "opp-build/CMakeCache.txt", "synthetic configure output")
        elif name == "compile-private-opp":
            tiling = write(root / "opp-build/libcust_opmaster_rt2.0.so", elf())
            compat = root / "opp-build/compat/liboptiling.so"
            compat.parent.mkdir()
            compat.symlink_to(f"lib/linux/{inputs['abi']['machine']}/{tiling.name}")
            custom = root / "opp-build/custom/op_impl/ai_core/tbe/op_tiling/liboptiling.so"
            custom.parent.mkdir(parents=True)
            custom.symlink_to(tiling)
        elif name == "stage-private-opp":
            package = root / manifest["opp_path"]
            write(package / "op_api/lib/libcust_opapi.so", elf())
            tiling = package / "op_impl/ai_core/tbe/op_tiling"
            write(tiling / f"lib/linux/{inputs['abi']['machine']}/libcust_opmaster_rt2.0.so", elf())
            (tiling / "liboptiling.so").symlink_to(f"lib/linux/{inputs['abi']['machine']}/libcust_opmaster_rt2.0.so")
            json_file(
                package / "op_impl/ai_core/tbe/config/ascend310p/aic-ascend310p-ops-info.json",
                {operator: {} for operator in build.OPERATORS},
            )
            for operator in build.OPERATORS:
                write(package / f"op_impl/ai_core/tbe/kernel/ascend310p/{operator}.o", "synthetic device binary")
        elif name == "compile-private-addon":
            write(root / manifest["addon_library"], elf())
        else:
            stdout.write(b"synthetic compiler output\n")
        assert env["TMPDIR"] == str(root / "tmp")
        assert check is True and stderr == subprocess.STDOUT

    monkeypatch.setattr(build.subprocess, "run", execute)
    return calls


def test_prepare_executes_nothing_and_cannot_pass_runtime_admission(inputs, monkeypatch):
    monkeypatch.setattr(build.subprocess, "run", lambda *args, **kwargs: pytest.fail("preparation executed a tool"))
    result = prepare(inputs)
    assert result["build_complete"] is False and result["hardware_validated"] is False
    assert result["operators"] == list(build.OPERATORS)
    assert result["base_ops_library"] == inputs["abi"]["paths"]["base_ops_library"]
    assert build.inventory(inputs["output"] / "source") == result["source_files"]
    with pytest.raises(ValueError, match="incomplete"):
        build.authenticate_build(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))


def test_fixed_command_plan_contains_only_private_three_operator_build(inputs):
    result = prepare(inputs)
    arguments = [argument for record in result["commands"] for argument in record["argv"]]
    assert all(".run" not in argument for argument in arguments)
    assert all("torch_binding.cpp" not in argument for argument in arguments)
    assert all("-D" + flag + "=ON" in arguments for flag in build.FLAGS)
    assert "-DASCEND_COMPUTE_UNIT=ascend310p" in arguments
    assert "-DFETCHCONTENT_FULLY_DISCONNECTED=ON" in arguments
    assert "-DCMAKE_INSTALL_PREFIX=" + str(inputs["output"] / "opp") in arguments
    assert "-D_GLIBCXX_USE_CXX11_ABI=1" in arguments
    assert result["addon_library"] == "addon/libsidecard_execution.so"
    cache = str(inputs["output"] / "work-source/csrc/sidecard_execution/prepare-cache.cmake")
    configure = next(record["argv"] for record in result["commands"] if record["name"] == "configure-private-opp")
    assert configure[configure.index("-C") + 1] == cache
    assert "-DSIDECARD_EXECUTION_INITIAL_CACHE=" + cache in configure
    assert "-DSIDECARD_EXECUTION_PREPARE_JOBS=2" in configure


@pytest.mark.parametrize("mutation", ["source", "abi", "source-anchor", "abi-anchor", "source-symlink", "path"])
def test_prepare_rejects_drift_before_output_or_compiler(inputs, mutation):
    if mutation == "source":
        write(inputs["source_root"] / "csrc/CMakeLists.txt", "changed")
    elif mutation == "abi":
        write(Path(inputs["abi"]["paths"]["compiler"]), "changed")
    elif mutation.endswith("anchor"):
        inputs["source_sha256" if mutation == "source-anchor" else "abi_sha256"] = "0" * 64
    elif mutation == "source-symlink":
        target = inputs["source_root"] / "csrc/CMakeLists.txt"
        raw = target.read_bytes()
        target.unlink()
        target.symlink_to(write(target.parent / "other.txt", raw))
    else:
        packet = json.loads(inputs["source_manifest"].read_bytes())
        packet["files"]["../escape"] = "0" * 64
        json_file(inputs["source_manifest"], packet)
        inputs["source_sha256"] = build.digest(inputs["source_manifest"])
    with pytest.raises(ValueError):
        prepare(inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("cann_version", "9.2.0"),
        ("torch_version", "2.12.0"),
        ("torch_npu_version", "2.13garbage"),
        ("soc", "ascend910b"),
        ("cxx11_abi", True),
        ("compiler_version", ""),
    ],
)
def test_abi_admission_is_exact_and_does_not_parse_absent_torchconfig_flag(inputs, field, value):
    packet = dict(inputs["abi"])
    packet[field] = value
    json_file(inputs["abi_manifest"], packet)
    with pytest.raises(ValueError):
        build.authenticate_abi(inputs["abi_manifest"], build.digest(inputs["abi_manifest"]))


def test_mocked_completed_build_admits_and_binds_each_output(inputs, monkeypatch):
    prepare(inputs)
    calls = mocked_compiler(inputs, monkeypatch)
    result = build.compile_prepared(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))
    assert result["build_complete"] is True and result["hardware_validated"] is False
    assert result["package_link_normalizations"] == {
        "opp-build/compat/liboptiling.so": {
            "link_target": f"lib/linux/{inputs['abi']['machine']}/libcust_opmaster_rt2.0.so",
            "source": "libcust_opmaster_rt2.0.so",
            "sha256": build.digest(inputs["output"] / "opp-build/libcust_opmaster_rt2.0.so"),
        }
    }
    assert calls == [record["name"] for record in result["commands"]]
    assert build.authenticate_build(inputs["output"], build.digest(inputs["output"] / build.MANIFEST)) == result


@pytest.mark.parametrize(
    "relative",
    [
        "source/csrc/CMakeLists.txt",
        "work-source/csrc/CMakeLists.txt",
        "addon/libsidecard_execution.so",
        "opp-build/CMakeCache.txt",
        "logs/00-host-cxx-version.log",
    ],
)
def test_completed_receipt_rejects_any_source_binary_or_log_drift(inputs, monkeypatch, relative):
    prepare(inputs)
    mocked_compiler(inputs, monkeypatch)
    build.compile_prepared(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))
    sha = build.digest(inputs["output"] / build.MANIFEST)
    write(inputs["output"] / relative, "changed after build")
    with pytest.raises(ValueError):
        build.authenticate_build(inputs["output"], sha)


def test_completed_receipt_rejects_package_drift_and_unlisted_outputs(inputs, monkeypatch):
    prepare(inputs)
    mocked_compiler(inputs, monkeypatch)
    result = build.compile_prepared(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))
    sha = build.digest(inputs["output"] / build.MANIFEST)
    target = inputs["output"] / result["opp_path"] / "op_api/lib/libcust_opapi.so"
    write(target, elf() + b"changed")
    with pytest.raises(ValueError):
        build.authenticate_build(inputs["output"], sha)
    write(target, elf())
    write(inputs["output"] / "addon/unlisted.so", elf())
    with pytest.raises(ValueError, match="unmanifested"):
        build.authenticate_build(inputs["output"], sha)


def test_invalid_compiler_output_records_failure_and_cannot_admit(inputs, monkeypatch):
    prepare(inputs)
    mocked_compiler(inputs, monkeypatch)
    execute = build.subprocess.run

    def invalid_addon(argv, **kwargs):
        execute(argv, **kwargs)
        if "-o" in argv:
            write(Path(argv[argv.index("-o") + 1]), "not a compiled shared library")

    monkeypatch.setattr(build.subprocess, "run", invalid_addon)
    with pytest.raises(ValueError, match="ELF64"):
        build.compile_prepared(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))
    result = json.loads((inputs["output"] / build.MANIFEST).read_bytes())
    assert result["build_complete"] is False and result["error"]["type"] == "ValueError"
    with pytest.raises(ValueError, match="incomplete or failed"):
        build.authenticate_build(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))


@pytest.mark.parametrize("location", ["source", "toolkit", "existing"])
def test_prepare_never_writes_into_source_installed_toolkit_or_existing_directory(inputs, location):
    if location == "source":
        inputs["output"] = inputs["source_root"] / "new-build"
    elif location == "toolkit":
        inputs["output"] = Path(inputs["abi"]["paths"]["cann_root"]) / "new-build"
    else:
        inputs["output"].mkdir()
    with pytest.raises(ValueError):
        prepare(inputs)


def test_missing_prepared_dependencies_stop_before_tools(inputs, monkeypatch):
    packet = json.loads(inputs["source_manifest"].read_bytes())
    del packet["files"][build.PREPARED_DEPENDENCIES[0]]
    json_file(inputs["source_manifest"], packet)
    inputs["source_sha256"] = build.digest(inputs["source_manifest"])
    result = prepare(inputs)
    assert result["missing_prepared_dependencies"] == [build.PREPARED_DEPENDENCIES[0]]
    monkeypatch.setattr(
        build.subprocess, "run", lambda *args, **kwargs: pytest.fail("missing dependency executed tool")
    )
    with pytest.raises(ValueError, match="do not download/extract"):
        build.compile_prepared(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))


def test_installed_links_are_materialized_only_inside_private_prefix(tmp_path):
    private = tmp_path / "private"
    target = write(private / "lib/liboptiling.so", "library")
    (private / "alias.so").symlink_to("lib/liboptiling.so")
    build.materialize_package_links(private)
    assert not (private / "alias.so").is_symlink()
    assert (private / "alias.so").read_bytes() == target.read_bytes()
    (private / "escape.so").symlink_to(write(tmp_path / "external.so", "outside"))
    with pytest.raises(ValueError, match="escapes"):
        build.materialize_package_links(private)


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_build_install_relative_optiling_link_is_materialized_with_receipt(tmp_path, machine):
    private = tmp_path / "private"
    root = private / "opp-build"
    target = write(root / "libcust_opmaster_rt2.0.so", elf(machine))
    compat = root / "compat/liboptiling.so"
    compat.parent.mkdir()
    literal = f"lib/linux/{machine}/{target.name}"
    compat.symlink_to(literal)
    assert not compat.exists()
    result = build.materialize_package_links(root, allowed_root=private, machine=machine)
    assert result == {
        "compat/liboptiling.so": dict(link_target=literal, source=target.name, sha256=build.digest(target))
    }
    assert not compat.is_symlink() and compat.read_bytes() == target.read_bytes()
    build.authenticate_link_normalizations(
        private,
        dict(
            abi=dict(machine=machine),
            package_link_normalizations={"opp-build/compat/liboptiling.so": result["compat/liboptiling.so"]},
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign-location",
        "foreign-target",
        "absolute-target",
        "missing-source",
        "text-source",
        "wrong-machine",
        "symlink-source",
        "missing-abi",
        "foreign-owner",
    ],
)
def test_build_optiling_normalization_rejects_other_dangling_links(tmp_path, monkeypatch, mutation):
    private = tmp_path / "private"
    root = private / "opp-build"
    machine = platform.machine()
    target = write(root / "libcust_opmaster_rt2.0.so", elf())
    compat = root / ("foreign/liboptiling.so" if mutation == "foreign-location" else "compat/liboptiling.so")
    compat.parent.mkdir()
    literal = f"lib/linux/{machine}/{target.name}"
    if mutation == "foreign-target":
        literal = "../unknown/" + target.name
    elif mutation == "absolute-target":
        literal = str(private / "missing" / target.name)
    elif mutation == "missing-source":
        target.unlink()
    elif mutation == "text-source":
        target.write_text("not an ELF shared library")
    elif mutation == "wrong-machine":
        target.write_bytes(elf("aarch64" if machine == "x86_64" else "x86_64"))
    elif mutation == "symlink-source":
        target.unlink()
        target.symlink_to(write(private / "other.so", elf()))
    elif mutation == "foreign-owner":
        original_stat = Path.stat

        def stat(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            if path == target:
                fields = list(result)
                fields[4] = result.st_uid + 1
                return os.stat_result(fields)
            return result

        monkeypatch.setattr(Path, "stat", stat)
    compat.symlink_to(literal)
    with pytest.raises(ValueError):
        build.materialize_package_links(
            root, allowed_root=private, machine=None if mutation == "missing-abi" else machine
        )
    assert compat.is_symlink()


def test_invalid_link_does_not_partially_materialize_other_links(tmp_path):
    root = tmp_path / "private"
    write(root / "library.so", elf())
    good = root / "good.so"
    good.symlink_to("library.so")
    (root / "broken.so").symlink_to("absent.so")
    with pytest.raises(ValueError):
        build.materialize_package_links(root)
    assert good.is_symlink()


def test_runtime_normalization_admission_does_not_require_caller_ownership(inputs, monkeypatch):
    prepare(inputs)
    mocked_compiler(inputs, monkeypatch)
    build.compile_prepared(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))
    original_stat = Path.stat

    def stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        fields = list(result)
        fields[4] = result.st_uid + 1
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "stat", stat)
    assert build.authenticate_build(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))["build_complete"]


@pytest.mark.parametrize("mutation", ["target", "source", "hash", "extra", "missing"])
def test_completed_build_rejects_changed_normalization_receipt(inputs, monkeypatch, mutation):
    prepare(inputs)
    mocked_compiler(inputs, monkeypatch)
    result = build.compile_prepared(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))
    record = result["package_link_normalizations"]["opp-build/compat/liboptiling.so"]
    if mutation == "target":
        record["link_target"] = "foreign/library.so"
    elif mutation == "source":
        record["source"] = "../outside.so"
    elif mutation == "hash":
        record["sha256"] = "0" * 64
    elif mutation == "extra":
        result["package_link_normalizations"]["other/link.so"] = dict(record)
    else:
        del result["package_link_normalizations"]
    build.write_manifest(inputs["output"], result)
    with pytest.raises(ValueError):
        build.authenticate_build(inputs["output"], build.digest(inputs["output"] / build.MANIFEST))


def test_addon_native_schema_preserves_carry_mutation_and_only_three_ops():
    root = Path(__file__).resolve().parents[3]
    source = (root / "csrc/sidecard_execution/bindings.cpp").read_text()
    declarations = re.findall(r"ops\.def\(((?:\s*\"(?:[^\"\\]|\\.)*\")+)\);", source)
    schemas = [
        torch._C.parse_schema("".join(ast.literal_eval(literal) for literal in re.findall(r'"[^"\n]*"', value)))
        for value in declarations
    ]
    assert [schema.name for schema in schemas] == list(build.TORCH_OPERATORS)
    assert schemas[0].arguments[-1].alias_info.is_write
    assert schemas[0].returns[0].alias_info is None
    assert "TORCH_LIBRARY_FRAGMENT(_C_ascend" in source
    assert "TORCH_LIBRARY(_C_ascend" not in source
    assert "TORCH_LIBRARY_IMPL(_C_ascend, Meta" in source
    assert "TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1" in source


def test_lazy_help_without_accelerator_imports():
    root = Path(__file__).resolve().parents[3]
    script = (
        "import sys; import tools.sidecard.execution_kernel_build; "
        "assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], cwd=root, check=True)
    result = subprocess.run(
        [sys.executable, "-m", "tools.sidecard.execution_kernel_build", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--prepare-only" in result.stdout and "--build-prepared" in result.stdout


@pytest.fixture
def prepare_lane(tmp_path):
    """Run only the shell/CMake forwarding layer; cmake/make compilers are fakes."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake script-mode cache validation requires CMake")
    repo = Path(__file__).resolve().parents[3]
    source = tmp_path / "source tree/csrc"
    script = source / "cmake/scripts/prepare.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(repo / "csrc/cmake/scripts/prepare.sh", script)
    cache = source / "sidecard_execution/prepare-cache.cmake"
    cache.parent.mkdir()
    shutil.copyfile(repo / "csrc/sidecard_execution/prepare-cache.cmake", cache)
    third_party = source / "third_party"
    third_party.mkdir()
    output = tmp_path / "nested build with spaces"
    cmake_log, make_log = tmp_path / "cmake-args.json", tmp_path / "make-args.json"
    values = tmp_path / "cache-values.json"
    probe = tmp_path / "inspect-cache.cmake"
    probe.write_text(
        "file(WRITE [=[" + str(values) + "]=] "
        '"{\\"kda\\":\\"${SIDECARD_BUILD_KDA_V310}\\",'
        '\\"pre\\":\\"${SIDECARD_BUILD_MHC_PRE_V310}\\",'
        '\\"post\\":\\"${SIDECARD_BUILD_MHC_POST_V310}\\",'
        '\\"include\\":\\"${ASCEND_PROTOBUF_SHARED_INCLUDE}\\",'
        '\\"protoc\\":\\"${ASCEND_PROTOC}\\",'
        '\\"rebuild\\":\\"${FORCE_REBUILD_CANN_3RD}\\",'
        '\\"type\\":\\"${CMAKE_BUILD_TYPE}\\"}")\n'
    )
    fake = tmp_path / "fake-bin"
    fake.mkdir()
    common = "#!/usr/bin/env python3\nimport json,sys,subprocess\nfrom pathlib import Path\n"
    cmake_program = write(
        fake / "cmake",
        common
        + f"Path({str(cmake_log)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        + "if '-C' in sys.argv:\n"
        + f"    subprocess.run([{cmake!r},'-C',sys.argv[sys.argv.index('-C')+1],'-P',{str(probe)!r}],check=True)\n",
    )
    make_program = write(fake / "make", common + f"Path({str(make_log)!r}).write_text(json.dumps(sys.argv[1:]))\n")
    cmake_program.chmod(0o755)
    make_program.chmod(0o755)
    return dict(
        source=source,
        script=script,
        cache=cache,
        output=output,
        third_party=third_party,
        cmake_log=cmake_log,
        make_log=make_log,
        values=values,
        cmake=cmake,
        environment=dict(os.environ, PATH=str(fake) + os.pathsep + os.environ["PATH"]),
    )


def invoke_prepare(lane, options=()):
    return subprocess.run(
        [
            "bash",
            str(lane["script"]),
            "-s",
            str(lane["source"]),
            "-b",
            str(lane["output"]),
            "-p",
            str(lane["source"] / "toolkit with spaces"),
            "--cann_3rd_lib_path",
            str(lane["third_party"]),
            "--ascend-op_name",
            "sidecard_kda_v310::sidecard_mhc_pre_v310::sidecard_mhc_post_v310",
            *options,
        ],
        env=lane["environment"],
        capture_output=True,
        text=True,
    )


def test_nested_prepare_loads_private_cache_flags_protobuf_paths_and_bounded_jobs(prepare_lane):
    lane = prepare_lane
    result = invoke_prepare(lane, ("--initial-cache", str(lane["cache"]), "--jobs", "2"))
    assert result.returncode == 0, result.stderr
    args = json.loads(lane["cmake_log"].read_text())
    assert args[:3] == ["-C", str(lane["cache"]), str(lane["source"])]
    assert "-DCUSTOM_ASCEND_CANN_PACKAGE_PATH=" + str(lane["source"] / "toolkit with spaces") in args
    assert "-DCANN_3RD_LIB_PATH=" + str(lane["third_party"]) in args
    assert "-DCMAKE_BUILD_TYPE=" not in args
    assert json.loads(lane["make_log"].read_text()) == ["-j2", "prepare_build"]
    values = json.loads(lane["values"].read_text())
    assert values == dict(
        kda="ON",
        pre="ON",
        post="ON",
        rebuild="OFF",
        type="Release",
        include=str(lane["third_party"] / "ascend_protobuf/src"),
        protoc=str(lane["third_party"] / "ascend_protobuf/bin"),
    )


def test_nested_prepare_default_omits_cache_and_preserves_existing_job_policy(prepare_lane):
    lane = prepare_lane
    result = invoke_prepare(lane)
    assert result.returncode == 0, result.stderr
    args = json.loads(lane["cmake_log"].read_text())
    assert args[0] == str(lane["source"]) and "-C" not in args and "-DCMAKE_BUILD_TYPE=" in args
    cpus = sum(line.startswith("processor") for line in Path("/proc/cpuinfo").read_text().splitlines())
    assert json.loads(lane["make_log"].read_text()) == [f"-j{cpus * 2}", "prepare_build"]
    assert not lane["values"].exists()


@pytest.mark.parametrize(
    "options", [("--jobs", "0"), ("--jobs", "17"), ("--jobs", "bad"), ("--initial-cache", "/missing/cache.cmake")]
)
def test_nested_prepare_rejects_bad_options_before_clean_or_tools(prepare_lane, options):
    lane = prepare_lane
    sentinel = write(lane["output"] / "sentinel", "preserve existing directory on invalid invocation")
    result = invoke_prepare(lane, options)
    assert result.returncode == 2 and sentinel.is_file()
    assert not lane["cmake_log"].exists() and not lane["make_log"].exists()


@pytest.mark.parametrize("private", [False, True])
def test_outer_config_forwards_private_cache_and_jobs_into_nested_script(prepare_lane, private):
    lane = prepare_lane
    repo = Path(__file__).resolve().parents[3]
    text = (repo / "csrc/cmake/config.cmake").read_text()
    start = text.index("        # Optional authenticated cache/profile")
    stop = text.index("        if (result)", start)
    script = lane["source"].parent / "outer-probe.cmake"
    settings = dict(
        CMAKE_CURRENT_SOURCE_DIR=lane["source"],
        CMAKE_CURRENT_BINARY_DIR=lane["output"].parent / "outer build",
        ASCEND_CANN_PACKAGE_PATH=lane["source"] / "toolkit with spaces",
        CANN_3RD_LIB_PATH=lane["third_party"],
        EP_ASCEND_OP_NAME="sidecard_kda_v310::sidecard_mhc_pre_v310::sidecard_mhc_post_v310",
        EP_ASCEND_COMPUTE_UNIT="ascend310p",
        BUILD_TYPE="Release",
    )
    script.write_text(
        "".join(f"set({key} [=[{value}]=])\n" for key, value in settings.items())
        + text[start:stop]
        + '\nif(result)\nmessage(FATAL_ERROR "nested forwarding failed")\nendif()\n'
    )
    options = ["-DSIDECARD_EXECUTION_INITIAL_CACHE=" + str(lane["cache"]), "-DSIDECARD_EXECUTION_PREPARE_JOBS=2"]
    result = subprocess.run(
        [lane["cmake"], *(options if private else []), "-P", str(script)],
        env=lane["environment"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    if private:
        assert json.loads(lane["values"].read_text())["kda"] == "ON"
        assert json.loads(lane["make_log"].read_text()) == ["-j2", "prepare_build"]
    else:
        assert "-C" not in json.loads(lane["cmake_log"].read_text())
        cpus = sum(line.startswith("processor") for line in Path("/proc/cpuinfo").read_text().splitlines())
        assert json.loads(lane["make_log"].read_text()) == [f"-j{cpus * 2}", "prepare_build"]
