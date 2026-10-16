# SPDX-License-Identifier: Apache-2.0
"""Prepare/build private 310P execution kernels from SHA-anchored snapshots.

No accelerator modules are imported. Preparation executes no external programs.
The compiler lane uses fixed argv, a new private directory, CMake's local
staging prefix, and a three-schema addon; it never executes .run installers.
Only authenticate_build() accepts a completed, unchanged build for runtime.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
from pathlib import Path

KIND = "sidecard-execution-build-v1"
MANIFEST = "execution-build.json"
OPERATORS = ("SidecardKdaV310", "SidecardMhcPreV310", "SidecardMhcPostV310")
TORCH_OPERATORS = ("sidecard_kda_decode_310", "sidecard_mhc_pre_310", "sidecard_mhc_post_310")
DIRECTORIES = (
    "attention/sidecard_kda_v310",
    "attention/sidecard_mhc_pre_v310",
    "gmm/sidecard_mhc_post_v310",
)
FLAGS = ("SIDECARD_BUILD_KDA_V310", "SIDECARD_BUILD_MHC_PRE_V310", "SIDECARD_BUILD_MHC_POST_V310")
ABI_PATHS = (
    "cann_root",
    "torch_root",
    "torch_npu_root",
    "python_executable",
    "cmake",
    "compiler",
    "c_compiler",
    "cann_compiler",
    "cann_version_file",
    "base_ops_library",
)
REQUIRED_SOURCES = (
    "csrc/CMakeLists.txt",
    "csrc/build.sh",
    "csrc/cmake/config.cmake",
    "csrc/cmake/custom_build.cmake",
    "csrc/cmake/func.cmake",
    "csrc/cmake/package.cmake",
    "csrc/common/include/cann_compat.h",
    "csrc/aclnn_torch_adapter/op_api_common.h",
    "csrc/aclnn_torch_adapter/NPUBridge.h",
    "csrc/aclnn_torch_adapter/NPUBridge.cpp",
    "csrc/aclnn_torch_adapter/NPUStorageImpl.h",
    "csrc/aclnn_torch_adapter/NPUStorageImpl.cpp",
    "csrc/sidecard_execution/bindings.cpp",
    "csrc/sidecard_execution/meta.h",
    "csrc/sidecard_execution/prepare-cache.cmake",
    "csrc/cmake/scripts/prepare.sh",
    "tools/sidecard/execution_kernel_build.py",
)
ABI_VERSIONS = ("compiler_version", "c_compiler_version", "cann_compiler_version", "cmake_version")
PREPARED_DEPENDENCIES = (
    "csrc/third_party/json/include/nlohmann/json.hpp",
    "csrc/third_party/abseil-cpp/absl/log/absl_log.h",
    "csrc/third_party/makeself/makeself.sh",
    "csrc/third_party/makeself/makeself-header.sh",
    "csrc/third_party/ascend_protobuf/src/google/protobuf/message.h",
    "csrc/third_party/ascend_protobuf/bin/protoc",
)
METADATA_SCRIPT = (
    "import importlib.metadata as m,json,platform,sysconfig; "
    "print(json.dumps({'torch_version':m.version('torch'),"
    "'torch_npu_version':m.version('torch_npu'),'machine':platform.machine(),"
    "'python_soabi':sysconfig.get_config_var('SOABI')}))"
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def checked_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("expected lowercase SHA256 anchor")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key: " + key)
        result[key] = value
    return result


def read_json(path, sha):
    if not Path(path).is_file():
        raise ValueError("manifest must be a regular file: " + str(path))
    if digest(path) != checked_sha(sha):
        raise ValueError("manifest SHA256 changed: " + str(path))
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def absolute(value):
    if not isinstance(value, str) or not value or any(char in value for char in "\x00\n\r;"):
        raise ValueError("invalid absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("expected absolute path without traversal")
    return path


def inside(root, name):
    """Snapshot/output files must be regular, with no symlink path components."""
    if not isinstance(name, str) or not name or any(char in name for char in "\x00\n\r;\\"):
        raise ValueError("invalid relative file path")
    relative = Path(name)
    if relative.is_absolute() or any(part in (".", "..") for part in relative.parts):
        raise ValueError("relative path escapes private root")
    if relative.as_posix() != name:
        raise ValueError("noncanonical relative path")
    path = Path(root)
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink in private source/output: " + name)
    if not path.resolve().is_relative_to(Path(root).resolve()) or path == Path(root):
        raise ValueError("relative path escapes private root")
    return path


def authenticate_files(root, records, *, exact=False):
    if not isinstance(records, dict) or not records:
        raise ValueError("nonempty source/file inventory required")
    for relative, checksum in records.items():
        path = inside(root, relative)
        if not path.is_file() or digest(path) != checked_sha(checksum):
            raise ValueError("file authentication failed: " + relative)
    if exact:
        actual = set()
        for path in Path(root).rglob("*"):
            if path.is_symlink():
                raise ValueError("symlink in private inventory")
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
        if actual != set(records):
            raise ValueError("unmanifested or missing private files")


def authenticate_abi(path, sha):
    abi = read_json(path, sha)
    if abi.get("schema_version") != 1 or abi.get("kind") != "sidecard-execution-abi-v1":
        raise ValueError("unsupported execution ABI receipt")
    versions = {"cann_version": "9.1", "torch_version": "2.13", "torch_npu_version": "2.13"}
    for field, prefix in versions.items():
        value = abi.get(field)
        if not isinstance(value, str) or not (value == prefix or value.startswith(tuple(prefix + c for c in ".+-"))):
            raise ValueError("unsupported pinned ABI version: " + field)
    if abi.get("soc") != "ascend310p" or abi.get("machine") not in ("x86_64", "aarch64"):
        raise ValueError("execution build requires 310P and a supported host ABI")
    if type(abi.get("cxx11_abi")) is not int or abi["cxx11_abi"] not in (0, 1):
        raise ValueError("exact torch C++ ABI flag required")
    for field in (*ABI_VERSIONS, "python_soabi", "cxx11_abi_source"):
        if not isinstance(abi.get(field), str) or not abi[field].strip():
            raise ValueError("missing pinned ABI metadata: " + field)
    paths, files = abi.get("paths"), abi.get("files")
    if not isinstance(paths, dict) or set(paths) != set(ABI_PATHS) or not isinstance(files, dict) or not files:
        raise ValueError("incomplete ABI path/file inventory")
    for name in ABI_PATHS:
        absolute(paths[name])
    for name, checksum in files.items():
        target = absolute(name)
        if not target.is_file() or digest(target) != checked_sha(checksum):
            raise ValueError("ABI file authentication failed: " + name)
    roots = {name: Path(paths[name]) for name in ("cann_root", "torch_root", "torch_npu_root")}
    if not all(root.is_dir() for root in roots.values()):
        raise ValueError("ABI package roots must exist")
    required = [paths[name] for name in ABI_PATHS if name not in roots]
    required += [
        str(roots["torch_root"] / name)
        for name in (
            "share/cmake/Torch/TorchConfig.cmake",
            "lib/libtorch.so",
            "lib/libtorch_cpu.so",
            "lib/libc10.so",
        )
    ]
    required += [
        str(roots["torch_npu_root"] / "lib/libtorch_npu.so"),
        str(roots["cann_root"] / "include/acl/acl.h"),
        str(roots["cann_root"] / "lib64/libascendcl.so"),
    ]
    for name in required:
        # External ABI inputs may use toolkit/venv symlinks. Accept a receipt
        # keyed by that lexical path or its canonical target, then hash again.
        expected = files.get(name, files.get(str(Path(name).resolve())))
        if expected is None or digest(name) != expected:
            raise ValueError("required ABI file lacks authentication: " + name)
    if not Path(paths["cann_compiler"]).resolve().is_relative_to(roots["cann_root"].resolve()):
        raise ValueError("CANN compiler escapes pinned toolkit")
    if not Path(paths["cann_version_file"]).resolve().is_relative_to(roots["cann_root"].resolve()):
        raise ValueError("CANN version file escapes pinned toolkit")
    if abi["cann_version"] not in Path(paths["cann_version_file"]).read_text():
        raise ValueError("CANN version receipt disagrees with toolkit metadata")
    return abi


def command_plan(root, abi, source_sha, jobs):
    """Only fixed build commands; no shell fragments, installer or extra flags."""
    root = Path(root)
    paths = abi["paths"]
    source = root / "work-source"
    cann, torch, npu = (Path(paths[name]) for name in ("cann_root", "torch_root", "torch_npu_root"))
    vendor = "sidecard_exec_" + source_sha[:16]
    op_names = [Path(name).name for name in DIRECTORIES]
    initial_cache = source / "csrc/sidecard_execution/prepare-cache.cmake"
    configure = [paths["cmake"], "-C", str(initial_cache), "-S", str(source / "csrc"), "-B", str(root / "opp-build")]
    configure += ["-D" + name + "=ON" for name in FLAGS]
    configure += [
        "-DBUILD_OPEN_PROJECT=ON",
        "-DBUILD_OPS_RTY_KERNEL=OFF",
        "-DENABLE_BUILT_IN=OFF",
        "-DENABLE_BUILD_PKG=OFF",
        "-DENABLE_OPS_HOST=ON",
        "-DENABLE_OPS_KERNEL=ON",
        "-DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TYPE=Release",
        "-DSIDECARD_EXECUTION_INITIAL_CACHE=" + str(initial_cache),
        "-DSIDECARD_EXECUTION_PREPARE_JOBS=" + str(jobs),
        "-DENABLE_TEST=OFF",
        "-DASCEND_COMPUTE_UNIT=ascend310p",
        "-DASCEND_OP_NAME=" + ";".join(op_names),
        "-DVENDOR_NAME=" + vendor,
        "-DCUSTOM_ASCEND_CANN_PACKAGE_PATH=" + str(cann),
        "-DPython3_EXECUTABLE=" + paths["python_executable"],
        "-DHI_PYTHON=" + paths["python_executable"],
        "-DCMAKE_CXX_COMPILER=" + paths["compiler"],
        "-DCMAKE_C_COMPILER=" + paths["c_compiler"],
        "-DCMAKE_INSTALL_PREFIX=" + str(root / "opp"),
        "-DVLLM_ASCEND_BUILD_CACHE_DIR=" + str(root / "build-cache"),
        "-DENABLE_CCACHE=OFF",
        "-DCANN_3RD_LIB_PATH=" + str(source / "csrc/third_party"),
        "-DFORCE_REBUILD_CANN_3RD=OFF",
        "-DASCEND_PROTOBUF_SHARED_INCLUDE=" + str(source / "csrc/third_party/ascend_protobuf/src"),
        "-DASCEND_PROTOC=" + str(source / "csrc/third_party/ascend_protobuf/bin"),
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
    ]
    addon = [
        paths["compiler"],
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        "-Wl,-Bsymbolic",
        "-Wl,-z,defs",
        "-DASCEND_PLATFORM_310P",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi['cxx11_abi']}",
        "-include",
        str(source / "csrc/common/include/cann_compat.h"),
    ]
    for include in (
        source / "csrc",
        source / "csrc/aclnn_torch_adapter",
        cann / "include",
        torch / "include",
        torch / "include/torch/csrc/api/include",
        npu / "include",
    ):
        addon += ["-I", str(include)]
    addon += [
        str(source / "csrc/sidecard_execution/bindings.cpp"),
        str(source / "csrc/aclnn_torch_adapter/NPUBridge.cpp"),
        str(source / "csrc/aclnn_torch_adapter/NPUStorageImpl.cpp"),
    ]
    for library_root in (torch / "lib", npu / "lib", cann / "lib64"):
        addon += ["-L", str(library_root), "-Wl,-rpath," + str(library_root)]
    addon += [
        "-ltorch_npu",
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
        "-lascendcl",
        "-ldl",
        "-pthread",
        "-o",
        str(root / "addon/libsidecard_execution.so"),
    ]
    return [
        dict(name="host-cxx-version", argv=[paths["compiler"], "--version"]),
        dict(name="host-c-version", argv=[paths["c_compiler"], "--version"]),
        dict(name="cann-compiler-version", argv=[paths["cann_compiler"], "--version"]),
        dict(name="cmake-version", argv=[paths["cmake"], "--version"]),
        dict(name="python-package-abi", argv=[paths["python_executable"], "-c", METADATA_SCRIPT]),
        dict(name="configure-private-opp", argv=configure),
        dict(
            name="compile-private-opp",
            argv=[paths["cmake"], "--build", str(root / "opp-build"), "--parallel", str(jobs)],
        ),
        dict(
            name="stage-private-opp",
            argv=[paths["cmake"], "--install", str(root / "opp-build"), "--prefix", str(root / "opp")],
        ),
        dict(name="compile-private-addon", argv=addon),
    ]


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in published private inventory")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def write_manifest(root, manifest):
    target = Path(root) / MANIFEST
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def prepare(source_root, source_manifest, source_sha256, abi_manifest, abi_sha256, output, *, jobs=2):
    """Copy only authenticated frozen inputs; compiler/runtime execution is absent."""
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    frozen = read_json(source_manifest, source_sha256)
    if frozen.get("schema_version") != 1:
        raise ValueError("unsupported frozen source manifest")
    records = frozen.get("files")
    authenticate_files(source_root, records)
    required = set(REQUIRED_SOURCES)
    for directory in DIRECTORIES:
        required.update(("csrc/" + directory + "/CMakeLists.txt", "csrc/" + directory + "/op_host/CMakeLists.txt"))
        if not any(name.startswith("csrc/" + directory + "/op_kernel/") for name in records):
            raise ValueError("candidate kernel sources missing: " + directory)
    if not required <= records.keys():
        raise ValueError("frozen source omits build dependencies: " + ",".join(sorted(required - records.keys())))
    if digest(__file__) != records["tools/sidecard/execution_kernel_build.py"]:
        raise ValueError("running builder differs from authenticated frozen helper")
    abi = authenticate_abi(abi_manifest, abi_sha256)
    if type(jobs) is not int or not 1 <= jobs <= 16:
        raise ValueError("private build jobs must be an integer from 1 through 16")
    forbidden = [
        source_root,
        *(Path(abi["paths"][name]).resolve() for name in ("cann_root", "torch_root", "torch_npu_root")),
    ]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in forbidden):
        raise ValueError("output must be separate from source and installed ABI roots")
    if output.exists():
        raise ValueError("private build output must be a new directory")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name in records:
        destination = inside(source, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(source_root, name), destination)
    authenticate_files(source, records, exact=True)
    shutil.copytree(source, output / "work-source")
    shutil.copyfile(source_manifest, output / "source-manifest.json")
    shutil.copyfile(abi_manifest, output / "abi-manifest.json")
    vendor = "sidecard_exec_" + source_sha256[:16] + "_transformer"
    manifest = dict(
        schema_version=1,
        kind=KIND,
        build_root=str(output),
        build_complete=False,
        hardware_validated=False,
        source_manifest_sha256=source_sha256,
        abi_manifest_sha256=abi_sha256,
        source_files=records,
        operators=list(OPERATORS),
        torch_operators=list(TORCH_OPERATORS),
        vendor=vendor,
        jobs=jobs,
        addon_library="addon/libsidecard_execution.so",
        opp_path="opp/packages/vendors/" + vendor,
        base_ops_library=abi["paths"]["base_ops_library"],
        abi=abi,
        commands=command_plan(output, abi, source_sha256, jobs),
        package_files={},
        addon_files={},
        command_logs={},
        build_files={},
        work_source_files={},
        missing_prepared_dependencies=[name for name in PREPARED_DEPENDENCIES if name not in records],
        ownership_note="Before a privileged device lane, chown only this new private vendor package "
        "to root: CANN ignores user-owned opapi libraries in root workers. Never change global OPP ownership.",
    )
    write_manifest(output, manifest)
    return manifest


def authenticate_prepared(root, sha):
    root = Path(root).resolve()
    manifest = read_json(inside(root, MANIFEST), sha)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != KIND or manifest.get("build_root") != str(root):
        raise ValueError("unsupported or relocated execution build receipt")
    source = read_json(inside(root, "source-manifest.json"), manifest["source_manifest_sha256"])
    if source.get("files") != manifest.get("source_files"):
        raise ValueError("source inventory changed")
    authenticate_files(root / "source", manifest["source_files"], exact=True)
    abi = authenticate_abi(inside(root, "abi-manifest.json"), manifest["abi_manifest_sha256"])
    if abi != manifest.get("abi") or manifest.get("base_ops_library") != abi["paths"]["base_ops_library"]:
        raise ValueError("execution build ABI identity changed")
    if manifest.get("operators") != list(OPERATORS) or manifest.get("torch_operators") != list(TORCH_OPERATORS):
        raise ValueError("execution operator set changed")
    vendor = "sidecard_exec_" + manifest["source_manifest_sha256"][:16] + "_transformer"
    if (
        manifest.get("vendor") != vendor
        or manifest.get("opp_path") != "opp/packages/vendors/" + vendor
        or manifest.get("addon_library") != "addon/libsidecard_execution.so"
    ):
        raise ValueError("execution private output identity changed")
    if manifest.get("commands") != command_plan(root, abi, manifest["source_manifest_sha256"], manifest["jobs"]):
        raise ValueError("execution command plan changed")
    if type(manifest.get("jobs")) is not int or not 1 <= manifest["jobs"] <= 16:
        raise ValueError("invalid execution job count")
    expected_missing = [name for name in PREPARED_DEPENDENCIES if name not in manifest["source_files"]]
    if manifest.get("missing_prepared_dependencies") != expected_missing:
        raise ValueError("prepared dependency inventory changed")
    return manifest


def materialize_package_links(root, *, allowed_root=None, machine=None):
    """CANN installs a relative optiling compatibility link; publish file bytes.

    Only links into this new staging tree are accepted. The build tree's one
    install-relative optiling link may refer to its real private ELF output.
    Return receipts for that normalization; arbitrary dangling links still fail.
    """
    root = Path(root)
    allowed_root = root if allowed_root is None else Path(allowed_root)
    normalizations = {}
    pending = []
    for path in list(root.rglob("*")):
        if not path.is_symlink():
            continue
        target = path.resolve()
        literal = os.readlink(path)
        relative = path.relative_to(root).as_posix()
        expected = f"lib/linux/{machine}/libcust_opmaster_rt2.0.so"
        if (
            not target.exists()
            and machine in ("x86_64", "aarch64")
            and root == allowed_root / "opp-build"
            and relative == "compat/liboptiling.so"
            and literal == expected
        ):
            # custom_build.cmake creates this link for its installed location,
            # while the staging output is next to compat/. Never generalize
            # missing-target lookup by basename or follow an alternate symlink.
            target = inside(allowed_root, "opp-build/libcust_opmaster_rt2.0.so")
            if not target.is_file() or not target.stat().st_uid == root.stat().st_uid == allowed_root.stat().st_uid:
                raise ValueError("optiling normalization requires an owned private build library")
            validate_shared_library(target, machine)
            normalizations[relative] = dict(
                link_target=literal,
                source="libcust_opmaster_rt2.0.so",
                sha256=digest(target),
            )
        if not target.is_relative_to(allowed_root.resolve()) or not target.is_file():
            raise ValueError("installed package link escapes private prefix")
        pending.append((path, target.read_bytes()))
    # Validate every link before replacing any, so an unrelated invalid link
    # cannot cause a partially materialized private tree.
    for path, raw in pending:
        path.unlink()
        path.write_bytes(raw)
    return normalizations


def authenticate_link_normalizations(root, manifest):
    """Check normalized build-link provenance without caller ownership checks."""
    records = manifest.get("package_link_normalizations")
    if not isinstance(records, dict):
        raise ValueError("missing private package link normalization receipt")
    expected_path = "opp-build/compat/liboptiling.so"
    if set(records) - {expected_path}:
        raise ValueError("unexpected private package link normalization")
    if records:
        record = records[expected_path]
        machine = manifest["abi"]["machine"]
        if (
            not isinstance(record, dict)
            or set(record) != {"link_target", "source", "sha256"}
            or record["link_target"] != f"lib/linux/{machine}/libcust_opmaster_rt2.0.so"
            or record["source"] != "libcust_opmaster_rt2.0.so"
        ):
            raise ValueError("invalid private optiling normalization receipt")
        target = inside(root, "opp-build/" + record["source"])
        materialized = inside(root, expected_path)
        if digest(target) != checked_sha(record["sha256"]) or digest(materialized) != record["sha256"]:
            raise ValueError("private optiling normalization bytes changed")
        validate_shared_library(target, machine)


def validate_shared_library(path, machine):
    """Reject missing/text/wrong-host outputs without dlopen or device imports."""
    with Path(path).open("rb") as stream:
        header = stream.read(64)
    expected_machine = {"x86_64": 62, "aarch64": 183}[machine]
    if (
        len(header) < 64
        or header[:6] != b"\x7fELF\x02\x01"
        or struct.unpack_from("<HH", header, 16) != (3, expected_machine)
    ):
        raise ValueError("expected ELF64 host shared library: " + str(path))


def validate_package(root, vendor, machine):
    root = Path(root)
    vendors = root / "packages/vendors"
    if not vendors.is_dir() or {path.name for path in vendors.iterdir()} != {vendor}:
        raise ValueError("private OPP contains unexpected vendors")
    package = vendors / vendor
    if not (package / "op_api/lib/libcust_opapi.so").is_file():
        raise ValueError("private OPP lacks opapi library")
    validate_shared_library(package / "op_api/lib/libcust_opapi.so", machine)
    config = package / "op_impl/ai_core/tbe/config/ascend310p"
    op_files = [path for path in config.glob("*.json") if "ops-info" in path.name or "ops_info" in path.name]
    if not op_files:
        raise ValueError("private OPP lacks 310P operator inventory")
    names = set()
    for path in op_files:
        value = json.loads(path.read_bytes(), object_pairs_hook=unique_object)
        if not isinstance(value, dict):
            raise ValueError("invalid private operator inventory")
        names.update(name for name in value if not name.startswith("_"))
    if names != set(OPERATORS):
        raise ValueError("private OPP operator inventory differs from candidate-only set")
    # CANN releases also place binaries in op_impl/ai_core/tbe/<vendor>_impl.
    binaries = [path for path in package.rglob("*") if path.is_file() and path.suffix in (".o", ".bin")]
    if not binaries:
        raise ValueError("private OPP has no compiled device kernels")


def compile_prepared(root, sha):
    """Compiler lane only; parent owns its scheduling and later device tests."""
    root = Path(root).resolve()
    manifest = authenticate_prepared(root, sha)
    if manifest.get("build_complete") is not False or manifest.get("command_logs"):
        raise ValueError("compile requires a fresh prepare-only receipt")
    if manifest["abi"]["machine"] != platform.machine():
        raise ValueError("host machine differs from pinned ABI")
    if manifest.get("missing_prepared_dependencies"):
        raise ValueError(
            "authenticated prepared dependencies required; do not download/extract: "
            + ",".join(manifest["missing_prepared_dependencies"])
        )
    authenticate_files(root / "work-source", manifest["source_files"], exact=True)
    (root / "addon").mkdir(exist_ok=False)
    (root / "logs").mkdir(exist_ok=False)
    # Do not rewrite the host's environment. Retain its established CANN paths
    # only in these child processes; override directory/cache outputs privately.
    environment = dict(os.environ)
    paths = manifest["abi"]["paths"]
    environment["PATH"] = str(Path(paths["python_executable"]).parent) + os.pathsep + environment.get("PATH", "")
    environment["TMPDIR"] = str(root / "tmp")
    (root / "tmp").mkdir(exist_ok=False)
    version_fields = dict(
        zip(("host-cxx-version", "host-c-version", "cann-compiler-version", "cmake-version"), ABI_VERSIONS, strict=True)
    )
    try:
        for index, command in enumerate(manifest["commands"]):
            name = command["name"]
            log = root / "logs" / f"{index:02d}-{name}.log"
            with log.open("xb") as stream:
                subprocess.run(
                    command["argv"], cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True
                )
            manifest["command_logs"][log.relative_to(root).as_posix()] = digest(log)
            if name in version_fields and log.read_text().strip() != manifest["abi"][version_fields[name]].strip():
                raise ValueError("compiler/tool version changed: " + name)
            if name == "python-package-abi":
                metadata = json.loads(log.read_bytes())
                keys = {"torch_version", "torch_npu_version", "machine", "python_soabi"}
                if set(metadata) != keys or any(metadata[key] != manifest["abi"][key] for key in keys):
                    raise ValueError("pinned Python package ABI changed")
            write_manifest(root, manifest)
        # Re-authenticate compiler inputs after completion, before publishing.
        authenticate_abi(root / "abi-manifest.json", manifest["abi_manifest_sha256"])
        authenticate_files(root / "source", manifest["source_files"], exact=True)
        authenticate_files(root / "work-source", manifest["source_files"])
        materialize_package_links(root / "opp")
        normalizations = materialize_package_links(
            root / "opp-build", allowed_root=root, machine=manifest["abi"]["machine"]
        )
        manifest["package_link_normalizations"] = {
            "opp-build/" + name: record for name, record in normalizations.items()
        }
        validate_package(root / "opp", manifest["vendor"], manifest["abi"]["machine"])
        validate_shared_library(root / manifest["addon_library"], manifest["abi"]["machine"])
        manifest["package_files"] = inventory(root / "opp")
        manifest["addon_files"] = inventory(root / "addon")
        manifest["build_files"] = inventory(root / "opp-build")
        manifest["work_source_files"] = inventory(root / "work-source")
        if set(manifest["addon_files"]) != {"libsidecard_execution.so"}:
            raise ValueError("addon output inventory differs")
        manifest["build_complete"] = True
    except Exception as error:
        # Preserve the failed command's bytes as well as successful logs; an
        # incomplete receipt still cannot pass runtime admission.
        manifest["command_logs"] = {
            path.relative_to(root).as_posix(): digest(path) for path in sorted((root / "logs").glob("*.log"))
        }
        manifest["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write_manifest(root, manifest)
    return manifest


def authenticate_build(root, sha):
    """Runtime admission: completed native build, fixed plan, every input/output.

    This checks build provenance only. It does not assert numerical correctness,
    hardware qualification, or permission to enable a runtime row policy.
    """
    root = Path(root).resolve()
    manifest = authenticate_prepared(root, sha)
    if manifest.get("build_complete") is not True or "error" in manifest:
        raise ValueError("native execution build is incomplete or failed")
    authenticate_files(root / "opp", manifest["package_files"], exact=True)
    authenticate_files(root / "addon", manifest["addon_files"], exact=True)
    authenticate_files(root / "opp-build", manifest["build_files"], exact=True)
    authenticate_files(root / "work-source", manifest["work_source_files"], exact=True)
    authenticate_link_normalizations(root, manifest)
    authenticate_files(root, manifest["command_logs"])
    expected_logs = {f"logs/{index:02d}-{command['name']}.log" for index, command in enumerate(manifest["commands"])}
    if set(manifest["command_logs"]) != expected_logs:
        raise ValueError("incomplete command execution evidence")
    if set(manifest["addon_files"]) != {"libsidecard_execution.so"}:
        raise ValueError("addon output inventory differs")
    validate_package(root / "opp", manifest["vendor"], manifest["abi"]["machine"])
    validate_shared_library(root / manifest["addon_library"], manifest["abi"]["machine"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "source-manifest", "abi-manifest", "output", "build-prepared"):
        parser.add_argument("--" + name, type=Path)
    for name in ("source-sha256", "abi-sha256", "manifest-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.build_prepared is not None:
        if (
            not args.manifest_sha256
            or args.prepare_only
            or any(
                getattr(args, name) is not None
                for name in ("source_root", "source_manifest", "source_sha256", "abi_manifest", "abi_sha256", "output")
            )
        ):
            parser.error("--build-prepared requires only --manifest-sha256")
        result = compile_prepared(args.build_prepared, args.manifest_sha256)
        output = args.build_prepared
    else:
        required = ("source_root", "source_manifest", "source_sha256", "abi_manifest", "abi_sha256", "output")
        if any(getattr(args, name) is None for name in required) or args.manifest_sha256:
            parser.error("provide source root/manifest/SHA, ABI manifest/SHA, and a new output directory")
        result = prepare(*(getattr(args, name) for name in required), jobs=args.jobs)
        output = args.output
        if not args.prepare_only:
            result = compile_prepared(output, digest(output / MANIFEST))
    print(
        json.dumps(
            dict(
                build_complete=result["build_complete"],
                manifest_sha256=digest(output / MANIFEST),
                addon_library=result["addon_library"],
                opp_path=result["opp_path"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
