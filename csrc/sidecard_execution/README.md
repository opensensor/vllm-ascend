# Private sidecard execution kernel build

This addon registers exactly three new `_C_ascend` schemas using
`TORCH_LIBRARY_FRAGMENT`, with PrivateUse1 and native Meta implementations.
Load it alongside the original qualified base library. Do not build or load a
second copy of the baseline extension or its existing Torch schemas.

The KDA schema declares `Tensor(a!) state`; its device adapter mutates FP32
carry and returns independent FP16 output. Meta retains carry shape/dtype
without inventing a data write or reading a pointer. mHC pre Meta returns FP32
post, FP32 logits and FP16 hidden; post Meta returns FP32 residual. Device
adapters remain responsible for NPU ownership and KDA pointer-alias checks.

## Frozen input contracts

The dependency-free builder is `tools.sidecard.execution_kernel_build`.
All source inputs come from a frozen root and an externally SHA-anchored JSON
manifest, never from a compiler reading the live research checkout:

```json
{"schema_version": 1, "files": {"csrc/CMakeLists.txt": "SHA256"}}
```

The complete file map must include the candidate packages, this addon,
`tools/sidecard/execution_kernel_build.py`, and the full required
`csrc/cmake`, `csrc/common`, adapter and supporting build sources. Preparation
copies only listed files and rejects source symlinks, traversal, changed
bytes, omitted mandatory inputs and a builder differing from its frozen copy.

The separately SHA-anchored ABI manifest has this schema:

```text
schema_version: 1
kind: sidecard-execution-abi-v1
soc: ascend310p
cann_version: exact installed 9.1 version string
torch_version: exact installed 2.13 version string
torch_npu_version: exact installed 2.13 version string
cxx11_abi: 0 or 1
cxx11_abi_source: how the pinned flag was obtained
machine: x86_64 or aarch64
python_soabi: exact pinned interpreter SOABI
compiler_version: exact host C++ --version output
c_compiler_version: exact host C --version output
cann_compiler_version: exact CANN --version output
cmake_version: exact CMake --version output
paths: absolute cann_root, torch_root, torch_npu_root, python_executable,
       cmake, compiler, c_compiler, cann_compiler, cann_version_file,
       base_ops_library
files: map of absolute input-file paths to SHA256
```

`files` authenticates every non-directory entry in `paths`, TorchConfig.cmake,
`torch/lib/{libtorch.so,libtorch_cpu.so,libc10.so}`,
`torch_npu/lib/libtorch_npu.so`, and the selected CANN `include/acl/acl.h` and
`lib64/libascendcl.so`. Additional pinned input files can be included.
External toolkit/venv symlink paths or canonical targets are accepted and
rehash on admission. Keep the lexical venv Python path in `paths`; resolving
it to the system interpreter would lose its package environment.

TorchConfig.cmake need not contain a C++ ABI flag. Obtain the flag from the
pinned CPU Torch installation with backend autoload disabled when required,
record its source, then freeze this metadata receipt. The builder imports
neither Torch nor torch_npu. Its child metadata check uses importlib.metadata
and sysconfig rather than importing device packages.

## Preparation and compilation

Preparation executes no external programs and creates a new private output:

```bash
python -m tools.sidecard.execution_kernel_build \
  --source-root FROZEN --source-manifest SOURCES.json --source-sha256 SOURCE_SHA \
  --abi-manifest ABI.json --abi-sha256 ABI_SHA --output NEW_BUILD --prepare-only
```

Run the helper from the parent frozen runtime, not the new build's pristine
`source` copy. This keeps Python bytecode caches out of that exact input map.
The compiler lane can then consume the SHA of `execution-build.json`:

```bash
python -m tools.sidecard.execution_kernel_build \
  --build-prepared NEW_BUILD --manifest-sha256 PREPARED_RECEIPT_SHA
```

Only the parent orchestrator schedules this lane. No compiler or accelerator
runtime was invoked while implementing this helper. Native compilation and
device qualification remain pending.

The builder uses a pristine source copy and a separate private working copy.
It enables the three default-off CMake flags explicitly, selects only
`SidecardKdaV310`, `SidecardMhcPreV310`, and `SidecardMhcPostV310`, pins CANN
9.1/310P and the recorded Torch 2.13 ABI, and gives CMake a private cache and
install prefix. It builds a standalone addon rather than the repo's baseline
Torch extension. The install command stages files locally; no `.run` installer
is executed, no global OPP is modified, and no archive is extracted by the
helper.

`csrc/sidecard_execution/prepare-cache.cmake` is a required frozen source
input. The builder loads it with outer `cmake -C` and forwards its absolute
path through `SIDECARD_EXECUTION_INITIAL_CACHE` to nested
`prepare.sh --initial-cache`. This preserves the three opt-in flags and
prepared protobuf include/protoc paths in both CMake invocations. Paths
derive from the cache file's private source location. Release mode and
dependency reuse remain explicit. `SIDECARD_EXECUTION_PREPARE_JOBS` forwards
the builder's bounded 1..16 job count through `prepare.sh --jobs`; ordinary
builds without either option retain their existing cache/default job behavior.
Paths containing spaces remain single arguments. Neither option is an
environment variable.

Upstream CMake has dependency download fallbacks. The compiler lane refuses
to start unless the frozen map contains prepared dependency markers:

- `csrc/third_party/json/include/nlohmann/json.hpp`
- `csrc/third_party/abseil-cpp/absl/log/absl_log.h`
- `csrc/third_party/makeself/makeself.sh` and `makeself-header.sh`
- `csrc/third_party/ascend_protobuf/src/google/protobuf/message.h`
- `csrc/third_party/ascend_protobuf/bin/protoc`

Freeze the complete corresponding prepared trees, including generated
headers and the existing protoc binary, from a qualified build. Missing
markers are recorded during preparation; compilation stops before tools run.
No dependency download or extraction should be used to bypass that boundary.

## Runtime admission

```python
from tools.sidecard.execution_kernel_build import authenticate_build

manifest = authenticate_build(native_build_root, manifest_sha256)
```

This API rejects prepare-only, failed, relocated or drifted builds. It
authenticates source and ABI receipts, the fixed command plan and logs, all
copied/generated source files, compiler build files, addon files and every
staged OPP file. Required host libraries must have the correct ELF64 machine
and shared-library type. Operator inventory must contain exactly the three
new CANN names. Safe private compatibility symlinks are materialized into
file bytes before publication; escaping links are rejected.

The build tree's `compat/liboptiling.so` uses an install-relative target that
is dangling before installation. Only its exact
`lib/linux/<pinned machine>/libcust_opmaster_rt2.0.so` target is normalized
from the regular, privately owned `opp-build/libcust_opmaster_rt2.0.so`
output after checking its ELF type and machine. Other dangling links fail.
`package_link_normalizations` records the original literal target, actual
source and SHA256; runtime admission verifies those bytes independently of
the caller's UID. All links are checked before any are materialized.

Returned fields include `operators` (CANN names), `torch_operators` (Torch
names), absolute `base_ops_library`, relative `addon_library`, and relative
`opp_path`. The private vendor is derived from the source manifest hash. Set
`CANN_CUSTOM_OPP_PATH` to include this vendor directory alongside existing
qualified private OPP directories before loading base and addon. The addon
does not replace existing Sinkhorn, experts or other baseline operators.

CANN root workers ignore user-owned opapi libraries. Before the privileged
device lane, the parent may change ownership to root on **only the newly
created private vendor package**. Never change ownership or install files in
the global OPP tree. Ownership changes do not alter content checksums.

A completed build receipt certifies provenance, not numerical or performance
qualification. Hardware validation remains false in this receipt. T12 must
run device, Meta, graph, precision and memory gates before enabling candidates.
