# Qwen38 single-card custom operator sources and rebuild recipe

The sources for the four operator packages in `qwen38-single-card-20261011`
are available in this fork. A normal clone contains everything needed to
restore the recovered source trees; no separate source download is required.
CANN, its Python build dependencies and third-party build dependencies remain
external. This is a source handoff for a CANN 9.2 port, not a qualified 9.2 release.

## Source locations

Current operator definitions and tiling are under `csrc/{attention,gmm,moe}`.
For example, `csrc/gmm/qwen_w4_a8_pack_v310/op_host/` contains the pack
operator's definition, inference and tiling sources. Handwritten APIs are
usually in an operator's `op_host/op_api/` directory.

The [historical source manifest](../../../../csrc/release_sources/qwen38_single_card_20261011/manifest.json)
pins the unchanged files to Git revision
`d9e59a91d19318b75f6a97262426d4d7268d22ff`. The adjacent `snapshots/` directory
contains the recovered differences and dependency sources. Per-package
manifests record the exact source inventory and SHA-256 of every file.
The preparation tool reads the pinned Git revision, so editing the working
checkout does not silently change a historical rebuild. A shallow clone may
need `git fetch --unshallow origin` to obtain that revision.

| Package argument | Release location | Operators | Build vendor name |
| --- | --- | ---: | --- |
| `coherent` | `operators/coherent` | 5 | `qwen38_batch256_coherent` |
| `retained` | `operators/retained` | 22 | `custom` |
| `gdn` | `operators/gdn-unified-v5-opp/vendors/qwen_gdn_unified_v5_transformer` | 4 | `qwen_gdn_unified_v5` |
| `embedded` | `runtime/vllm_ascend/_cann_ops_custom/vendors/custom_transformer` | 24 | `custom` |

The embedded selection covers all 24 requested operators. The retained
selection also contains the older `ChunkFwdO` and `QwenW8RouteCombineV310`.
GDN's ops-info JSON contains only two operators, but its build selection also
includes the two chunk operators with handwritten APIs. Use the recorded
selection for each package, rather than deriving all selections from JSON.

Both historical host/API libraries match the frozen release manifest for
each package: eight matching SHA-256 hashes. See the
[provenance](../../../../csrc/release_sources/qwen38_single_card_20261011/provenance.json).
Matching binaries identifies the historical build directories; it does not
prove bit-for-bit reproduction from the recovered sources. One retained
`QwenW8RouteCombineV310` device source was restored from the source shipped in
the release. Its original and restored hashes are recorded in
[source adjustments](../../../../csrc/release_sources/qwen38_single_card_20261011/source-adjustments.json).
All 247 compared kernel/common source files match their shipped versions after
that correction. Existing per-file notices and available licenses are preserved.

## Generated ACLNN APIs

Many operators do not have handwritten `op_api/*.cpp`. Their `*_def.cpp`
files compile into `libop_host_aclnn.so`; the target SDK's
`tools/opbuild/op_build` generates `build/autogen/aclnn_*.cpp` and headers.
`cmake/custom_build.cmake` invokes it with `OPS_PROTO_SEPARATE=1`,
`OPS_ACLNN_GEN=1` and `OPS_PROJECT_NAME=aclnn`. The generated `ops_aclnn`
objects and handwritten APIs link into `libcust_opapi.so`. The `cust_opmaster`
target produces `libcust_opmaster_rt2.0.so` from host/tiling sources.

The [reference_cann91 directory](../../../../csrc/release_sources/qwen38_single_card_20261011/reference_cann91)
contains the recovered generated API and prototype sources. These are for
inspection. A CANN 9.2 build must regenerate wrappers using the **9.2**
`op_build`; recompiling the old generated wrappers is not the migration recipe.
The source trees also include the fork's `aclnn_torch_adapter/op_api_common.h`,
which belongs to the PyTorch adapter and is separate from this generated glue.

## Prepare and build

From the repository root, prepare sources without a CANN installation:

```bash
python3 tools/qwen4exp/custom_op_sources.py embedded /tmp/qwen38-embedded-source
python3 tools/qwen4exp/custom_op_sources.py embedded /tmp/qwen38-embedded-source --verify
```

The output contains `csrc/` and a source receipt. Existing work directories
are rejected. The verification command checks every recorded input; build
outputs and newly fetched dependencies may coexist with those inputs.

For compilation, activate a Python build environment with `regex` and the
CANN build dependencies. Supply a complete SDK with its compiler, headers,
libraries and `op_build`. CMake, a C/C++ compiler, a build backend, `patch`
and archive utilities are also required. The source's third-party rules obtain
nlohmann/json 3.11.3, Abseil 20230802.1 and protobuf 25.1. Historical CATLASS
headers and makeself sources are included. An optional sixth argument supplies
a compatible third-party cache; use a fresh cache for a different compiler/libc.

```bash
source /absolute/path/to/cann-9.2.0/set_env.sh
bash tools/qwen4exp/build_custom_ops.sh embedded /absolute/path/to/cann-9.2.0 \
  /tmp/qwen38-embedded-host host 4
```

Repeat with `coherent`, `retained` and `gdn`, using a new work directory for
each invocation. `host` selects `--jit` and builds a package without precompiled
device kernels. To build a full candidate, use `full` in a separate directory:

```bash
bash tools/qwen4exp/build_custom_ops.sh embedded /absolute/path/to/cann-9.2.0 \
  /tmp/qwen38-embedded-full full 4
```

The wrapper restores the checksummed sources, pins SDK discovery and calls:

```bash
bash build.sh --pkg --ops=<original-comma-separated-list> \
  --soc=ascend310p --vendor_name=<original-vendor> \
  --package-path <target-sdk> --cann_3rd_lib_path=<dependency-directory> -j4
```

Host-only adds `--jit`. Standalone `--ophost`/`--opapi` select the built-in
library workflow, not this custom vendor package recipe. Outputs include
`csrc/build/libcust_opmaster_rt2.0.so`, `csrc/build/libcust_opapi.so`, generated
sources in `csrc/build/autogen/`, and a `.run` package. The wrapper records the
command, log, library hashes, API exports and ELF symbol-version requirements.
It does not install packages or access an NPU.

The original target was CANN 9.1.0 on x86_64/ascend310p. Source `version.info`
retains that version; it does not establish which SDK a new build used.
Record the SDK version, CMake cache and resolved libraries with the port.
Package metadata may need updating as part of a completed 9.2 port.

## Package coherence and validation

Stage packages into separate candidate roots with the installer's
`--install-path`. Both retained and embedded use `custom_transformer`; installing
them into one vendor directory would overwrite one with the other. Preserve
the bundle's lookup order in `ASCEND_CUSTOM_OPP_PATH` and API library paths:
**GDN → coherent → embedded → retained**.

The embedded 24-op package passed a fresh host-only CANN 9.1.0 build with
Python 3.12.13 and GCC 15.2.0, including `.run` packaging. It exports all 24
expected `GetWorkspaceSize` symbols. All 18 regenerated API headers match the
historical headers byte-for-byte. See the
[validation record](../../../../csrc/release_sources/qwen38_single_card_20261011/validation.json)
and its adjacent `validation/embedded-host-cann91/` receipts. The initial
system-Python attempt lacked `regex`; a later packaging failure established
that makeself must be present in the source tree. Both are addressed here.

The other three package builds, full device-kernel rebuilding, CANN 9.2
compilation and inference with rebuilt packages remain unvalidated. Before
deployment, check binding signatures, loaded-library paths, FP32 recurrent
state and production prefill shapes. Then exercise TP2/EP2 startup, the first
real request, cold multi-chunk prefill, prefix reuse, MTP2/decode graphs and
sustained correct generation using the [runtime runbook](qwen38_310p_runtime_runbook.md).
Upstream GDN/KDA operators require explicit ABI, layout and numerical comparison
before substitution.

Josh's glibc 2.43 observation applies to the tested binaries until their ELF
dependencies establish the precise requirement. Preserve `readelf --version-info`
output and the actual load error instead of treating that observation as a
universal CANN 9.2 minimum.
