# SPDX-License-Identifier: Apache-2.0
"""Build append-only QSA parent/shared-cache kernels in an immutable directory."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths

from tools.glm_perf.stage_qsa_accumulate_rows import transform as accumulate_rows
from tools.glm_perf.stage_qsa_output_rows import transform as output_rows
from tools.glm_perf.stage_qsa_shared_cache import stage
from tools.glm_perf.stage_qsa_softmax_heads import transform as softmax_heads

HERE = Path(__file__).resolve().parent
QUALIFIED_PARENT_SHA256 = "66baea1e2eb4c8601868a1f1fb347397ff75ac8e19f68d56dfa598b054d2ffb6"


def build(
    build_dir,
    source_root,
    cann_root,
    compiler,
    version,
    vector_output=False,
    vector_accumulate=False,
    softmax_head_batch=False,
):
    if any(type(flag) is not bool for flag in (vector_output, vector_accumulate, softmax_head_batch)):
        raise ValueError("vector_output must be boolean")
    if type(version) is not int or version <= 0:
        raise ValueError("version must be a positive integer")
    kernel_root = source_root / "csrc/attention/qsa_sparse_attention_v310/op_kernel"
    if (
        hashlib.sha256((kernel_root / "qsa_cube_sparse_attention_v310.h").read_bytes()).hexdigest()
        != QUALIFIED_PARENT_SHA256
    ):
        raise ValueError("QSA source differs from qualified parent")
    build_dir.mkdir(exist_ok=False)
    package = f"glm_qsa_shared_v{version}_helpers"
    helper = build_dir / package
    helper.mkdir()
    (helper / "__init__.py").write_text("# SPDX-License-Identifier: Apache-2.0\n")
    for name in (
        "qsa_shared_native.py",
        "qsa_shared_probe.py",
        "stage_qsa_shared_cache.py",
        "stage_qsa_output_rows.py",
        "stage_qsa_accumulate_rows.py",
        "stage_qsa_softmax_heads.py",
    ):
        shutil.copy2(HERE / name, helper / name)
    target = cann_root / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    options = [
        "--npu-arch=dav-2002",
        f"--sysroot={cann_root / 'tools/hcc/sysroot'}",
        f"-isystem{target}",
        f"-isystem{target / 'aarch64-target-linux-gnu'}",
    ]
    for variant in ("parent", "shared"):
        snapshot = build_dir / variant
        shutil.copytree(kernel_root, snapshot)
        if variant == "shared":
            header = snapshot / "qsa_cube_sparse_attention_v310.h"
            header.unlink()
            stage(kernel_root / header.name, header, QUALIFIED_PARENT_SHA256)
            if vector_output:
                header.write_text(output_rows(header.read_text()))
            if vector_accumulate:
                header.write_text(accumulate_rows(header.read_text()))
            if softmax_head_batch:
                header.write_text(softmax_heads(header.read_text()))
        entry = snapshot / "qsa_sparse_attention_v310.cpp"
        entry_source = entry.read_text()
        original = (
            "    REGISTER_TILING_DEFAULT(QsaSparseAttentionV310TilingData);\n"
            "    GET_TILING_DATA_WITH_STRUCT(QsaSparseAttentionV310TilingData, tilingData, tiling);"
        )
        if entry_source.count(original) != 1:
            raise ValueError("QSA entry tiling admission changed")
        fields = [
            "numTokens",
            "numQueryHeads",
            "numKvHeads",
            "headsPerTask",
            "taskTilesPerKvHead",
            "headDim",
            "cacheBlockSize",
            "cacheHeadDimBlocks",
            "maxBlocksPerSequence",
            "selectedGroupsWidth",
            "numRequests",
            "tasksPerCore",
            "taskCount",
            "scaleQ24",
        ]
        # ACLRTC has no generated GET_TILING_DATA_WITH_STRUCT macro. Preserve
        # the production 14-int64 ABI and read each field explicitly instead.
        replacement = (
            "    AscendC::GlobalTensor<int64_t> tilingGm;\n"
            "    tilingGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tiling));\n"
            "    QsaSparseAttentionV310TilingData tilingData;\n"
            + "\n".join(f"    tilingData.{field} = tilingGm.GetValue({i});" for i, field in enumerate(fields))
        )
        task_type = "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);"
        if entry_source.count(task_type) != 1:
            raise ValueError("QSA entry core annotation changed")
        entry.write_text(
            entry_source.replace(original, replacement).replace(
                task_type, "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);"
            )
        )
        subprocess.run(
            [
                str(compiler),
                str(snapshot / "qsa_sparse_attention_v310.cpp"),
                str(build_dir / (variant + ".bin")),
                *options,
                f"-I{snapshot}",
                *(["-DGLM_QSA_SHARED_CACHE"] if variant == "shared" else []),
                *(["-DGLM_QSA_OUTPUT_ROWS"] if variant == "shared" and vector_output else []),
                *(["-DGLM_QSA_ACCUMULATE_ROWS"] if variant == "shared" and vector_accumulate else []),
                *(["-DGLM_QSA_SOFTMAX_HEAD_BATCH"] if variant == "shared" and softmax_head_batch else []),
            ],
            check=True,
        )
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("build with the serving torch-npu environment")
    npu_root = Path(spec.origin).parent
    includes = [*include_paths(), str(npu_root / "include"), str(cann_root / "include"), sysconfig.get_path("include")]
    libraries = [*library_paths(), str(npu_root / "lib"), str(cann_root / "lib64")]
    namespace = f"glm_qsa_shared_v{version}"
    bridge = build_dir / (namespace + ".so")
    shutil.copy2(HERE / "reconstruction_bridge.cpp", build_dir / "reconstruction_bridge.cpp")
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"-DGLM_RECONSTRUCTION_NAMESPACE={namespace}",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *[f"-I{p}" for p in includes],
            str(build_dir / "reconstruction_bridge.cpp"),
            *[f"-L{p}" for p in libraries],
            *[f"-Wl,-rpath,{p}" for p in libraries],
            "-ltorch_npu",
            "-lascendcl",
            "-lc10",
            "-ltorch_cpu",
            "-ltorch",
            "-ltorch_python",
            "-o",
            str(bridge),
        ],
        check=True,
    )
    hashes = {
        str(p.relative_to(build_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(build_dir.rglob("*"))
        if p.is_file()
    }
    report = dict(
        namespace=namespace,
        helper_package=package,
        version=version,
        files=hashes,
        parent_sha256=QUALIFIED_PARENT_SHA256,
        extra_ub_bytes=256 if softmax_head_batch else 0,
        vector_output=vector_output,
        vector_accumulate=vector_accumulate,
        softmax_head_batch=softmax_head_batch,
        full_operator_evaluated=False,
        serving_evaluated=False,
    )
    (build_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "source-root", "cann-root", "compiler"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--vector-output", action="store_true")
    parser.add_argument("--vector-accumulate", action="store_true")
    parser.add_argument("--softmax-head-batch", action="store_true")
    args = parser.parse_args()
    build(
        args.build_dir,
        args.source_root,
        args.cann_root,
        args.compiler,
        args.version,
        args.vector_output,
        args.vector_accumulate,
        args.softmax_head_batch,
    )


if __name__ == "__main__":
    main()
