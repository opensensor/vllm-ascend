# SPDX-License-Identifier: Apache-2.0
"""Build queued native variants without installing into any serving package."""

import json
import shutil
import subprocess
from pathlib import Path

root = Path("/home/matteius/experiments/glm-next-queue-20261005")
source = Path("/srv/ai/src/glm-l1-wide-build-20261004")
staged = root / "staged"
base = source / "opp-l1-overlap-20261004"
records = {}


def command(args, log):
    with log.open("w") as out:
        subprocess.run(args, cwd=source, stdout=out, stderr=subprocess.STDOUT, check=True)


def configure(build, ops, flags="", hostflags=""):
    build.mkdir(parents=True, exist_ok=True)
    command(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DASCEND_COMPUTE_UNIT=ascend310p",
            "-DASCEND_OP_NAME=" + ops,
            "-DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0",
            "-DCANN_3RD_LIB_PATH=" + str(source / "third_party"),
            "-DBUILD_TYPE=Release",
            "-DVERSION=9.1.0",
            "-DENABLE_OPS_HOST=ON",
            "-DENABLE_OPS_KERNEL=ON",
            "-DOPS_COMPILE_OPTIONS=" + flags,
            "-DCMAKE_CXX_FLAGS=" + hostflags,
        ],
        build / "configure.log",
    )


def snapshot_patch(patch, files):
    originals = {path: (source / path).read_bytes() if (source / path).exists() else None for path in files}
    subprocess.run(["git", "apply", "--check", "-p2", str(staged / patch)], cwd=source, check=True)
    subprocess.run(["git", "apply", "-p2", str(staged / patch)], cwd=source, check=True)
    patched = {path: (source / path).read_bytes() for path in files}
    return originals, patched


def restore(pair):
    originals, patched = pair
    for path, content in patched.items():
        if (source / path).read_bytes() != content:
            raise RuntimeError("source modified concurrently: " + path)
    for path, content in originals.items():
        if content is None:
            (source / path).unlink()
        else:
            (source / path).write_bytes(content)


def group_package(name, lanes=None, routes=None):
    build = root / ("build-" + name)
    package = root / ("opp-" + name)
    assert not package.exists()
    flags = "-DGLM_W2_GROUPED_L1_WIDE;-DGLM_W2_GROUPED_L1_WIDE_PIPELINED"
    if lanes:
        flags += ";-DGLM_W2_GROUPED_ADAPTIVE_EXPERT_LANES=" + str(lanes)
    configure(
        build,
        "w2_blocked_dequant_matmul_v310;w2_grouped_blocked_dequant_matmul_v310",
        flags,
        "-DGLM_W2_GROUPED_MAX_ROUTES=" + str(routes) if routes else "",
    )
    target = "cust_opmaster" if routes else "ops_transformer_kernel"
    command(["cmake", "--build", str(build), "--target", target, "-j2"], build / "build.log")
    shutil.copytree(base, package)
    vendor = package / "vendors/custom_transformer"
    if routes:
        library = build / "libcust_opmaster_rt2.0.so"
        for name in ("lib/linux/x86_64/libcust_opmaster_rt2.0.so", "liboptiling.so"):
            shutil.copy2(library, vendor / "op_impl/ai_core/tbe/op_tiling" / name)
    else:
        op = "w2_grouped_blocked_dequant_matmul_v310"
        binaries = build / ("binary/ascend310p/bin/" + op)
        objects = list(binaries.glob("*.o"))
        assert len(objects) == 2, objects
        for obj in objects:
            for path in (obj, obj.with_suffix(".json")):
                shutil.copy2(path, vendor / ("op_impl/ai_core/tbe/kernel/ascend310p/" + op) / path.name)
    return {"package": str(package), "flags": flags, "routes": routes}


def kda_package(name, ops, flags, hostflags):
    build = root / ("build-" + name)
    package = root / ("opp-" + name)
    assert not package.exists()
    configure(build, ops, flags, hostflags)
    target = "cust_opapi" if name == "kda-skip" else "ops_transformer_kernel"
    command(["cmake", "--build", str(build), "--target", target, "-j2"], build / "build.log")
    # Overlay the qualified full split package; replace only the intended artifact.
    shutil.copytree("/srv/ai/src/kda-persistent-scores-opp", package)
    vendor = package / "vendors/custom_transformer"
    if name == "kda-skip":
        libraries = list(build.rglob("libcust_opapi.so"))
        assert len(libraries) == 1, libraries
        shutil.copy2(libraries[0], vendor / "op_api/lib/libcust_opapi.so")
    else:
        op = "kda_gate_cumsum"
        objects = list((build / ("binary/ascend310p/bin/" + op)).glob("*.o"))
        assert objects
        destination = vendor / ("op_impl/ai_core/tbe/kernel/ascend310p/" + op)
        destination.mkdir(parents=True, exist_ok=True)
        for obj in objects:
            for path in (obj, obj.with_suffix(".json")):
                shutil.copy2(path, destination / path.name)
    return {"package": str(package), "flags": flags, "hostflags": hostflags}


def run(name, fn):
    print("BUILD", name, flush=True)
    try:
        records[name] = fn()
    except Exception as exc:
        records[name] = {"error": repr(exc)}
    (root / "build-results.json").write_text(json.dumps(records, indent=2))
    print("RESULT", name, records[name], flush=True)


pair = snapshot_patch(
    "adaptive-expert-teams.patch",
    [
        "gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel/w2_grouped_blocked_dequant_matmul_v310.cpp",
        "gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel/adaptive_expert_core_groups.h",
    ],
)
try:
    for lanes in (2, 4):
        run("adaptive" + str(lanes), lambda lanes=lanes: group_package("adaptive" + str(lanes), lanes=lanes))
    run("batch20480", lambda: group_package("batch20480", routes=20480))
finally:
    restore(pair)
pair = snapshot_patch("kda-prepare-head.patch", ["attention/kda_gate_cumsum/op_kernel/kda_gate_cumsum.cpp"])
try:
    run("kda-head", lambda: kda_package("kda-head", "kda_gate_cumsum", "-DGLM_KDA_GATE_PREPARE_HEAD", ""))
finally:
    restore(pair)
pair = snapshot_patch("kda-skip-safe-cube.patch", ["attention/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp"])
try:
    run(
        "kda-skip",
        lambda: kda_package("kda-skip", "chunk_kda_fwd;kda_gate_cumsum", "", "-DGLM_KDA_SKIP_SAFE_SCORE_CUBE"),
    )
finally:
    restore(pair)
