#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compiler/package work only. Does not load an extension, initialize an NPU,
# modify the server's OPP environment, or install into a serving runtime.
set -eo pipefail
source /srv/ai/bin/ascend-env.sh
set -u
kernel_root=${1:?CANN source mirror containing the new operator}
build_root=${2:?new isolated build directory}
package_root=${3:?new isolated output package directory}
export CMAKE_GENERATOR='Unix Makefiles'
mkdir -p "$build_root"
nice -n 15 cmake -S "$kernel_root" -B "$build_root" -G Ninja \
  -DASCEND_COMPUTE_UNIT=ascend310p -DASCEND_OP_NAME=glm_kpool_prefill_score_v310 \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DCANN_3RD_LIB_PATH="$kernel_root/third_party" \
  -DBUILD_TYPE=Release -DVERSION=9.1.0 -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON \
  > "$build_root/configure.log" 2>&1
nice -n 15 cmake --build "$build_root" -j2 > "$build_root/build.log" 2>&1
cmake --install "$build_root" --prefix "$package_root" > "$build_root/install.log" 2>&1
