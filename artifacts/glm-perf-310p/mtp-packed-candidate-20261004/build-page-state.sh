#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build-only: source files must already be staged from the shared main checkout.
set -eo pipefail
source /srv/ai/bin/ascend-env.sh
set -u
kernel_root=${1:?existing CANN operator source root}
runtime_root=${2:?existing runtime mirror root}
log_root=${3:?new build log directory}
mkdir -p "$log_root"
# prepare.sh invokes make; explicitly select its generator independently of
# the top-level Ninja build and supply nonempty third-party arguments.
export CMAKE_GENERATOR='Unix Makefiles'
cmake -S "$kernel_root" -B "$kernel_root/build-mtp-pages-20261005" -G Ninja \
  -DASCEND_COMPUTE_UNIT=ascend310p \
  '-DASCEND_OP_NAME=recurrent_gated_delta_rule_v310;causal_conv1d_v310' \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DCANN_3RD_LIB_PATH="$kernel_root/third_party" \
  -DBUILD_TYPE=Release -DVERSION=9.1.0 -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON \
  > "$log_root/configure.log" 2>&1
cmake --build "$kernel_root/build-mtp-pages-20261005" -j8 > "$log_root/build.log" 2>&1
cmake --install "$kernel_root/build-mtp-pages-20261005" \
  --prefix "$kernel_root/opp-mtp-pages-20261005" > "$log_root/install.log" 2>&1
cmake -S "$runtime_root" -B "$runtime_root/build-mtp-pages-binding" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
  -DPYTHON_INCLUDE_PATH=/usr/local/python3.12.13/include/python3.12 \
  -DCMAKE_PREFIX_PATH=/srv/ai/venvs/qwen38-w4-test-ce1862/lib/python3.12/site-packages/pybind11/share/cmake/pybind11 \
  -DTORCH_NPU_PATH=/srv/ai/venvs/qwen38-w4-test-ce1862/lib/python3.12/site-packages/torch_npu \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 -DSOC_VERSION=ascend310p1 \
  > "$log_root/binding-configure.log" 2>&1
cmake --build "$runtime_root/build-mtp-pages-binding" -j4 > "$log_root/binding-build.log" 2>&1
# Install the resulting binding only when the runtime is stopped; retain the
# prior binding together with its prior OPP for rollback.
