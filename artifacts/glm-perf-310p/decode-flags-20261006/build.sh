#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
source /srv/ai/bin/ascend-env.sh
study=/home/matteius/experiments/glm-decode-flags-20261006
native=/srv/ai/src/glm-l1-wide-build-20261004
export PATH=/srv/ai/venvs/qwen38-w4-test-ce1862/bin:$PATH
cmake -S "$native" -B "$study/build" \
  -DASCEND_COMPUTE_UNIT=ascend310p \
  -DCANN_3RD_LIB_PATH="$native/third_party" \
  '-DASCEND_OP_NAME=w2_swiglu_v310;w2_route_combine_v310' \
  -DBUILD_OPEN_PROJECT=ON -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON \
  -DENABLE_BUILT_IN=OFF -DENABLE_TEST=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$study/install" > "$study/configure.log" 2>&1
cmake --build "$study/build" --target package -j 8 > "$study/package-build.log" 2>&1
package="$study/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-x86_64.run"
mkdir -p "$study/opp"
cp -a "$package/packages/vendors" "$study/opp/"
cd "$study"
ninja > "$study/binding-build.log" 2>&1
