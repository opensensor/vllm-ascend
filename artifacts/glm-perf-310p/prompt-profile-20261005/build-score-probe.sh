#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compile fixed-shape score probes; this script does not open NPU devices.
set -euo pipefail
study_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_root=/usr/local/Ascend/cann-9.1.0/x86_64-linux
compiler=/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/bisheng
flags=(
    -x cce -O2 -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200
    "-I${cann_root}/asc" "-I${cann_root}/asc/include"
    "-I${cann_root}/asc/include/interface" "-I${cann_root}/asc/include/basic_api"
    "-I${cann_root}/ascendc/include/highlevel_api"
)
"${compiler}" "${flags[@]}" -c "${study_root}/kda-score-probe.cpp" \
    -o "${study_root}/score-baseline.o" > "${study_root}/compile-baseline.log" 2>&1
"${compiler}" "${flags[@]}" -DGLM_KDA_SCORE_CACHE_COLUMNS \
    -c "${study_root}/kda-score-probe.cpp" -o "${study_root}/score-cached.o" \
    > "${study_root}/compile-cached.log" 2>&1
sha256sum "${study_root}/kda-score-probe.cpp" "${study_root}/score-baseline.o" \
    "${study_root}/score-cached.o" > "${study_root}/native-probe-sha256.txt"
