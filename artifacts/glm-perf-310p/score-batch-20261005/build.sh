#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
# shellcheck disable=SC1091
source /usr/local/Ascend/cann-9.1.0/set_env.sh
set -u
study_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_root=/usr/local/Ascend/cann-9.1.0
g++ -O2 -std=c++17 "$study_root/compile-rtc.cpp" \
    "-I$cann_root/include" "-L$cann_root/lib64" -lascendcl -lacl_rtc \
    -o "$study_root/compile-rtc"
cp /home/matteius/experiments/glm-prompt-profile-20261005/score-cached.bin "$study_root/score-baseline.bin"
"$study_root/compile-rtc" "$study_root/kda-score-probe.cpp" "$study_root/score-cached.bin" \
    > "$study_root/compile.log" 2>&1
sha256sum "$study_root/kda-score-probe.cpp" "$study_root/score-baseline.bin" \
    "$study_root/score-cached.bin" > "$study_root/native-sha256.txt"
