#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
# shellcheck disable=SC1091
source /usr/local/Ascend/cann-9.1.0/set_env.sh
set -u
study_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
compiler=/home/matteius/experiments/glm-score-batch-20261005/compile-rtc
for name in swiglu combine mhc_post; do
    "$compiler" "$study_root/$name.cpp" "$study_root/$name.bin" > "$study_root/$name-build.log" 2>&1
done
sha256sum "$study_root"/*.bin > "$study_root/binary-hashes.txt"
