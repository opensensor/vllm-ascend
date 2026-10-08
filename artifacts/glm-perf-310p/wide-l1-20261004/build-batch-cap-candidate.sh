#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compile only. Stage the main grouped tiling source in build_root first.
set -eo pipefail
source /srv/ai/bin/ascend-env.sh
set -u
build_root=/srv/ai/src/glm-l1-wide-build-20261004
build_dir="$build_root/build-pipelined-20261004"
package="$build_root/opp-l1-w3-batch2560-20261004"
record=/home/matteius/experiments/glm-w3-20261004/wide-l1-20261004/opp-l1-w3-batch2560-20261004
if [[ -e "$package" || -e "$record" ]]; then
  echo "refusing to overwrite batch-cap experiment" >&2
  exit 1
fi
mkdir -p "$record"
previous_cxx_flags=$(sed -n 's/^CMAKE_CXX_FLAGS:STRING=//p' "$build_dir/CMakeCache.txt")
restore_flags() {
  cmake -S "$build_root" -B "$build_dir" "-DCMAKE_CXX_FLAGS=$previous_cxx_flags" > "$record/restore-configure.log" 2>&1
}
trap restore_flags EXIT
cmake -S "$build_root" -B "$build_dir" \
  "-DCMAKE_CXX_FLAGS=$previous_cxx_flags -DGLM_W2_GROUPED_MAX_ROUTES=20480" \
  > "$record/configure.log" 2>&1
cmake --build "$build_dir" --target cust_opmaster -j4 > "$record/build.log" 2>&1
python3 - "$build_root" "$package" "$record" <<'PY'
from pathlib import Path
import hashlib
import json
import shutil
import sys

root, package, record = map(Path, sys.argv[1:])
shutil.copytree(root / 'opp-l1-w3-v2-20261004', package)
lib = root / 'build-pipelined-20261004/libcust_opmaster_rt2.0.so'
for name in ('lib/linux/x86_64/libcust_opmaster_rt2.0.so', 'liboptiling.so'):
    shutil.copy2(lib, package / 'vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling' / name)
source = root / 'gmm/w2_grouped_blocked_dequant_matmul_v310/op_host/w2_grouped_blocked_dequant_matmul_v310_tiling.cpp'
shutil.copy2(source, record / 'tested-tiling.cpp')
data = {
    'base_package': 'opp-l1-w3-v2-20261004', 'max_routes': 20480,
    'host_compile_flag': '-DGLM_W2_GROUPED_MAX_ROUTES=20480',
    'tiling_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'tiling_library_sha256': hashlib.sha256(lib.read_bytes()).hexdigest(),
}
(record / 'package.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(data))
PY
