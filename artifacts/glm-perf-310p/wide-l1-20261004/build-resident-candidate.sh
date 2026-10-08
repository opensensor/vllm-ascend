#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compile only on threadripper after staging the kernel sources and headers from main.
# This script neither opens an NPU device nor changes the serving package.
set -eo pipefail
source /srv/ai/bin/ascend-env.sh
set -u

variant=${1:?w3, w3-prefill, row-reuse, scales, all-resident, teams-control, teams2, or teams4}
flags='-DGLM_W2_GROUPED_L1_WIDE;-DGLM_W2_GROUPED_L1_WIDE_PIPELINED'
case "$variant" in
  w3) flags+=';-DGLM_W2_GROUPED_L1_W3' ;;
  w3-prefill) flags+=';-DGLM_W2_GROUPED_L1_W3;-DGLM_W2_GROUPED_L1_W3_PREFILL_ONLY' ;;
  row-reuse) flags+=';-DGLM_W2_GROUPED_L1_W3;-DGLM_W2_GROUPED_L1_W3_PREFILL_ONLY;-DGLM_W2_GROUPED_L1_ROW_REUSE' ;;
  scales) flags+=';-DGLM_W2_GROUPED_L1_SCALE_CACHE' ;;
  all-resident) flags+=';-DGLM_W2_GROUPED_L1_W3;-DGLM_W2_GROUPED_L1_SCALE_CACHE;-DGLM_W2_GROUPED_L1_LARGE_GROUPS' ;;
  teams-control|teams2|teams4)
    flags+=';-DGLM_W2_GROUPED_L1_W3;-DGLM_W2_GROUPED_L1_SCALE_CACHE;-DGLM_W2_GROUPED_L1_LARGE_GROUPS'
    if [[ "$variant" == teams2 ]]; then flags+=';-DGLM_W2_GROUPED_EXPERT_LANES=2'; fi
    if [[ "$variant" == teams4 ]]; then flags+=';-DGLM_W2_GROUPED_EXPERT_LANES=4'; fi
    ;;
  *) echo "unknown variant: $variant" >&2; exit 1 ;;
esac
build_root=/srv/ai/src/glm-l1-wide-build-20261004
build_dir="$build_root/build-pipelined-20261004"
package_name="opp-l1-$variant-v2-20261004"
record="/home/matteius/experiments/glm-w3-20261004/wide-l1-20261004/$package_name"
if [[ -e "$build_root/$package_name" || -e "$record" ]]; then
  echo "refusing to overwrite $package_name" >&2
  exit 1
fi
mkdir -p "$record"
cp "$build_root/gmm/w2_blocked_dequant_matmul_v310/op_kernel/w2_blocked_dequant_matmul_v310.h" "$record/tested-kernel.h"
cp "$build_root/gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel/w2_grouped_blocked_dequant_matmul_v310.cpp" "$record/tested-grouped.cpp"
cp "$build_root/gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel/expert_core_groups.h" "$record/expert_core_groups.h"
cp "$build_root/gmm/w2_blocked_dequant_matmul_v310/op_kernel/row_reuse_geometry.h" "$record/row_reuse_geometry.h"
printf '%s\n' "$flags" > "$record/compile-flags.txt"
cmake -S "$build_root" -B "$build_dir" "-DOPS_COMPILE_OPTIONS=$flags" > "$record/configure.log" 2>&1
cmake --build "$build_dir" --target ops_transformer_kernel -j4 > "$record/build.log" 2>&1

python3 - "$build_root" "$build_dir" "$package_name" "$record" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys

root, build, name, record = sys.argv[1:]
root, build, record = Path(root), Path(build), Path(record)
for source, snapshot in (
    (root / 'gmm/w2_blocked_dequant_matmul_v310/op_kernel/w2_blocked_dequant_matmul_v310.h', record / 'tested-kernel.h'),
    (root / 'gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel/w2_grouped_blocked_dequant_matmul_v310.cpp', record / 'tested-grouped.cpp'),
    (root / 'gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel/expert_core_groups.h', record / 'expert_core_groups.h'),
    (root / 'gmm/w2_blocked_dequant_matmul_v310/op_kernel/row_reuse_geometry.h', record / 'row_reuse_geometry.h'),
):
    if source.read_bytes() != snapshot.read_bytes():
        raise RuntimeError(f'Source changed during build: {source}')
package = root / name
shutil.copytree(root / 'opp-l1-overlap-20261004', package)
source = build / 'binary/ascend310p/bin/w2_grouped_blocked_dequant_matmul_v310'
destination = package / 'vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend310p/w2_grouped_blocked_dequant_matmul_v310'
objects = sorted(source.glob('*.o'))
if len(objects) != 2:
    raise RuntimeError(f'Expected canonical and NZ objects, got {objects}')
hashes = {}
for obj in objects:
    for path in (obj, obj.with_suffix('.json')):
        shutil.copy2(path, destination / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
for path in (record / 'tested-kernel.h', record / 'tested-grouped.cpp', record / 'expert_core_groups.h', record / 'row_reuse_geometry.h', record / 'compile-flags.txt'):
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
(record / 'kernel-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(json.dumps({'package': str(package), 'hashes': hashes}))
PY
