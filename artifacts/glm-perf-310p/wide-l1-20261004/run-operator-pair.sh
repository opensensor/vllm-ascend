#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
source /srv/ai/bin/ascend-env.sh
set -u

source_root=/srv/ai/src/glm-selective-w3-nz-test-20261004
baseline_root=/srv/ai/src/build-only-glm-w3-nz-csrc-20261004
candidate_root=/srv/ai/src/glm-l1-wide-build-20261004
candidate_opp_name=${1:-opp-l1-wide}
result_root=/home/matteius/experiments/glm-w3-20261004/wide-l1-20261004/$candidate_opp_name
python_bin=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
kernel_relative=op_impl/ai_core/tbe/kernel/ascend310p/w2_grouped_blocked_dequant_matmul_v310/W2GroupedBlockedDequantMatmulV310_21ed540d36cafc1d1bee8d7cdd208e81.o
mkdir -p "$result_root"
export PYTHONPATH="$source_root:${PYTHONPATH:-}"

for label in baseline candidate; do
  if [[ "$label" == baseline ]]; then
    build_root=$baseline_root
    opp_name=opp-w3-nz-candidate
  else
    build_root=$candidate_root
    opp_name=$candidate_opp_name
  fi
  vendor_root="$build_root/$opp_name/vendors/custom_transformer"
  export ASCEND_CUSTOM_OPP_PATH="$vendor_root"
  export LD_LIBRARY_PATH="$vendor_root/op_api/lib:${LD_LIBRARY_PATH:-}"
  "$python_bin" -m tools.glm_perf.operator_bench measure \
    --opp-root "$vendor_root" --opp-binary "$vendor_root/$kernel_relative" \
    --source "$build_root/gmm/w2_blocked_dequant_matmul_v310/op_kernel/w2_blocked_dequant_matmul_v310.h" \
    --source "$build_root/gmm/w2_grouped_blocked_dequant_matmul_v310/op_kernel/w2_grouped_blocked_dequant_matmul_v310.cpp" \
    --layout nzpacked --label "$label" \
    --rows 5120 --route-patterns uniform_all_experts_quarter \
    --warmup 2 --repeats 5 --output "$result_root/$label.json"
done

"$python_bin" -m tools.glm_perf.operator_bench compare \
  --baseline "$result_root/baseline.json" \
  --candidate "$result_root/candidate.json" \
  --output "$result_root/compare.json"
