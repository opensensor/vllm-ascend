#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Hardware run: use only with exclusive NPU access.
set -eo pipefail
source /srv/ai/bin/ascend-env.sh
set -u
source_root=/srv/ai/src/glm-selective-w3-nz-test-20261004
build_root=/srv/ai/src/glm-l1-wide-build-20261004
study_root=/home/matteius/experiments/glm-w3-20261004/wide-l1-20261004
python_bin=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
export PYTHONPATH="$source_root:${PYTHONPATH:-}"
cd "$source_root"
for variant in overlap all-resident teams-control teams2 teams4; do
  if [[ "$variant" == overlap ]]; then package=opp-l1-overlap-20261004;
  else package="opp-l1-$variant-v2-20261004"; fi
  result="$study_root/$package"
  vendor="$build_root/$package/vendors/custom_transformer"
  export ASCEND_CUSTOM_OPP_PATH="$vendor"
  export LD_LIBRARY_PATH="$vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
  if [[ "$variant" != overlap ]]; then
    "$python_bin" -m pytest --noconftest -xq \
      tests/e2e/nightly/310p/single_node/ops/test_w2_grouped_nz_bytes_310.py \
      tests/e2e/nightly/310p/single_node/ops/test_w2_grouped_decode_table_reuse_310.py \
      > "$result/grouping-tests.log" 2>&1
    tail -n 1 "$result/grouping-tests.log"
  fi
  "$python_bin" -m tools.glm_perf.operator_bench measure \
    --opp-root "$vendor" \
    --opp-binary "$vendor/op_impl/ai_core/tbe/kernel/ascend310p/w2_grouped_blocked_dequant_matmul_v310/W2GroupedBlockedDequantMatmulV310_21ed540d36cafc1d1bee8d7cdd208e81.o" \
    --source "$result/tested-kernel.h" --layout nzpacked --label "$variant" \
    --rows 5120 --route-patterns uniform_all_experts_quarter --warmup 2 --repeats 7 \
    --output "$result/grouping-operator.json" > "$result/grouping-operator.log" 2>&1
  "$python_bin" -m tools.glm_perf.benchmark_w3_nz_310 --repeats 5 \
    --output "$result/grouping-w3.json" > "$result/grouping-w3.log" 2>&1
  if [[ "$variant" != overlap ]]; then
    "$python_bin" -m tools.glm_perf.operator_bench compare \
      --baseline "$study_root/opp-l1-overlap-20261004/grouping-operator.json" \
      --candidate "$result/grouping-operator.json" --output "$result/grouping-compare.json"
  fi
  printf 'Finished %s\n' "$variant"
  cat "$result/grouping-w3.log"
done
