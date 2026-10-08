#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run after the other NPU users release one 310P device.
set -eo pipefail

source_root=${1:?pass the staged vllm-ascend source root}
visible_device=${2:-0}
opp_root=${3:-$source_root/opp-w3-native}
source /srv/ai/bin/ascend-env.sh
set -u

w3_vendor="$opp_root/vendors/custom_transformer"
if [[ ! -d "$w3_vendor/op_api/lib" ]]; then
  echo "native W3 OPP package is missing: $w3_vendor" >&2
  exit 1
fi
export ASCEND_CUSTOM_OPP_PATH="$w3_vendor:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="$w3_vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$source_root:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$visible_device"
export SOC_VERSION=ascend310p1

cd "$source_root"
test_file=tests/e2e/nightly/310p/single_node/ops/test_w2_grouped_blocked_dequant_matmul_310.py
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m pytest -sv \
  --confcutdir=tests/e2e/nightly/310p/single_node/ops \
  "$test_file::test_grouped_packed_projection_matches_reference[3]" \
  "$test_file::test_standalone_projection_keeps_default_table_preparation[3]" \
  "$test_file::test_grouped_w3_glm_projection_matches_reference" \
  "$test_file::test_grouped_w3_nz_matches_canonical"
