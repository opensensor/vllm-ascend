#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
# shellcheck disable=SC1091
source /srv/ai/bin/ascend-env.sh
set -u
study_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
label=$1
package=/home/matteius/experiments/glm-prompt-profile-20261005/opp-score-cache
if [[ "$label" == cached ]]; then package="$study_root/opp-score-batch"; fi
vendor="$package/vendors/custom_transformer"
overlap=/srv/ai/src/glm-l1-wide-build-20261004/opp-l1-overlap-20261004/vendors/custom_transformer
split=/srv/ai/src/kda-persistent-scores-opp/vendors/custom_transformer
base=/srv/ai/src/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$vendor:$overlap:$split:$base"
export LD_LIBRARY_PATH="$vendor/op_api/lib:$overlap/op_api/lib:$split/op_api/lib:$base/op_api/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/srv/ai/src/glm-selective-w3-nz-test-20261004:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python "$study_root/validate-full-kda.py" "$label"
