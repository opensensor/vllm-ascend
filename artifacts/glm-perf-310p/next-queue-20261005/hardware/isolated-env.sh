#!/usr/bin/env bash
set -eo pipefail
source /srv/ai/bin/ascend-env.sh
set -u
package=$1
shift
source_root=/srv/ai/src/glm-selective-w3-nz-test-20261004
root=/home/matteius/experiments/glm-next-queue-20261005
vendor="$package/vendors/custom_transformer"
overlap=/srv/ai/src/glm-l1-wide-build-20261004/opp-l1-overlap-20261004/vendors/custom_transformer
split=/srv/ai/src/kda-persistent-scores-opp/vendors/custom_transformer
base=/srv/ai/src/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$vendor:$overlap:$split:$base"
export LD_LIBRARY_PATH="$vendor/op_api/lib:$overlap/op_api/lib:$split/op_api/lib:$base/op_api/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$source_root:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python "$root/isolated.py" "$@"
