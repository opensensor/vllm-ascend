#!/usr/bin/env bash
set -eo pipefail
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
export PYTHONPATH=/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016/runtime:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}
runtime_root=/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016/runtime
tools_root=$runtime_root
coherent_opp=/srv/ai/src/qwen38-prefill-batch256-coherent-opp-20261005/vendors/qwen38_batch256_coherent_transformer
packaged_opp=${runtime_root}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
retained_opp=/srv/ai/src/native-int4-w4a8.KiuhBN/opp-retained-good-20260928
stride_opp=${runtime_root}/gdn-unified-v5-opp/vendors/qwen_gdn_unified_v5_transformer
for required in "$coherent_opp" "$packaged_opp" "$retained_opp" "$stride_opp"; do
  [[ -d "$required" ]] || { echo "Required operator package missing: $required" >&2; exit 1; }
done
export PYTHONPATH="${tools_root}:${runtime_root}:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
stride_opp=${runtime_root}/gdn-unified-v5-opp/vendors/qwen_gdn_unified_v5_transformer
export ASCEND_CUSTOM_OPP_PATH="${stride_opp}:${coherent_opp}:${packaged_opp}:${retained_opp}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${stride_opp}/op_api/lib:${coherent_opp}/op_api/lib:${packaged_opp}/op_api/lib:${retained_opp}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5
cd /srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016/runtime
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python "$@"
