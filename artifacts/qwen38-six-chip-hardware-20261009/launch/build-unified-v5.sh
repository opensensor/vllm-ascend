#!/usr/bin/env bash
set -eo pipefail
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
runtime=/srv/ai/src/qwen-six-chip-test-f25e70c09-20261009
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export CPATH=/srv/ai/src/qwen38-w4-ce1862e52/csrc/third_party/catlass/include:${CPATH:-}
cd "$runtime/csrc"
bash build.sh --pkg --ops=chunk_fwd_o_vllm,chunk_gated_delta_rule_fwd_h,causal_conv1d_v310,recurrent_gated_delta_rule_v310 --soc=ascend310p --vendor_name=qwen_gdn_unified_v5 --cann_3rd_lib_path=/srv/ai/src/qwen38-coherent-opp-src-20261001-r2/csrc/third_party -j4
bash build/cann-ops-transformer-qwen_gdn_unified_v5_linux-x86_64.run --install-path="$runtime/gdn-unified-v5-opp"
