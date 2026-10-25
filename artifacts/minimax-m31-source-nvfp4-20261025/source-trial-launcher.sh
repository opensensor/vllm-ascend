#!/usr/bin/env bash
source /workspace/minimax-m31-env.sh
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/minimax-m31-merged-opp/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
cd /workspace
export VLLM_SERVER_DEV_MODE=1
export VLLM_PP_LAYER_PARTITION=21,20,19
exec vllm serve /srv/ai/storage/models/MiniMax-M3.1-Flash-W4 \
  --served-model-name minimax-m31-flash --load-format minimax_m3_w4 --dtype half \
  --model-loader-extra-config '{"expert_layout":"nz256","max_grouped_routes":2048,"source_nvfp4":"/srv/ai/minimax-source-trial/MiniMax-M3.1-Flash-NVFP4-experts","source_nvfp4_kernel":"/workspace/m31-nvfp4-v3/manifest.json"}' \
  --tensor-parallel-size 2 --pipeline-parallel-size 3 --enable-expert-parallel \
  --enforce-eager --block-size 128 --max-model-len 32768 --max-num-seqs 1 \
  --additional-config '{"hccl_buffer_size_mb":32}' --max-num-batched-tokens 512 --gpu-memory-utilization 0.95 --kv-cache-memory-bytes 943718400 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --mm-processor-kwargs '{"max_long_side_pixel":448}' --port 8000 \
  --worker-extension-cls tools.minimax_m3.resident_worker.MiniMaxResidentExtension \
  --middleware tools.glm_perf.resident_middleware.ResidentControlMiddleware \
  --enable-auto-tool-choice --tool-call-parser minimax_m3 \
  --no-async-scheduling --disable-custom-all-reduce --reasoning-parser-plugin /vllm-workspace/vllm-ascend/tools/minimax_m3/reasoning_parser.py --reasoning-parser minimax_m31
