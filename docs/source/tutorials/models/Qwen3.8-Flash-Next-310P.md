# Qwen3.8 Flash Next W8A8：310P TP4 部署

## Introduction

模型：[matteiuspi/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i)。
本文发布 2026-09-25 实际服务使用的性能代码，而非官方上游通用配置。
模型包含 48 层、512 个路由专家（每 token 选 10 个）、GDN/QSA 混合注意力、PLE 和 MTP。

## Supported Features

| 功能 | 当前范围 |
| --- | --- |
| 硬件 | 2 × Atlas 300I Duo，4 × Ascend 310P3；TP4 |
| 生成 | MTP k=1、breakable FULL_DECODE_ONLY 图；禁用 async scheduling |
| 缓存 | prefix caching、chunked prefill、Mamba align、QSA KV 头分片 |
| 上下文 | 默认 160000、max-num-seqs=1；不是双 160K 并发保证 |
| 工具调用 | qwen3_xml；推理解析器 qwen3 |
| EP / flashcomm1 | 此配置未启用；专家按 TP rank 切片，不等同于验证 EP 模式 |
| 多模态 | 此次仅验证文本；image/video 禁用 |

## Environment Preparation

需要已安装驱动、CANN、ATB 的 310P 主机；不要套用 A2/A3 容器配置。
已运行环境：Python 3.12、CANN 9.1.0、npu-smi 26.0.rc1、
torch 2.13.0+cpu、torch_npu 2.13.0rc1、transformers 5.17.0。
这些是实际版本记录，不代表任意 PyPI 包组合兼容；不要单独升级 transformers。

vLLM core 必须使用
[`opensensor/vllm@3ab5dda29acabea01f6a63d0806bdbbb4a27bde5`](https://github.com/opensensor/vllm/commit/3ab5dda29acabea01f6a63d0806bdbbb4a27bde5)，
其中包含 breakable graph 等依赖。插件使用包含本文的 `opensensor/vllm-ascend` main 提交；
用 `git rev-parse HEAD` 记录部署版本。不要用未固定版本的 upstream vLLM 替换。

在已配置的 Ascend Python 环境中，安装上述 core，然后从插件仓库根目录重新构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export SOC_VERSION=ascend310p1
git submodule update --init --recursive
COMPILE_CUSTOM_KERNELS=1 python -m pip install --no-build-isolation --no-deps -e .
python -c 'import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)'
```

本次包含原生 QSA score/sparse-attention/value-gather-NZ 内核变更；
仅复制 Python 文件或复用旧 `.so` 不足以部署。构建后重启自己的服务才能加载新内核。
不要让旧 `/tmp/qsa-runtime-vllm-ascend` 或实验目录通过 `PYTHONPATH` 覆盖安装。

先把完整 checkpoint 下载到本地目录（约 240 GB，PLE 表从主机端 mmap 读取）：

```bash
hf download matteiuspi/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i \
  --local-dir /srv/ai/models/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i
```

## Deployment

```bash
bash examples/start_qwen38_flash_next_310p.sh \
  --model /srv/ai/models/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i \
  --host 127.0.0.1 --port 8001
```

脚本前台运行，不杀进程、不自动重启、不下载依赖；`--dry-run` 打印命令。
远程客户端访问时可设置 `--host 0.0.0.0`，需自行配置网络访问控制。
`--max-model-len 131072` 可降低容量；内存利用率固定为 **0.965**，不要提高。
启动后必须确认服务报告足够的 KV token 容量；容量不足时降低 context，不要提高内存上限。
脚本固定 2048-token prefill chunks、TP4、MTP k=1、单序列和图执行，避免误用未经验证的组合。

排障时可从 `--dry-run` 命令复制出独立诊断配置：移除 `--compilation-config`，
加入 `--enforce-eager` 并降低 `--max-model-len`。这不是 15 tok/s 配置，也不能证明图路径正确。
`TORCHDYNAMO_DISABLE=1` 不能替代图回放验证。

## Functional Verification

启动完成并非成功条件。必须用真实权重完成请求：

```bash
curl --fail http://127.0.0.1:8001/v1/models
curl --fail http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-w8a8","messages":[{"role":"user","content":"Reply with exactly READY. Do not explain."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

运行快照曾完成 HTTP 200 / `READY`，并在实际 Kilo 请求中持续生成。
图捕获日志、MTP acceptance 和完成请求日志应同时保留；dummy 只验证结构，不能替代真实权重质量测试。
脚本启用 `VLLM_ASCEND_LOG_REQUEST_TIMINGS=1`（默认关闭的非敏感诊断选项），
输出 prefill、缓存 token 数、decode、queue、total 时间，不记录提示内容。
`--enable-prompt-tokens-details` 使客户端能够读取 cached_tokens；客户端未展示缓存不等于服务端没有命中。

## Accuracy Evaluation

之前 5 个 GPQA Diamond 样本为 5/5；这是管线 smoke，不是完整准确率结论。
完整 198 题、160K 全长质量、并发长请求及 bitwise determinism 仍未完成验证。
因此未提供包含虚构完整评测指标的 accuracy gate YAML。
模型的理论最大上下文请看 checkpoint 配置；160000 是本硬件的服务限制，不是模型上限。

## Performance

9 月 25 日单序列实测约 14.8–16.0 tok/s（短请求最高约 17 为用户观察），不保证所有提示达到 15。
发布审计时的现役服务记录：2026-09-26 02:34:37 UTC，
1156 个输出 token、1155 个间隔、decode 79452.6 ms，即 14.5 tok/s；
3789 个新 prompt token 加 37376 个缓存 token，prefill 9896.6 ms。
原始运维日志路径：`/home/matteius/logs/serve_qwen38_mtp_graph_8001.log`（部署主机，不随代码分发）。
同次启动捕获图约 0.26 GiB，KV pool 170675 tokens，配置单窗口 160000。
这些是运行快照证据；本次发布期间未重启现役服务，也未宣称从全新安装复现了所有 NPU 测试。

关键修复：预打包专家/NZ 权重减少逐步 Cast；分组路由减少小算子；
图回放刷新 QSA/PLE 状态并保持 MTP hidden buffer 地址；Mamba postprocess 使用与 forward 一致的紧凑 slot，
避免 global block ID 索引 64-slot 状态池；prefix checkpoint 可溢出到主机并恢复。
MTP 专家加载时转 W8A16；仅在权重逐元素相等时共享 target/draft LM head。
QSA TP4 每 rank 保留一个 KV 头，同时保留正确的 query/GQA 映射。
不包含实验中的专家 LRU offload 或双窗口调度。

CPU 回归入口（无需 NPU）：

```bash
python -m pytest -q --confcutdir=tests/ut/qwen38_1m tests/ut/qwen38_1m
python -m pytest -q --noconftest \
  tests/ut/_310p/test_prefix_mamba_state.py \
  tests/ut/_310p/test_mamba_align_fallback_310p_source.py
```

NPU 数值回归/benchmark 在 `tests/e2e/nightly/310p/single_node/ops/`，
包括 QSA score、sparse attention、value gather NZ、分组 MoE 和线性 NZ 路径。
运行前需停止占用对应 NPU 的服务；不能把 CPU 回归当作这些内核测试的替代。
