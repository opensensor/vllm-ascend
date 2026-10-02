# Qwen3.8 Flash Next: offline W4 candidate

## Introduction

This is an **experimental alternate checkpoint**, not a replacement for the
validated W8A8 + MTP + decode-graph deployment. It quantizes only the 48 target
layers' routed experts directly from the original BF16 checkpoint. The router,
shared experts, attention, vision, embeddings, PLE and MTP checkpoint tensors
remain floating-point (BF16 is exported as FP16 for 310P; FP32 stays FP32).

The converter uses **ModelSlim IR asymmetric per-group min/max RTN**, group size
128. It does not perform GPTQ, AWQ, activation calibration, or accuracy evaluation.
The custom checkpoint format is not interchangeable with stock AWQ/GPTQ or the
generic Ascend W4A16 fused-MoE format.

The full offline export completed on 2026-09-26: 1,610 shards, 222,746 tensors,
and 169.1628 GiB of tensor payload including the host PLE table and floating
MTP weights. A later authorized TP4/310P hardware smoke loaded the full checkpoint
and completed three correct arithmetic/counting/Python-output answers. This does not establish production
speed or broad model quality; see the hardware results below.

## Supported Features

| Feature | W4 candidate status |
| --- | --- |
| ModelSlim conversion and packed checkpoint | Offline path implemented |
| Strict loader, signed packing, group zero points | CPU tests and real-weight 310P smoke |
| Contiguous expert-TP ownership, shared-expert TP | CPU numerical tests, including uneven expert ownership |
| PLE disk-backed lazy lookup | Preserved; W4 explicitly selects the standard HF index |
| W8 runtime, MTP and graph defaults | Unchanged unless W4 checkpoint metadata is present |
| W4 NPU inference | Full checkpoint loaded on TP4/310P; three completed correct answers and seven operator regressions passed |
| Experimental W4 Cube projection | Separate group/routed 310P operators; 143 NPU regressions plus 17 RoPE/replay gates and real-weight TP4 smokes passed |
| W4 ACLGraph | FULL_DECODE_ONLY verified with cube_310_routed; older backends still require eager |
| W4 MTP | k=1/k=2/k=4 real-weight smokes and FULL replay verified; sustained speed depends on workload |
| W4 multimodal / flashcomm1 / EPLB | Not validated; language-model-only with existing collectives |
| Long context / concurrent sessions | ~23.4k batch-one k=1/k=2/k=4 benchmarks complete, including latest compact-expert k=2; maximum capacity and concurrency unvalidated |

Uneven expert ownership is not proof of whole-model TP3/TP6 support: attention,
shared-expert and MTP divisibility constraints still apply.

## Environment Preparation

Use the existing pinned `opensensor/vllm` fork and an isolated worktree containing
this plugin change. Do not upgrade the serving environment, install this worktree
over it, or change the W8 launcher to try this candidate.

Conversion needs Python, CPU PyTorch, safetensors, and the local ModelSlim checkout.
It never imports the model, initializes an accelerator, or contacts the server.
Allow approximately 170 GiB of destination disk space plus temporary working space.
The source must contain its original `config.json`, index, and all shards.
No A2/A3 Docker image is prescribed here: this path targets the existing 310P
environment, and no new container or NPU environment was validated offline.

```bash
MODELSLIM_PYTHON=/path/to/msmodelslim/.venv-cpu/bin/python
"$MODELSLIM_PYTHON" tools/quantization/qwen38_modelslim_w4.py \
  --source /path/to/original/Qwen3.8-Flash-Next \
  --output /path/to/separate/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --group-size 128 --threads 4
```

After an interrupted build, rerun the same command with `--resume`. Resume checks
source/config/settings/tool revision identity and hashes of completed output
shards. It rejects a nonempty unrelated destination and source/output overlap.
The final model index is published only after all tensors are exported. Preserve
`build-journal.json`, `build-receipts/`, and `quantization_provenance.json`.

### Storage contract

Selection is explicit in `config.json` → `text_config.ascend_expert_quantization`:

```json
{
  "format": "qwen4exp_w4a16_group_v1",
  "bits": 4,
  "group_size": 128,
  "symmetric": false,
  "packing": "signed_int4_low_nibble_first_in_axis",
  "backend": "eager_dequant",
  "scale_dtype": "float16",
  "offset_dtype": "int8"
}
```

Each `model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj`
contains `weight` (INT8 bytes, two signed INT4 values per byte, low nibble first),
`weight_scale` (FP16), and `weight_offset` (INT8 signed zero point).
Packing is along the input dimension. Dequantization is `(q - offset) * scale`,
with one scale/zero-point pair per output row and input group of 128.

The resident routed-expert bank is `0.5 + 3/128` bytes per weight: **58.8867 GiB**
for this model, excluding all other tensors, caches and runtime workspace.
That number is not a whole-model fit or context-capacity guarantee. PLE remains
on disk/host and is not counted as NPU expert memory.

## Deployment

**Do not use the W8 production launcher for this checkpoint.** Its generic
quantization flags are incompatible. The following is the experimental
hardware-smoke configuration, not a production serving profile. It does not
stop another process and uses port 8002 instead of the production port.

After separately activating the pinned Ascend environment and sourcing its CANN
setup, from the isolated plugin worktree (or its installed test environment):

```bash
export SOC_VERSION=ascend310p1
export VLLM_ASCEND_ENABLE_310P=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export TASK_QUEUE_ENABLE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ASCEND_KV_CACHE_FRACTION=0.65
python -m vllm.entrypoints.cli.main serve /path/to/separate/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --served-model-name qwen38-w4-experimental \
  --host 127.0.0.1 --port 8002 \
  --dtype float16 --tensor-parallel-size 4 \
  --no-async-scheduling --disable-custom-all-reduce \
  --max-model-len 32768 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --language-model-only \
  --enable-expert-parallel --enable-ep-weight-filter \
  --reasoning-parser qwen3 --mamba-cache-mode align --enable-chunked-prefill \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --cudagraph-metrics --enable-logging-iteration-details \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}' \
  --hf-overrides '{"text_config":{"ascend_expert_quantization":{"backend":"cube_310_routed","bits":4,"format":"qwen4exp_w4a16_group_v1","group_size":128,"offset_dtype":"int8","packing":"signed_int4_low_nibble_first_in_axis","scale_dtype":"float16","symmetric":false}}}' \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

Omit `--quantization ascend`:
the model-specific metadata selects W4, and generic quantization is rejected.
Never raise memory utilization above the established 0.965 cap. The model config
advertises 262,144 positions; W4's usable maximum has **not** been measured.

较大的 MTP draft count 需要同时修改两个参数：k=2 使用
`num_speculative_tokens=2` / `cudagraph_capture_sizes=[1,3]`，k=4 使用
`num_speculative_tokens=4` / `cudagraph_capture_sizes=[1,5]`。本轮两者均已
通过真实 smoke 和相应 FULL replay；更大的 k 不保证更快，需看实际接受率。
保留上述 k=1 已验证长请求基线，未自动替换生产 W8 的参数。

The command requires an isolated installation rebuilt with both
`QwenW4GroupMatmulV310` and `QwenW4RoutedMatmulV310` plus matching Torch bindings.
It leaves the checkpoint's default and the W8 runtime unchanged. Detailed
commands and build provenance are in `artifacts/qwen38-w4-offline/CUBE_KERNEL.md`.
The `cube_310_tiled` variant losslessly re-encodes nibbles in Cube NZ order at
load time and biases codes/offsets equally. It preserves parameter shapes,
dtypes, byte counts, and the quantization formula; the checkpoint is unchanged.
The matching operator is mandatory because the in-memory byte layout differs.
Neither group-only variant makes full-model graph capture safe while routing
uses the CPU. The new `cube_310_routed` variant reads expert IDs on device and
supports bounded decode graphs up to 80 routes (8 tokens for this top-k=10 model).
Larger prefill uses the existing grouped host route; oversized capture fails
explicitly. Workspace is bounded by `routes*N*K*2 + CANN reserve`, plus
`routes*routes*N*2` for at most 80 routes to reuse a repeated expert's unpacking
within a verification batch (31.25 MiB additional scratch at R=80/N=2560).
It is not a persistent expanded expert bank.
For eager isolation, select `cube_310_tiled`
or remove the overrides, add `--enforce-eager`, and remove speculative and
compilation configuration. Do not use `TORCHDYNAMO_DISABLE=1` as graph evidence.

## Functional Verification

Offline artifact audit, after conversion completes:

```bash
"$MODELSLIM_PYTHON" tools/quantization/audit_qwen38_w4.py \
  /path/to/separate/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --report /tmp/qwen38-w4-audit.json
```

This checks all tensor names, shapes, dtypes, byte accounting, untouched-tensor
inventory, and sampled expert reconstruction. It is not full-model inference.

For an explicitly authorized NPU test:

```bash
curl --fail http://127.0.0.1:8002/v1/models
curl --fail http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-w4-experimental","messages":[{"role":"user","content":"Return the integers 1 through 50 in order."}],"temperature":0,"seed":1024,"max_tokens":512,"chat_template_kwargs":{"enable_thinking":false}}'
```

Require a completed correct response and clean worker logs; startup alone is not
a pass. Next compare W8 and W4 on identical held-out coding prompts, prompts that
cross cache-block boundaries, and perplexity/answer accuracy. Longer context and
concurrency require separate gates. Check actual FULL runtime-mode statistics
for two-token verification batches and increasing drafted/accepted counters;
configuration or startup alone is not graph/MTP evidence. The full 198-item
GPQA result below is a quality observation; no accuracy-threshold CI YAML has
been set from one benchmark run.

## Accuracy Evaluation

The 2026-10-02 end-to-end GPQA Diamond comparison completed all 198 matched
questions on the native W4A8 TP4/EP4 service and an RTX 6000 Pro IQ4_XS
reference. Both scored **140/198 (70.71%)** with the AISBench-style last
`Answer: X` extractor. Ascend produced 114 correct final responses versus 120
on RTX; 81 versus 74 requests hit the 8,192-token limit. The extractor also
credited 26 Ascend and 20 RTX correct letters that appeared only in unfinished
reasoning. Across the 112 questions where both emitted a final answer, every
letter agreed. The [complete protocol, domain breakdown, performance, and
scoring audit](../../../../artifacts/qwen38-w4-offline/GPQA_DIAMOND_20261002.md)
records the comparison and its limitations. This result applies to W4, not to
the separate W8A8 checkpoint.

Other full-model evaluation remains pending. The completed artifact passed a
full header/index audit. Across 45 sampled real expert projections, mean relative
weight RMSE was 0.103625 and mean weight cosine was 0.994629; all sampled
reconstructions were finite. Weight cosine and Gaussian-input errors are diagnostics, not
task-quality evidence. Earlier W8 or other-platform four-bit results cannot be
transferred to this RTN artifact.

## Performance

The original three TP4 hardware requests decoded at approximately **0.23 tok/s**, not
15 tok/s. The reference backend keeps only packed weights resident and
dequantizes selected experts one at a time, with host routing synchronization.
Each rank reported **18.6917 GiB** model-load memory with MTP and graphs disabled.
That is not a measured maximum context or concurrency capacity.

The first accelerated `cube_310` implementation completed the same three
correct smokes at approximately **3 tok/s**, with the same 18.6917 GiB/rank
model memory. This is not production parity: the refreshed W8 + MTP + graph
baseline is 19.073 tok/s at short context and 18.091 tok/s near 23.4k context.
Whole-model graphs/MTP were not enabled in that first run. See
`artifacts/qwen38-w4-offline/CUBE_KERNEL.md` for measurements
and the distinction between raw smoke timing and corrected decode throughput.

The subsequent `cube_310_tiled` backend passes the same three correct smokes
at **4.24–4.28 tok/s**, retaining 18.6917 GiB/rank. Its 51 operator tests pass,
including changed-weight replay and K-batch boundaries. Load-time re-encoding
adds startup work (about 233 seconds for model loading in this run). These
smokes remain eager/no-MTP and do not establish production parity.

Detailed provenance, timings and limitations are in
`artifacts/qwen38-w4-offline/HARDWARE_SMOKE.md`. Hardware regression coverage is
`tests/e2e/nightly/310p/single_node/ops/test_qwen4exp_w4_310.py`.

### 最新 MTP + graph 验证

`cube_310_routed` 已完成 TP4 真实整模型 MTP k=1 + FULL_DECODE_ONLY
验证。三个 smoke 均完整正确，1–50 为 **11.592 tok/s**，Python 输出题为
11.653 tok/s；这些短 smoke 不能代替 sustained coding throughput。
model-load 为 19.3821 GiB/rank，graph capture 报告 0.30 GiB。
82 项 NPU operator/layer 回归与 33 项 CPU/build tests 通过。

首次完整 graph 启动因 NZ shared weights 的 FP32 Cast 失败；修正仅限
routed NPU 后端，改为 resident FP16 operand policy，与 W8 一致。
增加真实 NZ post-load 和改变输入/route 的 replay 回归后，实际服务的
两-token verification batch 显示 `Runtime Mode = FULL`，MTP 接受计数
也递增；不是只验证 capture。最新速度证据与 runbook 在上述 CUBE_KERNEL。

同协议三条 512-token prompts 的 sustained 中位数为：短上下文 **10.722**、
约 23.4k 上下文 **10.660 tok/s**。六题都正常完成，最高温度 80°C。
长前缀冷 TTFT 为 368.799 秒，prefill 仍有明显性能缺陷。下一步是完整
replay 的瓶颈 profile；仍未达到 W8 的 19.073/18.091 tok/s。
不能靠常驻 INT8/FP16 全专家展开
制造 W4 提速；必须保留量化内存收益与动态 replay 正确性。

整模型八步 replay profile 已完成：W4 projection 占各 rank 累计任务时间的
46.5–49.8%（不是 critical-path 占比）。同 batch 的重复专家复用解包后，
真实 smoke 通过，短/23.4k 中位数提高到 **11.203 / 11.050 tok/s**。
更宽 unpack 候选已通过 101 项 NPU 回归、真实权重 layer replay 和三个
整模型 smoke；短/23.4k coding 中位数为 **11.310 / 11.239 tok/s**。
后续 persistent N-tile 调度通过 107 项 NPU 回归与整模型 smoke；
MTP k=1 + FULL graph 短/23.4k coding 中位数为 **11.830 / 11.052 tok/s**。
长请求低于 wide 的 11.239，且 acceptance 不同，不声称普遍提速。
其两-token 单层 replay 更快，五-token 单层没有改善；不推广到其它 MTP k。
下一 80-route 候选通过 122 项 NPU tests；修正输入对齐后，五-token
partial-layer replay 为 3.524 ms，不使用输入不同的 2.136 ms 作 A/B。
MTP k=2 的三个真实 smoke 与三-token FULL replay 通过；短 coding 中位数
为 **11.972 tok/s**，并非 counting 的 14.600 tok/s。
MTP k=4 的真实 smoke、五-token FULL replay 也通过，但短 coding 中位数
回落至 **10.155 tok/s**（三题 12.377/9.877/10.155）；不能用 counting 的
16.179 tok/s 宣称 coding 提速。k=4 的约 23.4k 热前缀三题也完成，
中位数 7.914 tok/s，低于 k=1 的 11.052；不自动升级生产配置。
最新 Qwen-only L1 tile 候选通过 **140 项 NPU tests** 和三个真实 smoke，
MTP k=2 + 三-token FULL replay 确认有效。短 coding 中位数提高到
**12.470 tok/s**（三题 14.207/11.895/12.470），比旧 k=2 高 4.16%；
约 23.4k 长测试中位数为 **9.283 tok/s**（10.483/9.283/9.147），
低于旧 k=1 的 11.052，不能推广短 prompt 的优势。没有常驻 expanded expert bank，W8/GLM helper
未修改，model-load 仍为 19.3821 GiB/rank；没有据此宣称新的 context 容量。
各候选、原始样本、kernel SHA 和边界说明见
`artifacts/qwen38-w4-offline/REPLAY_PROFILE.md`。

批量 peer-output 清零候选随后通过 **143 项 NPU tests**，包括用 NaN
污染输出后的 changing-route replay；37 项 CPU/build tests 通过。
三个整模型 smoke 正确、MTP k=2 + 三-token FULL replay 有效。
短 coding 三题为 **14.288/11.902/12.552 tok/s**，中位数 **12.552**；
相对 12.470 仅高 0.66%，输出和 acceptance 也变化，不能视为显著提速。
新长测试完成，约 23.4k 三题为 **11.023/9.709/8.866 tok/s**，
中位数 **9.709**；全部生成 512 tokens，复用 23,168 prefix tokens。
输出和 acceptance 不同，第三题回退，不声称普遍长上下文提速。
未改变现有 W8 环境，也未实现 W8 的实测速度目标。

下一隔离 W4 候选在 Q/K/index-query 之间共享当前 positions 的 RoPE 表，
保留 compute 精度和独立 index-key 坐标；MRoPE axis map 提前创建为
非持久 buffer，避免 graph capture 中同步 H2D。113 项 CPU tests 与
160 项 NPU tests 通过；三-token query RoPE 独立 graph 从 0.477 降至
0.194 ms、bitwise equal。三个完整真实 smoke 与三-token FULL replay
已通过；MTP k=2 短 coding 三题为 **14.412/12.091/13.034 tok/s**，
中位数 **13.034**（比 12.552 高 3.84%）。第一题输出相同，其余两题
输出和 acceptance 改变，不能把全幅提高归因于 RoPE，也未达到 W8。
约 23.4k 长上下文三题也完成：**10.852/10.120/9.004 tok/s**，中位数
**10.120**；均生成 512 tokens，复用 23,168 prefix tokens。冷 TTFT
358.790 秒，输出与 acceptance 不同、第一题回退，不能声称普遍提速。
同一 kernel + RoPE 的 k=1 对照已完成：短/23.4k 中位数为
**12.787 / 12.754 tok/s**，两-token FULL replay 和 MTP acceptance 有效。
长上下文 k=1 比 k=2 高 26.03%，但输出不同；代码中 grouped QSA decode
还仅对最多两个 query tokens 开启，后续需独立验证扩展到更多 MTP tokens。
当前没有将 k=2/4 自动推广为默认。原始证据见 replay report。

随后 compact-expert 候选只对同一专家的匹配行计算，复用现有 workspace、
不新增常驻专家展开；163 项 NPU tests 与三个真实整模型 smoke 通过。
MTP k=2 + 三-token FULL replay 有效，含重复专家的八-token partial-layer
graph 从 5.942 降到 4.513 ms。完整模型短三题为
**15.000/12.669/12.926 tok/s**，中位数 **12.926**，比旧 13.034 低 0.82%；
输出与 acceptance 改变，不宣称整模型提速。23.4k 长三题也完成：
**10.946/9.777/9.251 tok/s**，中位数 **9.777**（比旧 10.120 低 3.39%），
三个输出 SHA 都变化。下一步独立验证 W4 的三/五-token batched QSA
decode，仍保留 MTP 与 graphs；当前服务留在 :8002，未替换 W8 launcher。

新的隔离 QSA 候选仅对 routed-W4 将 grouped decode 上限扩展到八-token，
并预建对应 group-list；其它 backend 与 W8 仍为两-token。128 项 CPU、
72 项 NPU QSA tests 通过（含 30 项动态 replay）。Q/KV=6/1 的三-token
QSA-only graph 从 2.071 降到 0.297 ms，不能视为整模型同比提速。
`mtp2-r6` 已通过三个真实权重 smoke，MTP k=2 与三-token FULL replay
均有 runtime 证据。短 coding 15.167/12.555/13.601 tok/s，中位数
13.601，比 12.926 高 5.22%，但输出与 acceptance 均变化。
23.4k 三题为 12.856/10.933/10.919 tok/s，中位数 10.933（比 9.777
高 11.82%）；每题生成 512 tokens、复用 23,168 prefix tokens，输出
SHA 均变化。cold TTFT 357.285 秒，未宣称 W8 parity 或生产等价。
下一步采集保留 MTP/graphs 的新整模型 trace，不能沿用旧 profile 比例。

该新 trace 已完成：W4 projections 和 QSA index scoring 分别占每 rank
累计 task time 的 30.64–32.83% / 23.28–24.24%，不是可相加的
critical-path 占比。进一步仅对 routed-W4 放宽 score GEMM 的两-token
上限到八-token；W8/default 和其它安全 guards 不变。63 项 CPU、84 项
NPU QSA tests 通过，包含动态 query/页表/positions 与精确 ties。

`mtp2-r7` 仍保留 MTP k=2、FULL `[1,3]`、TP4/EP，三个真实 smoke
全部正确。短三题中位数 **13.363 tok/s**（旧 13.601）；约 23.4k
长三题 **14.440/13.683/12.241 tok/s**，中位数 **13.683**（旧 10.933，
+25.15%）。全部六题完成 512 tokens，长题均复用 23,168 tokens。
输出与 acceptance 有变化，不将全部差异归因于单个算子。cold TTFT
356.843 秒，仍未改善。权重内存仍 19.3821 GiB/rank、graph 0.38 GiB；
W8 launcher 未修改。无 profiler 的实际 tok/s 仍低于 W8 的
19.073/18.091；不声称完整任务质量、生产速度等价或更大容量已验证。
证据见 `artifacts/qwen38-w4-offline/replay-r4/` 与 `REPLAY_PROFILE.md`。

相同代码的 k=4 / FULL `[1,5]` 重测也完成：三个 smoke 正确，六个
512-token 请求完成，五-token runtime replay 有证据。短三题为
15.783/11.199/12.189、23.4k 三题为 15.843/12.025/11.392 tok/s；
中位数 **12.189 / 12.025** 均低于 k=2。所有输出 SHA 改变，部分题目
draft 接受率降低。没有用 counting 的 20.198 tok/s 代替真实 coding
吞吐，也不提升 k=4 为更快默认。当前最佳长题配置仍是 k=2，正在重新
profile QSA 修复后的剩余成本；W8 基线 19.073/18.091 仍未达到。

新 trace 已完成，native QSA score 每 rank 从 112 calls 降至 8；
W4 projections 仍是最大具名算子成本，占累计 task time 的 41–43%，
不是 critical-path 占比。进一步复用 QSA position 商的候选通过
80 CPU / 92 NPU tests，20 个局部 selection cases 均精确一致；
三-token / 5,856 groups 的 callback 从 0.510 降到 0.397 ms。
局部 gain 不能等同 tok/s。`mtp2-r8` 通过三个正确 smoke，继续使用
MTP k=2 与有 runtime 证据的 FULL `[1,3]`。三题各生成 512 tokens，
短中位数 13.213 tok/s，比旧 13.363 低 1.12%，没有短题提速证据。
约 23.4k 长三题也完成，14.639/12.990/12.119 tok/s，中位数
12.990（比 r7 的 13.683 低 5.07%）；均生成 512 tokens、复用
23,168 prefix tokens。cold TTFT 358.321 秒，输出与 acceptance
改变，没有整模型提速证据，仍未达到 W8 基线。

新的固定形状 greedy rejection 路径移除均匀 2–8 drafts 的 host-count
H2D 和动态 boolean indexing，保留 k=1、ragged 与 random fallback。
55 CPU、64 NPU changing-input graph tests 和既有 21 sampler UT
通过。两 drafts / batch-one 的独立 eager sampler 从 1.824–2.022 ms
降到 0.182–0.207 ms；这不是整模型同倍数提速。
`mtp2-r9` 三个真实 smoke 正确、MTP k=2 和 FULL `[1,3]` runtime
replay 有证据。短三题全部完成 512 tokens，15.358/13.239/13.729
tok/s，中位数 **13.729**（比 r8 高 3.90%）；输出和 acceptance 改变，
不将全部差异归因于新 sampler。23.4k 三题也完成 512 tokens，
15.510/13.689/12.834 tok/s，中位数 **13.689**；均复用 23,168
prefix tokens，cold TTFT 357.555 秒。W8 基线尚未达到。

进一步将 routed W4 的 gate/up 在加载时写入一个 packed bank，
以一次 N=1280 projection 替代两次 N=640。无常驻重复 bank，W8
未改；950 个 Qwen CPU tests 通过（7 skipped），另有 7 个 NPU
prefill/changing-route replay cases 通过。真实 layer 对照精确一致，
局部 graph latency 降低 4.67–9.24%，不是同等比例整模型提速。
`mtp2-r10` 三个 smoke 正确、MTP k=2 / FULL `[1,3]` runtime
replay 有证据；短三题 15.928/13.148/13.841 tok/s，中位数
**13.841**（比 r9 高 0.82%）。23.4k 三题也完成 512 tokens，
15.599/13.903/12.821 tok/s，中位数 **13.903**（比 r9 高 1.57%）；
三题均复用 23,168 prefix tokens，cold TTFT 324.894 秒。
这些固定长度请求不等于完整 coding 任务正确率评测。隔离 :8002
服务保持运行，生产 W8 未改。
权重仍 19.38 GiB/rank，但 graph memory 从约 0.38 增至
0.53–0.54 GiB；完整 W8 速度目标仍未达成。
