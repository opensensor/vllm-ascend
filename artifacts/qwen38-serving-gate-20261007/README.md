# Qwen Flash-Next 310P 实机复测：2026-10-07

本轮验证完整真实 W4 权重及最新 GDN decode 候选。冷 23K 前缀约
59.4 秒，随后确认真实 prefix-cache 命中；GDN 尚无一致的整模型吞吐收益。
当前恢复服务使用 `[3,9]`、三个 active requests、1024-token scheduler
batch，并启用图片。cache fraction0.70 新 profiling 提供997,292 cache tokens
（3.80 × 262144 的 planner 容量），rank0 archive194 slots、post-capture
free9.30GiB、reserve4GiB。此前四请求图片实例触发96°C watchdog后已退出。
当前状态见 [recovery-readiness.json](recovery-readiness.json)，复跑见
[RUNBOOK.md](RUNBOOK.md)。

## 环境与配置

- 主机：`matteius@192.168.53.187`，四颗 Ascend 310P3。
- API：`http://192.168.53.187:8000/v1`；模型 ID：
  `qwen38-flashnext-latest-gate`。
- 模型：`/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i`。
  48 层、512 experts、top-10、MTP 层数 1；配置最大上下文 262144。
- 当前完整 runtime：`/srv/ai/src/qwen38-demo-recovery-runtime-20261008`。
  完整复制原 `qwen38-decode-next-runtime-20261005`，再应用已提交的
  `483b57819` metadata 与 mixed-prefill 修复。此前 A/B 使用原快照。
  GDN 候选 SHA256 为
  `331b32da4f832247d16500d1beef118221cdd8f7db46c31150a45e23f5995c22`，
  与本地主工作树候选一致。没有拼接共享工作树中的 GLM 修改。
- TP4/EP4、MTP2、FP16 参数、FP32 recurrent state、native INT4 W4A8；
  built-in FP16 SwiGLU、CANN finalizer、2560-token grouped prefill。
- 当前环境是既有独立 venv/source snapshot；主机没有 `/workspace`。
  本轮沿用已资格验证的 launcher 和 OPP，不更换 transformers 或安装根目录。

GLM API 的 PID、命令、工作目录和日志先保存，再停止明确识别的服务。
其 NPU workers 退出后，残留等待的 API parent 单独终止。
GLM 的完整私有重启环境仅保存在远端权限 0600 文件中，没有复制进仓库。

首轮 affinity helper 的固定 8001 校验拒绝了端口 8000；使用仅修改该
身份校验的独立副本，并实际确认四个 worker 的六核 CPU mask。
第二次启动直接使用此副本。未修改原 launcher 或生产 helper。

## 启动、功能与容量

使用完整真实权重，没有 dummy-only 验收。首轮四个 rank 的总模型加载
时间为 111.39、112.16、111.21、112.47 秒；C4 服务为 102.05、121.75、
122.17、121.08 秒。首次加载的 affinity 修正发生较晚，不能将这些时间
解释成受控 loader 性能比较。

前两个服务都报告 1,068,936 cache tokens、262144-token 请求的最大并发
4.08x，并成功捕获各自的两个 decode graph。所有候选切换都保留本次
加载的四个 worker PID 和 weight-storage digest，完成 cache reset、
图重捕获及请求恢复。

原始配置、两个 residual control、两个 GDN 候选，以及 C4 服务都保存了
结构化 text/tool smoke 和 thinking-enabled `17 * 23` 请求。
smoke 保留 **6/7**：反转 `ASCEND` 应为 `DNECSA`，模型返回 `DNESCA`。
其余算术、文本和工具调用通过；此已知错误不能标成全质量验收通过。

| 功能 | 本轮证据 |
| --- | --- |
| EP | 四 rank 各加载 128/512 experts，完整 HTTP 生成 |
| MTP2 | 真实 target/draft 权重、draft/accepted counters、持续生成 |
| ACLGraph | `[3,6]`、`[9,12]`、最终 `[3,12]` 捕获；resident recapture |
| 冷分块 prefill | 8192/23410 exact-token prompts，均 `cached_tokens=0` |
| 暖 prefix cache | 23410-token repeat：23296 cached tokens，TTFT 1.895 秒 |
| FlashComm1 | 既有 310P 资格配置关闭，本轮没有启用或宣称验证 |
| 多模态 | A/B text-only；最终启用图片，用户自行验证；video 关闭 |

本轮实际并发测量为四个短请求，没有提交真实 `4 × 256K` 请求。
当前 cache 总量约能容纳七个完整 128K 请求，不足以容纳 `16 × 128K`；
未将 `max_num_seqs` 扩至 16，也未更改已资格验证的 state/cache 几何。

## GDN A/B：独立 `[3,6]` 图配置

control 使用已通过历史实机资格验证的 native HC residual；候选在此
基础上只替换最多六 token 的 GDN output RMSNorm，并缓存小型 FP32 gamma。
大 prefill 和 C4 的 12-token decode 保留原 GDN 路径。
次序为 A1/B1/A2/B2，每轮三个固定 prompt 各生成 512 tokens。

| 指标 | Residual control | Residual + GDN |
| --- | ---: | ---: |
| 六次 serial 的 server-gap tok/s 中位数 | 28.768 | 28.576 |
| 两次 C2 aggregate tok/s 中位数 | 41.375 | 40.699 |
| 两次冷 8192-token TTFT 中位数 | 20.205 s | 20.051 s |
| 两次冷 23410-token TTFT 中位数 | 59.400 s | 59.419 s |

六个按 prompt/轮次配对的输出只有两个 text hash 相同；这两个样本的
速度分别提高 5.79% 和 0.78%。配对速度比中位数约 +2.33%，但 pooled
tok/s 中位数约 -0.67%，C2 约 -1.63%，且恢复 control 也有输出漂移。
因此不能宣称普适 decode 提升或归因冷 prefill 收益。

候选额外完成 2048-token sustained decode：33.652 client tok/s，无停顿
或请求失败。它是一个功能/稳定性样本，没有对应同长度 control。

首轮 counter helper 仅在计时外等待一秒，而 engine 周期发布统计。
保留的 draft/accepted delta 可能跨请求边界，不能当精确 per-request
acceptance 或计算准确 round time。后续 C4 重放在请求前后等待 11 秒。
decode 速率以 server token-gap 日志为准，C2/C4 aggregate 使用总输出
tokens / concurrent wall time，TTFT 独立统计。

## C4 图配置与最终运行态

首轮 `[3,6]` 的 C4 eager aggregate 为约 36.6–39.0 tok/s；原日志中的
`decode_tokens=12` fallback 是真实性能限制。这些值没有当作 C4 图结果。
独立重启 `[9,12]` 后，分别交替测试原始 HC 与 native residual 三轮。
所有 C4 请求共用相同参数与短 prompt 集合；每轮记录 hash、server-gap
timing、aggregate wall time 和 worker 状态，见 `c4_graph`。

四请求配置的 C1 会填充到 9 tokens：原始路径的 512-token C1 样本为
20.234 tok/s，低于 `[3,6]` 的对应原始路径样本 31.113 tok/s。两次输出
hash 不同，因此数值不作为精确配对回归率；填充成本确实存在。

`[9,12]` 下 original/residual C4 aggregate 中位数分别为 58.267 和
58.637 tok/s，差异小于样本波动，不能宣称普适 residual 提速。

用户指出 C1 tradeoff 后，最终改用 native residual 和 `[3,12]`。
先用 0.90 cache fraction 新 profiling：1,254,982 tokens、4.79x；512-token
C1 为 31.306 client tok/s，C4 两轮为 54.696 / 54.667 aggregate tok/s。
冷 23410-token TTFT 为 59.628 秒，repeat 为 1.903 秒并命中 23296 tokens。
六个短会话及后续请求全部完成，但后续请求 cached_tokens 均为 0；这不是
六个已保留完整长窗口的证明。

此前 0.95 profiling：逻辑 KV 预算 105.68 GiB，1,331,971 tokens、5.08x。
compact Mamba pool 仍为 64；primary slots 63、device archive 32（0.87 GiB）；
post-capture free 4.88 GiB，reserve 4.00 GiB。没有使用固定 KV budget
跳过 profiling，也没有提交真实 `5 × 256K` 压力测试。
`demo5-health.json` 保留最终容量与资格 receipt；最终 128-token C1/C4、
4096-token 冷分块 prefill、text/tool/thinking smoke 均完成，服务 unpaused。
最终短 C1 为 29.819 client tok/s，cold 4096-token TTFT 为 10.289 秒。

GDN 候选仍保留为独立实验，没有
启用到最终演示服务；它的 decode-only 资格范围也不覆盖 9/12-token 图。
未启用此前发生 regression/stall 的 native gated-mean 候选。

## 图片与 checkpoint headroom 的取舍

真实会话揭示 0.95 profile 的 archive 每 rank 仅 12–32 slots，超过 100 个
checkpoints/group 已 spill 到 CPU。恢复次数仅 0–1/group，且长 prefill 与
decode 混合执行，不能把整体低速全部归因于 CPU restore。

用户改为优先 3–4 窗口及图片。停止明确识别的旧服务并确认 NPU 空闲后，
使用独立 `start-demo-vision-profile.sh`：删除 language-model-only，允许每个
prompt 一张图片、最多 1,048,576 pixels，关闭 video，0.75 fraction 重新
profile。该实例 1,001,364 tokens、3.82x；primary 63，archive 各 rank 分别
185/166/181/170 slots。rank 0 archive 5.04 GiB，post-capture free 9.05 GiB，
reserve 4.00 GiB。三份完整 256K 窗口可落入 planner 容量，四份完整窗口
超过该容量；四个 active requests 仍可用于较短上下文。

模型及 ViT 权重完整加载，`[3,12]` capture 和 HTTP readiness 完成。
用户要求自行测试，所以已终止等待中的 `run_vision_gate.py`，没有运行本
agent 的图片语义或三份 32K history gate。用户随后确认 fresh 图片请求
可以处理，实时日志也出现 MM cache hit；这不是独立图片质量 benchmark。
新实例没有重新加载 native residual 实验资源，使用快照内既有 W4/MTP
优化路径。没有把之前 text-only native candidate 的 A/B 归给图片实例。

Kilo 的目标 profile 曾仅声明 text 输入；检查时用户已加入 image。
`kilo models ... --verbose` 确认 image capability 为 true，config check
通过。此前历史回复不是新服务的图片能力证据。没有安装额外依赖或修改
用户其它 provider、选定 agent/model、token budget。

## 热保护后的恢复配置

四请求图片实例在04:57:45触发96°C watchdog并退出，旧 waiting requests
随引擎退出终止，需要客户端重试。此前 metadata 热修复清除了 prefix cache；
后续冷长前缀会延长排队，不能仅凭 unpaused 宣称 waiting 请求已完成。

用户明确要求重新上线后，建立完整 recovery 快照，仅回移 `483b57819` 中
qLens grow-before-write 和 mixed-prefill QSA matrix-scoring 修复。保留原
OPP顺序、FP32 state与pool64；使用 `[3,9]`、max-num-seqs3、scheduler
batch1024和fraction0.70，降低混合 prefill 单步工作量并扩大 archive。
rank0：primary63、archive194（5.28GiB）、post-capture free9.30GiB。
启动报告997,292 tokens、3.80x；完整 `3 × 256K` 压力与图片质量未测。

当前实例没有加载 native residual/GDN 实验资源，也没有应用 resident
metadata candidate；对应修复已在新快照磁盘文件内。模型与 vision 权重加载、
两张 decode 图捕获、API readiness 与 unpaused 状态已确认。用户自行测试；
没有增加 agent inference 请求、重置缓存或再次中断服务。

用户日志中一条请求命中26880 prefix tokens、TTFT2.882秒并生成849tokens，
decode21.1tok/s；另一条143tokens为23.9tok/s。短请求受混合 prefill 干扰仍
可能较慢。这是非受控用户流量，不能宣称普适吞吐改善或热稳定性已经验收。

HF两张 model card 同步记录 text+image 能力；W4 fresh-image 为本轮用户
确认，W8使用相同 frontend，既有 W8 text benchmark 未重跑。只发布 README，
不改权重。卡片源文件见 [W4](model-cards/w4.md) 与 [W8](model-cards/w8.md)。

## 恢复实例仍出现 Mamba spill

用户05:19后的流量在 KV usage约10%时耗尽 Mamba device tiers；archive
每rank分别194/175/191/180 slots。只读 `resident_status` snapshot显示每个
state group已spill10–67 checkpoints，但所有group的CPU→NPU restore_count
仍为0，graph_clean且baseline。说明不能只用 KV planner 或 active-window
数量推断保留会话的 checkpoint 容量，也不能将低吞吐归因于重复 restore。
spill-out仍包含同步与CPU拷贝，实际占比需要独立profile，不能从warning推算。

原始用户流量还出现6.4、5.7和3.1tok/s；吞吐变化未做受控分解。
没有重启、pause/reset、驱逐缓存或改动当前请求。证据见
[只读 Mamba 状态](recovery-mamba-status.json)。两张HF卡明确记录W4长会话
仍可能spill；不能宣称当前profile满足spill-free或持续吞吐资格。

## 驻留切换与后续扩容边界

HC/GDN A/B 都通过既有 resident control 在同一次权重加载中切换。
本轮 graph profile 和 KV allocation 改动需要独立启动；现有 control 没有
同步更新 dispatcher、engine scheduler 与 worker cache allocator 的接口。
最终 demo 已在线，graph/cache budget 热切换功能尚未实现。

用户报告旧 snapshot 的 qLens CPU buffer 从四行被隐式 resize 至十二行。
主仓库 `483b57819` 已有容量修复和 regression test。独立 resident candidate
`query_lens_capacity.py` 回移其 grow-before-write 逻辑，保留 pinned allocation
与 drafting clone；通过已有 drain/reset/recapture 控制器应用，不 reload
权重。四项 focused CPU regression tests 通过；部署 receipt 记录 PID 与
weight-storage digest。没有新增环境变量、NPU 同步或数值内核。

后续 graph/cache 热切换需：阻止新请求进入、drain、验证所有 rank、重建
dispatcher/capture、同步 KV planner/scheduler/worker allocation、清理 QSA
index views 与 Mamba tiers、保留 PID/storage digest、失败回滚并保持暂停。
不能仅改配置字段就宣称支持热切换。

当前 QSA rotary 路径仅处理 theta/mrope，不处理 YaRN factor/rope_type。
单改 max-model-len 或 rope override 不构成 1M 资格；YaRN 也不会增加
KV 容量。2–3 个完整 1M 会话仍需额外 cache 容量方案。

## 主机检查与证据

首组23个 focused CPU tests通过；本次最终发布检查共78项通过：

```bash
python -m pytest --noconftest -q \
  tests/ut/qwen38_1m/test_gdn_output_rms_candidate.py \
  tests/ut/qwen38_1m/test_resident_reset.py \
  tests/ut/qwen38_1m/test_native_hc_residual_candidate.py
```

默认 UT conftest 在此本地环境缺少
`vllm.third_party.flash_linear_attention`，故 focused tests 使用
`--noconftest`；没有宣称完整 UT suite 通过。
Python 编译、focused Ruff、markdownlint、launcher syntax/coherence/provenance
和 coherent OPP symbols 检查保留。仓库级 logger 和 symbolic-meta hooks 已通过 scoped 检查。原始日志和 JSON
中随机 request ID 触发 typos 误报；保留其原值，不改写原始性能证据。
`bash format.sh ci` 在隔离 worktree 执行，已有非本轮文件的 Ruff、codespell、
clang-format、markdownlint 和 forbidden-imports 失败；本轮 scoped 全部通过。
隔离检查产生的旧文件格式改动没有写回共享工作树。

- [完整比较](summary.json)：serial 配对、冷前缀和 C4 图统计。
- [最终演示容量](demo5-health.json)、[最终 worker 状态](demo5-final-status.json)、
  [五窗口 server log](server-demo5.log)：此前 `[3,12]` 与 0.95 profiling 的证据。
- [恢复实例](recovery-readiness.json)、[恢复日志](server-recovery.log)、
  [磁盘修复](recovery-runtime-fixes.patch)：当前运行态。
- [此前图片实例](vision-readiness.json)、[此前图片 log](server-vision.log)、
  [metadata 热修复](metadata-capacity-receipt.json)：已退出实例及驻留切换历史。
- [首轮事件](events.jsonl)、[首轮 server log](server.log)、
  [C4 server log](server-c4.log)：请求和捕获的原始证据。
- `*-smoke.json`、`*-thinking.json`、`*-serial.json`、`*-c4.json`：
  响应、token 数、hash 和 latency。
- `*-reset.json`、`*-switch-*.json`、`*-status.json`：drain/reset、
  PID/storage digest、graph cleanliness 和 native resource 状态。
- [source hashes](source-hashes.json)、[OPP symbols](coherent-opp-symbols.json)、
  [runtime paths](runtime-paths.json)：实际快照与 binary provenance。
- `hardware-samples*.jsonl` 和 `watchdog*.log`：硬件状态及 cutoff 记录。

本轮是既有实验模型的实机验证，没有新增生产模型适配、环境变量或 NPU
内核。交付包含本复测目录、独立 resident metadata candidate 和其 UT。
