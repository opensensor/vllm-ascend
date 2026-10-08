# GLM 夜间性能实验：2026-10-08

本轮在已有 310P3 / TP4 / MTP1 服务上减少重复归约、转换与布局复制；保留
`fp16_storage_fp32_compute_v1` 专家尺度格式、INT4 Cube 计算及 FP32 累加。
详细的 US English 报告见 [README.en.md](README.en.md)，复现见
[RUNBOOK.md](RUNBOOK.md)。语言质量由用户评估，本轮不声明模型准确率通过。

## 已保留的路径

- v925 仅 gather/scatter 选中的 KDA 状态行，避免每层复制整个持久状态 bank。
- v971 使用向量单元完成独立 4×4 Sinkhorn 归一化，保留 PyTorch 在 128 行
  前后的不同求和顺序。210 个硬件 cases 逐位、changed-input replay 和 guards
  全部通过。
- v941 QSA metadata 与现有捕图组合，descriptor 在 host 根据 shape 准备；
  适配 frozen legacy helper 的 API，测试服务 shape 的 metadata fallback 为零。
- KPool 仅写已完成的池与保留的 tail 状态，保留永久 compressor 及 FP16
  key cache dtype；将合格的整数除法接到 v927。converter descriptor 在捕图前
  重新分配，避免引用已释放的旧 graph allocator pool。
- v975 在永久 target/draft converters 上向量化 BF16 modes 0/1/4/5；mode 2
  保留 v967 query 路径，mode 3 保留原实现。76 项硬件检查覆盖 exhaustive
  BF16 bits、FP16 舍入边界、guards、changed replay 和 640×32×128 shape。
  310P FP32→FP16 使用 `CAST_NONE`；`CAST_RINT` 不符合所需舍入语义。
- KDA beta 向量化保持 FP32 乘法及最终 FP16 舍入，使用三个独立 UB buffers。
  完整 operator 的 17 个 safe-gate cases，所有返回值、recurrent state 与重复
  执行逐位一致，输入不变。640-token operator 39.86→34.74 ms；服务 profile
  中 34 层 W/U preparation 合计约 207→34.69 ms。仅替换原 KDA vendor slot，
  保留其余 OPP 与 host libraries。

独立 beta v978 probe 有 70 个 shape/layout cases，每项六种 changed replay；
frozen 与执行的 probe source hash 一致。另两个 nonsafe cases 在 parent 已
nonfinite，不属于通过范围。生产 GLM 始终保持 safe gate。

## 完整服务测量

温度零；generation 请求强制输出 128 tokens，以第一段文字为 TTFT，以
`(completion_tokens-1)/(last_text-first_text)` 为生成速度。冷请求主动清空 prefix。
这些是按顺序各运行一次的 synthetic token-ID 实验，不是统计学加速结论；
MTP acceptance 比真实文本更高。

| 候选 | C1 两个 prompt | 冷 640 TTFT | 冷 6,400 TTFT | C4 总吞吐 |
| --- | --- | --- | --- | --- |
| vector conversions 与其它 fusions | 9.67 / 10.32 tok/s | 6.16 s | 72.85 s | 9.96 tok/s |
| 同上 + 永久 W4 down7 | 9.75 / 10.31 tok/s | 6.20 s | 73.09 s | 9.95 tok/s |
| 同一 W4 layout + vector beta KDA | 9.50 / 10.46 tok/s | 5.99 s | 70.84 s | 10.02 tok/s |
| beta + 合格 full-chunk score reduction | 9.63 / 10.16 tok/s | 5.85 s | 70.32 s | 10.23 tok/s |
| 同一 W4 layout + vector tail W/U | 9.69 / 10.29 tok/s | 5.85 s | 70.07 s | 8.69 tok/s |
| 恢复原较小 layout + 全部合格 fusions | 10.24 / 10.26 tok/s | 5.96 s | 69.97 s | 10.02–10.11 tok/s |
| 较小 layout + gated-key reuse | 10.07 / 9.51 tok/s | 5.63 s | 68.17 s | 10.00–10.68 tok/s |

C4 包含 prefill，不是 steady-state aggregate decode。vector beta 三个直接文本
completion 的 C1 为 **9.27 / 8.63 / 8.96 tok/s**，不能声明真实文本稳定超过 10。
原始响应已保存，未将其作为 chat-template 或语言质量检查。

W4 down7 将七个 W3 down banks 无损提升为 W4 存储，每 rank 增加 **504 MiB**。
真实权重 2/8/640-token 配对、decode/prefill 调度与 changed replay 逐位一致，
但服务结果未显示完整模型加速。W3/W4 都已经使用 INT4 Cube，此变更只影响
重建与存储，不能解释为从 FP16 改为 INT4 math。

仅 draft 改为 A8 的三个文本请求为 9.48 / 8.26 / 7.23 tok/s；A4 为
9.64 / 8.58 / 9.04。acceptance 未稳定提高，保留 A4。不同实验生成文本有变化，
不能据此推断语言质量排序。

## Profile 与剩余成本

![真实服务 rank-zero profile](profiles/serving-profile.png)

使用本轮 CANN traces。图上半部分同时包含多项变更及 profiler overhead；
下面数值来自最后一个 profiled step。task duration 总和不是 critical path；
KDA stage 根据 source 中九次 launch 的固定顺序映射，而非 trace 内部 stage 字段。

- Decode step 243.95→211.22 ms；prefill 最后 640-token step 7,542.48→6,755.72 ms。
- 每步 ReduceSum 与对应 MemSet 各从 3,747 降为 237 次。
- KDA score finalization 仍为约 994.72 ms。safe-gate Cube score stage 是 no-op；
  不能关闭 safe gate 来绕过长累计 decay 的 FP16 factorization overflow。
- BF16 conversion 在 intermediate trace 为 decode 1.04 / prefill 2.30 ms；
  vector 路径约 0.15 / 0.14 ms。
- AI_CPU FloorDiv 仍有 decode 12 / prefill 24 次，以及 SearchSorted、小型
  reduction/scatter。并非完全消除 AI_CPU，也不是完整无 eager 的 prefill fusion。

部分 chunk 存在明显 cliff，实际主因见后续 stage trace：63-token KDA 约 60 ms，而 64
约 3.6 ms。优化前服务 63-token TTFT 2.98–3.09 s，64 为 1.05–1.09 s；639 为
8.38 s，640 为 5.89 s。实际服务差异还包含 shape、routing 等变化。

第一次 vector score reduction 完整 chunks 正确，但 tails 出现错值且完整
chunks 更慢；未装入服务。修订版每行统一合并 partials，要求 full-chunk cache
生效，tails 保留 scalar reduction；17 个 safe cases 均逐位通过，640-token
KDA 约 35→33 ms。部分 cache 门槛与 cliff 相关，但简单扩展 cache 未消除 operator 延迟，且
63-token 的 recurrent state 与其它返回值错位；更短的多 chunk case 也改变
solve matrix。该 candidate 未进入服务，staging tool 显式记录
`serving_eligible=false` 与已知失败。合格的 full-chunk score serving 结果在
后续章节记录。

## Prefix 与运行范围

重复 6,400-token prompt 实际命中 **5,760 prefix tokens**，冷 71.76 s、重复
7.55 s。vector beta 下 1,280-token 为冷 12.53 / 重复 6.69 s。主动清 prefix 的
benchmark 命中率自然为零；最后 640-token block 仍需要处理。

服务端口 **8001**，四个 310P3 chips / TP4，MTP1，4 request slots，640 chunk，
配置 context **311,040**。最大本轮 prompt 为 6,400，未验证配置容量上限。
decode graphs 为 2/8；prefill 为 **46 segments / 45 eager boundaries**。
EP / FlashComm1 未单独变更或验证，image/video 禁用，无 dummy 测试；保持用户
已有配置，不扩展为 skill 默认的 16-slot 容量基线。

测量 JSON、四 rank receipts、frozen `.py.txt` controllers 与 native archives
位于本目录。CANN 原始 traces 保留在远程
`/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/night-speed-20261008/trace-*`。
大 `.pt` 与模型 shards 未提交；W4 plan/export JSON 保存为 gzip 原始字节。
staging tools 校验 parent SHA、拒绝覆盖，compiler flags 默认关闭，production
header 没有被修改。最终检查与当前日志路径见 runbook。

## Tail cliff 的实际归因

63/64-token 新 trace 显示主要成本是 **stage 4，post-W/U finalization**：
34 层合计 1,862.10 对 1.96 ms。score finalization 为 130.58 对 92.55 ms，
因此 column cache 不是主要解释。stage 4 的 `ComputeTailWuRow` 在 310P 上
对每行、每 channel、每个 coefficient 使用嵌套 scalar GM 读取；完整 chunks
走 Cube。[10 月 5 日报告](../prompt-profile-20261005/README.md) 已记录 cache-tail
拒绝，本轮复现了该失败。

合格的 tail 候选使用独立 coefficient/value/accumulator/output buffers 与向量广播，
避免旧实现注明的 scalar-read UB lane corruption。保留所有 j 项与 W 的
倒序 alias overwrite。有限 FP16 乘积可在 FP32 精确表示，FP32 顺序求和与
最终 FP16 舍入不变。仅 BT64、1–63 rows、128 channels 启用，其它 geometry
保留 scalar。完整 gate 与 serving 数据见下文。

首个 tail W/U vector build 的 16-token 对齐 case 正确，63-token operator
59→8 ms，但其它非对齐 coefficient load 错位；未进入服务。修订版读取每个
prepared score row 实际拥有的完整 BT64、64 个 half，再仅计算有效 tail 项，
绕过该路径的短 `DataCopyPad`。失败与 build 记录保留在
`measurements/kda-tail-wu-rejected-v1-*`。

修订的 tail W/U v2 **17 个完整 safe-gate operator cases 全部通过**：所有
返回值、recurrent state 逐位相同，repeat 相同且输入不变。63-token 为
8.13–8.22 ms 对约 59 ms；16-token 约 0.96 ms 对 2.9 ms。保留 beta/score
fusions 的 640-token operator 仍约 32.4–32.7 ms。package 归档为
`qualified-kda-tail-wu-v2.tar.gz`，并已装入实际服务。重复 cold 63-token TTFT
从 2.98→**1.27 s**，首次请求 1.75 s；639-token 8.38→**6.60 s**。
64/640-token 仍约 1.04/5.82 s。新 trace 的 rank-zero stage 4，63-token 34 层
合计 **1,862.10→123.34 ms**；64-token 仍 1.92 ms。63-token 的 score
finalization 仍为 130.45 ms。以上为顺序测量，非统计学稳定加速承诺。

![tail finalization 与实际 TTFT](profiles/tail-profile.png)

完整 operator gate 验证 eager 重复执行逐位一致；独立 beta probe 验证
changed-input graph replay。真实权重服务重启后重新捕获 decode 和分段 prefill；
不将这些证据描述为每个完整 KDA case 的 changed-input capture sweep。

## 检查与剩余工作

本地目标 CPU suites **1,833 passed / 19 warnings**。三个既有 GLM test 文件因
CPU 环境缺少 vLLM/NPU runtime dependencies 被排除，归档日志含完整命令。
完整真实权重 NPU operator 与实际服务请求提供硬件验证；未声明全仓 test suite
或模型准确率全部通过。

已运行要求的 `bash format.sh ci`。全仓存在原有 formatting、spelling 与
forbidden-import 问题；仅恢复 isolated worktree 中无关 formatter 改动。
修复自有文件格式后，全部 scoped manual pre-commit hooks 通过，未增加
allowlist 或跳过 hook。未修改并行 Qwen 工作。

长 context 冷 prefill 仍需优化 score finalization、重复 expert 重建/权重读取
以及 attention/indexer eager boundaries。tail vector 解决短 remainder chunks，
并非完整大 context 流水线。C4 repeat 另记，不声明 tail package 带来稳定 C4
加速。

## 较小 checkpoint 的最终服务选择

额外 W4 down7 checkpoint 保留在磁盘，但从 serving 移除，每 rank 回收其
**504 MiB** 权重。八个 native resources、beta/score/tail 向量化均保留；重新
捕获 decode 与 segmented prefill。scale bank 仍为每 rank 43、152,174,592
FP16 bytes。启动报告 KV cache 335,184 tokens，未将配置 311,040 视为容量验证。

三次 C4（含 prefill）为 **10.03 / 10.11 / 10.02 aggregate tok/s**。
冷 63/64/639/640/1,280/6,400 TTFT 约
**1.27 / 1.08 / 6.58 / 5.96 / 12.82 / 69.97 s**。
synthetic C1 生成 **10.24 / 10.26 tok/s**；首次 TTFT 含 first-use 成本。
直接文本 completion 为 **9.16 / 7.83 / 7.50 tok/s**，该文本 pass 与 CPU-only
kernel build 重叠，且 continuation / acceptance 不同，不据此声明因果回退
或提升，也不声明真实文本稳定 10。

独立 1,280-token cold / repeated 为 **12.59 / 6.74 s**，实际 prefix hits
为 **640**。所有请求完成，四 rank graphs clean、API unpaused；准确启动参数
与日志见 runbook 和 frozen process/launch receipts。

## Gated-key reuse 候选

K×K 与 Q×K scores 原先重复计算相同的 decay-gated key columns。新实现保留
每个 score row 内分离的 column passes 和 GM-output fences，以 **16 KiB
per-core UB** 保存已舍入的 FP16 gated keys。每个 first pass 必须重置 writable
scratch pointer，否则旧 cached alias 会覆盖先前 columns。首个 build 在 review
发现此问题后取消，未执行硬件测试；修订 v2 已冻结并通过完整 operator gate。

**17 个 safe-gate cases** 的十二个返回值、carry、repeat、inputs 全部逐位或
保持不变。两个 nonsafe cases 仍 nonfinite，不属于合格范围。每项七个 eager
samples 的 640-token median：**BSND 32.79→26.14 ms，BNSD 32.58→25.84 ms**。
tail geometry 不变；未提升或重排 FP16 运算。flag 默认关闭，production source
未修改。hash、完整 gate、取消 receipt 与 frozen build 均保留；package 归档
`qualified-kda-gated-key-reuse-v2.tar.gz`，完整 serving 结果见后续记录。

## Gated-key reuse 完整服务结果

合格 package 已载入原较小 checkpoint，保留其它 fusions。顺序测量的冷 TTFT
为 **640-token 5.63 s、1,280-token 12.32 s、6,400-token 68.17 s**；前一 pass
为 5.96 / 12.82 / 69.97。synthetic C1 **10.07 / 9.51 tok/s**；三次 C4（含
prefill）**10.59 / 10.00 / 10.68 aggregate tok/s**。此变更针对 prefill，不能
据此声明它提升了 decode。

独立 1,280-token prefix 测量为 cold **12.02 s** / repeated **6.44 s**，实际
hits **640**。chat 请求正常结束，返回非空的一句解释。直接文本若以完整请求
结束作为 generation endpoint，为 **9.39 / 7.78 / 6.92 tok/s**；首次请求若
只取 last visible text 会得到 11.98，但遗漏其后约 2.9 s。
`full-generation-timing-summary.json` 保存原 response SHA 与
`(completion_tokens-1)/(total_s-ttft_s)` 推导，原始记录不变；未推断该 tail
的内部原因，且 SSE chunk 可含多个 tokens。真实文本稳定 10 的目标仍未达到。

## 最终 trace 与服务 receipt

rank-zero profiled decode **205.66 ms**，最后一个 640-token prefill step
**6,515.00 ms**；之前 vector-beta 为 211.22 / 6,755.72 ms。
34 层 score-finalization task sums **994.72→702.97 ms**，包含 batched reduction
与 reuse 两项变更，不能单独归因给 reuse。beta preparation 仍 34.34 ms；完整
KDA stage tasks 合计 838.70 ms。

剩余主要 AI-core task sums：expert gate/up **1,635.50 ms**、down
**838.92 ms**、sparse attention **756.29 ms**。这些含 profiler overhead，不是
wall critical path。expert weight 复用/重建与 sparse attention 是下一批更大的
优化对象，不能将所有剩余耗时归因于小 casts。

`running-state-final.json` 验证实际 API PID/create-time、四 ranks、frozen source
SHA、qualified package binary SHA / OPP stack、FP16 scale bytes、clean graphs、
模型注册与 unpaused API。profiling 后恢复同一 worker PIDs / weight digests，
没有重新处理权重。当前 log：
`/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/gated-key-reuse-server-20261008.log`。
