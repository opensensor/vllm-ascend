# GLM 310P 原生 INT4：权重准备与 scale 调度

## 实现与范围

在四个 Ascend 310P3 worker（两张双芯片卡）上运行真实 selective-W3 checkpoint，
TP4、MTP1、decode graph sizes 2/8、context 上限 311040。该轮是短上下文测试，
没有验证满 context、EP 或 flashcomm1；模型没有启用多模态输入。attention、dense
和 shared experts 沿用现有实现，所有 routed experts 使用原生 INT4；MTP 的源布局差异与修正见下文。

- W4 直接读取预排的 packed INT4 bytes。W2/W3 保持原始精度与存储大小，在 UB
  精确扩展为 INT4；没有 FP16 weight matmul。
- token 只量化一次；两段 fused kernel 完成 gate/up + UB SwiGLU/hidden quant
  以及 down + UB FP32 weighted reduction。不生成 GM FP16 expert 中间矩阵。
- 两个相邻 K32 scale groups 在未用 Cube rows 共享一次 K64 Mmad/readback。
  两组各自的 dot、scale 与累加顺序保留。
- v29 将共享 block32 scale 的 even/odd channel strips 合并为 repeated vector
  instruction，并将逐 row Add 改为 active rows 的批量 Add。
- 无损 bank preparation 改为四个 CPU threads 的 byte-field 重排，整 bank 一次
  H2D；每个 expert inverse 必须逐 byte 相同，device D2H SHA256 必须匹配。
  CPU 保留原 bytes，切换前逐 bank 恢复并核对 SHA256；NPU 不保留第二份 bank。

每个 rank 共 86 banks，备份 31,255,953,408 bytes（29.109 GiB）；全机 116.438 GiB。
v28 初次准备及 graph capture 合计 **402.33 秒**，相比旧的小 NPU tensor 操作路径
约 26 分钟，等待明显减少。该时间属于旧在线准备路径。后续已生成完整永久布局 checkpoint 与直接 loader，
不再将这 116.438 GiB 备份作为正常启动的一部分。

## 已完成的 v28 在线结果

[完整请求与四 rank receipt](live-forward-v28-fused_int4a4-trial.json)；
[同一组新 worker 的原实现速度](live-forward-v28-baseline.json)。

| 模式 | c1 tok/s | c4 aggregate tok/s | c1 TTFT 秒 | c4 TTFT p50 秒 |
| --- | ---: | ---: | ---: | ---: |
| 原实现 | 5.190 | 14.401 | 3.101 | 5.199 |
| v28 fused INT4/A4 | 5.130 | 7.650 | 3.490 | 11.814 |

v28 完成 26/26 有效 HTTP 请求；五个速度请求均生成 64 tokens，没有 early EOS。
20 个 exact-answer 题通过 17 个，tool 通过 1/1。失败 ID 为 `instr_reverse`、
`instr_first`、`code_slice`，与重启前同 checkpoint 的
[原实现小质量 gate](live-fused-v24-baseline-quality.json)相同；reverse 的错误字符串
不同。这不是大型准确率、困惑度或长上下文资格评测。

四 rank 的 capture 与 eager prefill Python audit 均记录 native>0、fallback=0。
它们证明分发覆盖 W2/W3/W4、gate/up/down 和 MTP，计数不包含 graph replay。
该轮没有完成 A8 在线吞吐测试；A8 的组件与真实 expert gates 不等价于在线测试。

## v29 算术与隔离时序

[36 个独立参考/变更输入 replay/真实 expert gates](native-fused-v29-gates.json)通过。
相同真实 expert 和合成输入在 tokens=2/8、W2/W3/W4、A4/A8 共 12 组比较中，
v29 与 frozen v28 输出 bitwise 相同：
[两行比较](native-fused-v28-v29-t2-exactness.json)、
[八行比较](native-fused-v28-v29-t8-exactness.json)。

| W4/A4 阶段 | 两行 v28→v29 ms | 八行 v28→v29 ms |
| --- | ---: | ---: |
| gate/up + SwiGLU + hidden quant | 0.652→0.416 | 1.674→0.804 |
| down + weighted reduction | 0.313→0.206 | 0.806→0.364 |

这是一个真实 expert、topk1 routing 的 device launch intervals；不能换算为完整模型
加速倍数。原始 samples 与 hash 在[两行时序](native-fused-v28-v29-t2-stage-events.json)
和[八行时序](native-fused-v28-v29-t8-stage-events.json)。更早的 v22/v23/v24
布局比较亦保留于本目录，不能与不同轮次的 serving 吞吐混算。

## Prefix cache：零 hit 的原因与在线验证

运行时 metrics 显示 `enable_prefix_caching=True`、attention block size 640，
fine-grained Mamba prefix caching 关闭。小质量/速度题的实际 prompts 只有
26–177 tokens，没有完整可复用 block，零 hit 不能据此判定 cache 失效。

保持 v28 原生服务与 graph 不变，连续提交相同 1281-token prompt：

| 次序 | 计算 prompt tokens | cached tokens | 总请求秒 |
| --- | ---: | ---: | ---: |
| 第一次 | 1281 | 0 | 61.989 |
| 相同重复 | 1 | 1280 | 1.138 |

输出均为 `over the`，生成两个 tokens。hit counter 从 0 增至 1280，query counter
从 1145 增至 3707。缓存确实命中；该测试的收益主要是重复 prefill，不能解释
冷 prompt 的 decode 加速。完整 response、前后 metrics 及脚本在
[prefix-cache-v28.json](prefix-cache-v28.json)和
[prefix-cache-check-source.txt](prefix-cache-check-source.txt)。

## 失败原因与修复

v24 的逐 expert NPU 重排超过旧 900 秒 RPC deadline。worker 后来完成 capture，
但旧 executor 留下未消费的 RPC reply；恢复 serving 后 scheduler 将 status dict
当作 `ModelRunnerOutput`，出现 `sampled_token_ids` AttributeError，原 engine 退出。
该请求没有有效性能结果，不能将它算作 native INT4 benchmark 或 NPU 数学错误。
证据：[崩溃摘录](native-v24-rpc-crash-excerpt.txt)。

按照用户“fix forward”要求，从保存的 command/OPP/source hashes/checkpoint 配置恢复
服务，并将 native candidate 装入新 workers。CPU byte preparation 与控制工具的
3600 秒 deadline 修正了本次路径；重排期间只读取独立的 per-bank report 文件，
不提交并发 worker RPC。不能在一次超时后仅凭晚到 receipt 恢复 serving。

旧在线 trial worker PIDs 为 95816 / 96191 / 96617 / 97080。原始 checkpoint
未修改。永久布局 checkpoint 有独立 config/index，未变 dense/scale shards 使用 hard links。
启动命令见[配置](forward-v28-server-process.json)，构建与热切换命令见
[runbook](../../../tools/glm_perf/reconstruction_experiments.md)。

## 本地校验

140 个相关 CPU tests 通过；正常仓库 conftest 需要本机缺少的 upstream
`vllm.third_party.flash_linear_attention`，因此本轮使用 `--noconftest`。
不是全仓测试通过的声明。NPU 使用实际 checkpoint 的 gates 与在线请求，未用
dummy 权重替代。相关文件的 lint/format hooks 独立检查；用户原有修改不纳入提交。

## 永久布局 checkpoint 与 MTP 修正

完整输出位于 Threadripper：
`/srv/ai/models/GLM-5.3-Flash-selective-W3-native-int4-310p`。
共 43 层 × 288 experts × 三个投影 = 37152 code tensors，172 native shards。
`native-layout.json` 在四个 rank 的覆盖与 metadata 校验全部通过后才标记 complete。
W2/W3/W4 bit allocation 与 scale 不变；离线重排不改变 signed code 值。

导出独立对照 canonical checkpoint 时发现：42 个主模型层的旧 bank 是 NZ，
MTP 层45的旧 bank 却是 canonical。旧通用在线重排对 MTP 错用了 NZ 解码。
导出器拒绝发布该 bank；MTP 三个投影随后直接从原 checkpoint 在 CPU 上生成
Cube layout，并逐 code inverse 核对。v28/v29 旧在线结果因此是 **MTP 修正前的
实验数据**，不能作为修正后的永久模型最终资格结果。

`--load-format glm_native_int4` 及 MTP 的 `draft_load_config.load_format` 使用同一个
直接 loader。只按 authoritative index 读取本 rank 的 experts；忽略 hard-linked
source shards 中已被替代的 canonical codes。每个 MoE 组合 native method，包含
主模型与 MTP，默认路径没有 legacy weight-dequant fallback。启动报告记录
`transformed_code_tensors=0`、`layout_backup_bytes=0`；真实服务校验另附。

## 永久布局 v29 的真实服务结果

[26 个请求及前后 worker receipts](native-disk-v29-trial.json)。

| 模式 | c1 tok/s | c4 aggregate tok/s | c1 TTFT 秒 | c4 TTFT p50 秒 |
| --- | ---: | ---: | ---: | ---: |
| 永久布局 v29、修正 MTP | 6.517 | 10.654 | 3.425 | 9.155 |

五个速度请求均完成 64 tokens；quality 18/20，tool 1/1，26/26 有效 HTTP。
剩余失败：`instr_first` 输出 `Red` 而非 `red`，`code_slice` 输出带 backticks 的
`lan`。reverse 已通过。主模型权重加载 **42.53 秒**、draft **1.31 秒**，四 rank
总 load_model 为 48.46–48.84 秒；旧主模型 load_weights 为245.96秒，之后另有
402.33秒的在线 bank preparation/capture。启动导入、warmup 与 cache/capture
时间不包含在上述 load_weights 比较中。

新 worker PIDs：327411 / 327821 / 328263 / 328726。每 rank 的
[native loader audit](native-disk-load-report-rank0.json)显示 main 84 banks/9072
code tensors、draft 2 banks/216 code tensors，全部 method 为
`NativeInt4MoEMethod`，零 transformed codes、零 rollback backup bytes。
其他 rank 的同名 audit 保存在本目录。[启动配置](native-disk-v29-server-process.json)。

首次直接 mmap→NPU 尝试卡在 Ascend driver 的 `pin_user_pages_fast` / huge-page
splitting / TLB flush。短 host perf trace 留在
[profiling 目录](../native-trace-20261006/startup-stack-report.txt)。按单 tensor
在普通 CPU storage staging 后载入成功；staging 是无损 byte copy，不生成 FP16
weights，不做 quant/packing，不保留全模型备份。

新的四-rank c1/c4 CANN trace 已收集；stop/finalize 后恢复相同 native disk model、
相同 workers/weight storage，并完成八-token smoke。
[profiling 报告](../native-trace-20261006/README.md)单独区分 profiled device task
时序与上述 unprofiled throughput。
