# GLM 310P：单份 NZ 输出头、冷启动图修复与低位内核实验

本轮尚未稳定达到 c1 10 tok/s。历史最好单次实验为 v73 的 **9.746 tok/s**；
最新 W3 专用 v80 为 **9.524 tok/s**。最终保留 v56 专家内核，永久 indexer bundle
更新为 v6，并在真实冷启动中验证输出头只保留一份 NZ 权重。

## 实测结果

所有在线实验均使用真实磁盘模型、TP4、MTP1、相同 26 个请求，包含 c1/c4、
20 个质量题及一个工具调用。c1 为 64 completion tokens 的单次结果，包含 reasoning；
不能当作重复采样中位数，也不能把孤立内核加速等同于端到端加速。

| 版本 | 变更 | c1 tok/s | c4 总 tok/s | 质量 |
|---|---|---:|---:|---:|
| v56 单份 NZ head | 原专家路径，输出头格式一次转换 | 8.879 | 见原始 JSON | 18/20 |
| v69 | 重用向量 mask | 9.188 | 15.133 | 17/20 |
| v73 | K128 与重复 INT32→FP32 readback | 9.746 | 14.098 | 16/20 |
| v74 | 可选重复 readback、K128、prefill pairing | 9.666 | 见原始 JSON | 17/20 |
| v75 | 自适应 K256 | 9.539 | 14.781 | 17/20 |
| v56 + 全 post mixer | FP16/FP32 输入，直接四流累加 | 9.560 | 15.070 | 17/20 |
| v56 + post epilogue | 保留原 BMM，只融合乘、加与 state round | 9.431 | 13.975 | 18/20 |
| v73 + post epilogue | 两项组合 | 9.241 | 见原始 JSON | 见原始 JSON |
| v78 | 通用 W3 fragment + 重复 readback，K64 | 8.731 | 13.101 | 17/20 |
| v80 | W3 专用双阶段、解码循环展开、K128 | 9.524 | 14.450 | 17/20 |

v56 的两个格式失败为 `Red`/`red`、带反引号的 `lan`。
`instr_reverse` 不能仅凭一次结果判定数值退化：相同种子重复三次，原 v56 和
v73+epilogue 都只成功一次。原版本还出现空答案和原始 thinking marker。
这是需要继续排查的端到端不稳定性；本轮未放宽答案规则，也未修改 checkpoint。
各版本完整请求与工具结果保留在 `measurements/`。

## 已落地的路径

部分量化模型的 head 被上游分配为普通 `UnquantizedEmbeddingMethod`，310P 构造器
只检查 `quant_config is None`，因此每步重新转换整个 vocabulary 权重。
现在对实际未量化的普通 head 选择 NZ 方法，保留专用量化方法及其子类。
head 原 Parameter 不变，`weight_nz` 是同一 storage 的 Tensor 别名；embedding lookup
仍保留 ND 权重。避免额外注册 Parameter，也避免额外 303 MiB/device 常驻副本。

真实单卡完整 head `[38720,4096]` 的 1/2/8 行验证均逐位一致。
独立 probe 中 ND linear 约 **5.62 ms**，NZ linear 约 **1.70 ms**；
转置 NZ GMM 约 1.98 ms，未采用。早期双份缓存导致 recapture OOM，随后改为单份布局，
CPU 逐字节比对及所有非 head storage 指针校验通过。冷启动无需该 CPU rollback 副本。

BF16 helper 首次遇到新形状时原来使用 `torch.tensor(..., device='npu')`，在冷启动
图捕获中触发同步 H2D copy，导致 `aclrtMemcpy 107030`。现在用设备 `empty` 与
`fill_` 创建两个 INT64 descriptor 值，覆盖第一次调用就在 capture 内的场景。
新 indexer v6 使用原 v5 的同一 kernel/bridge 二进制，仅 helper 改变；18 个冷捕获转换
及 8 个真实 compression/replay gate 全部逐位一致。专家权重与 scale 不变。

## 算术、融合与选择边界

原生 post mixer 可以保持 FP32 专家输出到最终 state round；没有先 `.half()`。
半舍入边界测试明确证明提前 cast 会改变结果。不过直接四流 Axpy 与原 BMM 存在少量
舍入差异，未设为默认。`--finish-only` 保留原 einsum/BMM 的四流归约顺序，只融合
broadcast product、add 和 FP16-in-FP32 state round，1/2/8/640 行及 changed-input
replay 均逐位一致。640 行孤立 FP32 mixer 从 **6.320 ms 降至 5.579 ms**；
小 decode 的孤立时间并未改善，端到端 c1 仍低于 10。

W3 crossing fields 可写为 unsigned low fragment + signed high fragment；其值在
FP16 中完全精确，减少 INT16/FP16 往返。通用分支 v78 虽然正确，却更慢。
v80 额外编译 W3 gate/up 与 down 两个专用入口，以常量位宽和循环展开去除 scalar
分支，W2/W4 保留通用入口。选择仅依赖已校验的 host geometry，不读取 NPU 数值。

v80 的真实 W3/A4 单专家 t2 gate/up：**0.588→0.500 ms**，down：
**0.297→0.252 ms**；640 行分别为 **50.100→48.018 ms**、
**24.976→23.629 ms**。36 个独立 gate、真实 W2/W3/W4 权重及 A4/A8 组合、
全输入/route/weight 变化后的 replay 全部通过；对 v56 的 t2/t640 输出逐位一致。
完整 v80 在线 26 请求为 17/20，工具调用成功，未更新默认专家 bundle。

decode shadow 额外把 v73/v56 放在同一 capture 中比较当前激活与路由：每 rank
172 个 slot，非零输入检查分别为 129/86/86/86，全部逐位一致。
诊断 Tensor 引用存在保留旧 graph pool 的风险；恢复捕获发生 rank-local OOM，需重启。
此原始诊断仅作为证据；重用时应释放 slot 并保留 wrapper 的原实现标记，不能直接作为
长期 serving policy。此前 eager parity 只覆盖 prefill，不能替代这次 decode 检查。

## 验证及运行状态

- 隔离 HEAD review worktree 的 `tests/ut/glm_perf`：388 passed，排除三个已有外围依赖测试。
- 新 single-copy head NPU 测试：4 passed；mHC（含首次 capture 与 FP32 tie）：10 passed；
  BF16 首次 capture：18 passed。
- 所有在线切换记录 worker PID、weight storage digest、fallback、图状态及 replay。
- 全部新优化开关默认关闭，无新增环境变量；专用 W3 文件也纳入 gate hash、manifest
  和永久 checkpoint bundle 的复制与校验。
- 一次 recapture OOM 后重启；首次恢复发现并修复上述 BF16 冷图问题，第二次启动成功。
  冷加载 head 格式、别名、唯一 registered weight 另有 rank 级检查。
- 未运行仓库完整硬件矩阵；根目录有大量其他未提交工作，广域 UT 的四个 fixture 错误
  来自其 KDA header hash 不匹配。本提交未修改那些文件。

当前地址 `http://192.168.53.187:8001`，模型 `glm53-flash-selective-w3`，
最大模型长度仍为 311040，专家 `native-kernels-v56`、indexer `indexer-kernels-v6`。
最终 health 记录另验证两个 640-token prefill chunk 的 replay 与同一前缀的第二次请求。

```bash
tail -f /home/matteius/experiments/glm-reconstruction-20261006/serve-native-disk-head-nz-recovery-20261007-v2.log
```

`protocols/` 保存精确实验控制脚本，`live-candidates/` 保存完整生效源码及 receipt，
`frozen-sources/` 保存不经重新格式化的构建源码与 provenance。
恢复脚本采用已加载永久 bundle 的校验资源，不能再次注册相同 namespace。
前次结论与 v56 原始发布记录见
[上一轮报告](../native-throughput-20261006/README.md)。
