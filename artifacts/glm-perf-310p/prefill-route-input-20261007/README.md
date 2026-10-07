# GLM prefill：一次路由输入打包（2026-10-07）

接续 v912 prefill / v913 decode。本轮把同一专家批次的 INT4 activation rows 和
block32 activation scales 先准备一次，供所有 gate/up N128 输出 tile 共用。
用户自行判读生成质量；验证只包含真实权重数学、graph replay 和 API 性能。

## 实现

- `--route-packed-input` 默认关闭，需 prepared / M32 / paired scale groups / shared
  cached gate/up activation 组合；不兼容参数在创建 build 目录前拒绝。
- 新 `glm_fused_route_input_v1` 在 input quantization 后执行。每个 expert 的每个
  31-row batch 打包两组独立 K32 的 A4 行，按旧 consumer 相同顺序排列；unused rows
  和 lanes 均置零，不改 weight banks、量化 scale、rounding、FP32 accumulation。
- consumer 直接 GM→L1 读取 packed tile，替代每个输出 tile 内重复 zero/copy/packing。
  gate→up 仍复用该 L1 内容。FP32 activation scale 矩阵从原八 lane broadcast 压成
  scalar `[32,128]` 后整块读取，不再逐 token scatter/gather。
- slot index=`first/31 + expert + batch`。单调 expert ends 保证 slots 不相交；
  分配上界只依赖 CPU shape，不读取设备 route counts，不新增 `.item()`/host sync。
- 只对 >16 token 的 A4 启用；A8 和小形状保留原算法，decode 保持 v913。
  使用 A4 原先 unused 的 input-high 参数传递 packed tensor，gate-input-scales 参数
  在该私有分支传递 compact scales；down、route reduction 和对外输出 ABI 不变。
- 640 token 的额外 scratch 跟其他 scratch 一起按现有生命周期复用/释放。
  额外保留的 INT4/FP32 buffers 和一次 producer launch 都纳入 profile/API 测量。
  没有重新加入 FP16 gate/up GM workspace。

## 硬件与 CPU 门

真实 W2/W3/W4 × A4/A8 的 66 个 paired replay 全部逐位一致；包含
2/15/16/17/30/31/32/33/62/63/640 token、changed inputs/routes/packed codes/scales、
duplicates/zero weights/peer suffix/all-peer replay。独立数学/replay 30 + 真实参考 12
病例通过。新增 producer binary 与 frozen helpers 均做 SHA256 校验。

隔离 CPU suite **519 passed**；新的 13 个 UT 覆盖 slot 不重叠、上界、空专家、
不合规配置、launch tensor 身份、A8/decode bypass、shared scratch 重用和清理。
scoped pre-commit 通过。未使用 dummy；硬件与 API 均来自真实模型。

## 640-token 单专家设备 profile

同真实 expert 0、top-k=1，2 warmup / 5 samples；表格为全部阶段的各自 median 相加。
计入新 producer，属于 enqueue/device interval，不等同于纯 Cube 指令耗时或 HBM 流量。

| 精度 | v912 全阶段 (ms) | v914 全阶段 (ms) | 时间减少 |
| --- | ---: | ---: | ---: |
| W2A4 | 34.382 | 30.980 | 9.89% |
| W3A4 | 34.576 | 31.080 | 10.11% |
| W4A4 | 34.167 | 30.795 | 9.87% |

A4 gate/up：约 18.8–19.1 → 15.2–15.3 ms；新 producer 约 0.30 ms。
A8 没有启用 producer，变化约 −0.18% 至 +0.06%，没有宣称 A8 改善。
31-token profile 也保留，但小区间存在明显 enqueue 抖动，不用来推导完整服务收益。

## 在线证据

两次 matched median：**v912 16.335 s → v914 15.499 s**，TTFT 再减少 **5.12%**。
最终驻留 **v914 prefill / v913 decode**，服务 unpaused。全四 rank 至少两次新图 replay，
fallback=0，PID 与 weight storage digest 不变，完整回执在 `validation.json`。
没有据此宣称 decode throughput 改善：decode kernel resource 仍为同一个 v913。
性能使用同一模型、seed 42、1280 个 token ID 42、输出上限 1 token；每次清空 prefix
cache，顺序 v912/v914/v912/v914。load/capture/status 时间不计入 TTFT。
controller 只在 idle 开始，`pause mode=wait` 不取消正在执行的用户请求。

实际 scratch capacity 由 `routed_input_pack` 回执记录，包括 packed/scales shape 和 bytes。
实际四 rank 均为 640 token、top-k=8、72 个 local experts、hidden=4096，预留 238 slots，
packed 29.75 MiB + compact scales 3.72 MiB。这是预留容量，不是实测 GM 流量；
只有 live local batches 写入，peer 和空专家不会消费未写入的 tile。

FULL decode 2/8 和单请求 640 prefill 的 46 segments / 45 eager breaks 保持既有策略。
TP4/MTP1、max len 311040、max sequences 4 不变；mixed prefill/full graph、EP/flashcomm1
和新并发配置不在此轮测试范围。没有发布新磁盘默认 bundle，没有变换模型权重。

## 复现

frozen build 保留 `.bin`、bridge、provenance 和 helpers；历史 `.py.txt` 恢复 `.py`
文件名并校验 hash 才可导入。v914 参数是 v912 的构建参数加 `--route-packed-input`。
新增 header 与 producer 源码在 `frozen-source`；控制协议在 `protocols/*.py.txt`。
原始 SDK log 以 `.log.gz` 保存，解压 hash 位于 `raw-log-hashes.json`。
基准 v912/v913 的 frozen binaries 在上一轮 `prefill-quant-batching-20261007` 中。

API：`http://192.168.53.187:8001/v1/models`。
服务日志：
`/home/matteius/experiments/glm-reconstruction-20261006/serve-native-disk-head-nz-recovery-20261007-v2.log`。
