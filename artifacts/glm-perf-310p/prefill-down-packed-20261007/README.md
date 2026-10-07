# GLM prefill：gate/up 直接生成 down Cube 行布局

本轮接续 v914 prefill / v913 decode。用户自行判读生成质量；实验只检查真实权重数学、
重放安全和性能，不运行文本质量评分。

## 实现

- `--route-packed-down` 默认关闭，要求 `--route-packed-input` 的 prepared / M32 /
  paired / shared cached gate/up 组合。不兼容配置在创建 build 目录前拒绝。
- gate/up 在已有 hidden quantization 内直接写 paired K32 × 31-row batch 布局；
  down 直接 GM→L1 读取 2048-byte pair，替代每个输出 tile 的逐行 zero/copy/packing。
  没有新 producer launch，A4 bulk 仍为既有五阶段 pipeline。
- 每个 gate/up N128 输出 tile 独占两对 group。每次 launch 先清零两对完整 buffer，
  再写有效行；空专家不消费 buffer。两组行按 `[0,count)` / `[count,2count)` 排列，
  >16 行时仍使用两块 M32 Cube。31 行保留原 padding/scale/FP32 累加顺序。
- slot=`first/31 + expert + batch`，容量只依赖 CPU shapes，不读取设备 route counts。
  640-token shared scratch key 增加 expert count，避免不同专家数重用不足容量的 buffer。
- down activation scales 保持原 FP32 八 lane broadcast；A8 和 <=16-token 分支保留旧布局。
  decode 仍使用独立 v913；weight banks、scale precision、rounding、route reduction 均不变。
- 640/top-k8/72 local experts/2048 intermediate 的 low bank 是 `[238,32,2048]`，
  **14.875 MiB**，旧 row-major 是 10 MiB；净增加 **4.875 MiB/rank**。
  加速来自更少的小 DMA/packing 指令，不宣称实测 HBM 字节减少；padding 和清零都会增加
  一部分 traffic。所有这些代价包含在下列 profile/API 测量中。

## 验证

66 paired W2/W3/W4 × A4/A8 真实权重 replay 逐位一致；包含 2/15/16/17/30/31/32/33/
62/63/640 token、改变 activations/routes/packed weights/scales、重复 route、零权重、
peer suffix 和全 peer 输出。独立数学/replay 30 + 真实参考 12 病例也通过。

隔离 CPU suite **530 passed**。新增检查覆盖不同专家数的 scratch 容量、partial pair
producer/consumer 布局、zero tail、A8/decode bypass、scale/high bank ABI、不合规 build
和 provenance；scoped pre-commit 在干净 HEAD 隔离目录通过。未使用 dummy 权重。

## 单真实专家 640-token 设备 profile

同 checkpoint expert 0，top-k1，2 warmup / 5 samples；各阶段 median 相加，计入 input
quant、route input producer、gate/up、down 和 stable reducer。属于 device launch intervals，
不等同于纯 Cube 指令时间或完整服务器性能。

| 精度 | v914 全阶段 ms | v916 全阶段 ms | 时间减少 |
| --- | ---: | ---: | ---: |
| W2A4 | 30.991 | 27.614 | 10.90% |
| W3A4 | 31.155 | 28.019 | 10.07% |
| W4A4 | 30.838 | 27.570 | 10.60% |

A4 down 约 11.5–11.6 → 8.1–8.3 ms，减少约 29%；gate/up 增加约 0.02–0.12 ms，
已计入总量。A8 不启用新布局，实测总量增加约 0.44–0.83%，不宣称 A8 加速。
31-token profile 也保留，短区间存在 enqueue 抖动，不从它推导完整服务收益。

## 在线短 prompt

两次 matched median：**v914 15.481 s → v916 14.734 s**，冷 prefill TTFT 再减少 **4.82%**。
同一模型、seed42、1280 个 token ID42、输出上限1；每次清空 prefix cache，顺序
v914/v916/v914/v916。load/capture/status 时间不计入 TTFT；controller 遇到外部请求
等待完成，`pause mode=wait` 不取消用户请求。最终四 rank graph replay>=2，fallback=0，
PID 与 weight storage digest 不变，服务 unpaused。decode 仍为独立 v913，没有宣称新的
生成 tok/s 改善。

另一次 matched 7680-token 长 cold prompt：**102.751 s → 97.980 s**，单次观察减少
**4.64%**。7680 是十二个完整640 chunk；每 build 仅一个样本，不解释为稳定统计 median。
最后保留 v916 / v913，四 rank 新 prefill graph replay>=12，fallback=0，
`native_failed=false`、`graphs_dirty=false`、unpaused；完整回执在 `validation.json`。

首次 API controller 因外部请求未开始 timing；下一次 controller 在一条 v914 样本后遇到
过期 version key，保持服务 v914/healthy。失败记录单独归档，不混入完成后的两组 median。
正确协议修复 key 并加入 idle wait；不会把控制器错误解释为 kernel/质量失败。

## 复现与边界

v916 构建参数为 v914 加 `--route-packed-down`。v915 是尚未运行硬件的中间构建；
v916 增加容量敏感的 scratch key 后才执行完整验证，没有把 v915 加载入服务。
frozen `.bin`/bridge/provenance/helpers 与控制协议在本目录；历史 `.py.txt` 恢复文件名并
校验 hash 才能导入。SDK 原始日志以 `.log.gz` 保存，解压 SHA256 在 raw-log-hashes.json。

TP4/MTP1、max len 311040、max sequences4、batch tokens640 与永久磁盘模型保持原配置。
FULL decode2/8、单请求640 prefill 46 segments/45 eager breaks 保持既有策略；mixed/full
prefill graph、EP/flashcomm1、并发/新 context 配置未在本轮重新验证。

API：`http://192.168.53.187:8001/v1/models`。
服务日志：
`/home/matteius/experiments/glm-reconstruction-20261006/serve-native-disk-head-nz-recovery-20261007-v2.log`。
