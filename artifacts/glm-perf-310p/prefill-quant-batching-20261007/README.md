# GLM 融合隐藏量化：四行批次与直接 Gather（2026-10-07）

接续 [31 行专家批次](../prefill-wide-rows-20261007/README.md)。用户明确自行评判
生成质量；本轮只运行真实权重数学/重放门、完整 API 性能测试和 stream 有效性检查，
不运行输出质量 suite，不以生成文本相同作为保留条件。

## 改动

- `--quad-hidden-quant` 默认关闭，仅 gate/up 编译为 16 个独立 block32 组，
  一次处理四个 N128 输出行。尾批仍零填充；scale、rounding 和 clamp 与原路径相同。
  与既有 `--pair-hidden-quant` 冲突时在创建 build 目录前拒绝。
- `--direct-hidden-gather` 默认关闭，把物理输出行直接 Gather 到 quantizer input，
  删除 `outputRow_ → Adds(input, outputRow_, 0)` 的中间复制及一次 barrier。
- input-pack 仍为八组，已有 GM activation/scale ABI 不变。新增 compile-time
  边界检查防止把十六组 header 意外用于 input-pack。
- 四行 quantizer 使用 15,168 B 临时区，小于已有 16 KiB allocation；
  offset 索引增为 4,160 B。31 行版本 gate/up 的总 UB 为 221,568 B，
  down 仍为 233,792 B，没有新增 GM FP16 gate/up hidden buffer。
- manifest 和独立 probe 对新选项要求真实 multibatch gate。
  16 token workspace 边界和 FP32 reduction 保持原语义。

## 成对硬件验证

| 版本 | 改动 | 对照 | 真实成对 replay | 独立数学/replay | 真实参考 gate |
| --- | --- | --- | ---: | ---: | ---: |
| v910 | 31 行批次 | v909 prefill / v905 decode | 66 | 30 | 12 |
| v911 | 四行隐藏量化 | v910 | 66 | 30 | 12 |
| v912 | 四行 + 直接 Gather | v911 | 66 | 30 | 12 |
| v913 | M16/K128 decode + 直接 Gather | v905 | 66 | 30 | 12 |

所有病例通过，共 432 个算术/重放门。覆盖真实 W2/W3/W4、A4/A8，
2/15/16/17/30/31/32/33/62/63/640 token。paired 输出逐位一致，
独立参考使用原误差门；replay 改写输入、routes、codes、scales、零权重和 peer routes。
未运行 dummy，所有权重来自当前 canonical checkpoint。隔离 CPU suite：506 passed；
scoped pre-commit 通过。三项独立实验文件仍按既有范围排除。

## 单专家 640 token 设备区间

各阶段 median 相加，每组合 2 warmup / 5 sample。属于一个真实专家、top-k=1，
不能当作全模型时延或纯 Cube 指令时间；尤其小 decode 区间受 enqueue 抖动影响。

| 精度 | v910 → v911 全 pipeline | v911 → v912 全 pipeline |
| --- | ---: | ---: |
| W2A4 | −3.10% | −0.51% |
| W3A4 | −2.98% | −0.66% |
| W4A4 | −3.08% | −0.62% |
| W2A8 | −1.42% | −0.18% |
| W3A8 | −1.31% | +0.26% |
| W4A8 | −1.79% | −0.03% |

四行量化的 A4 gate/up 时长减少 5.42–5.54%。直接 Gather 是较小收益，
A8 profile 没有一致改善；当前 GLM 服务使用 A4。

## 完整冷 prefill API

同四 worker、权重 storage digest、TP4/MTP1、640 chunk 和 resident segmented graph。
每次清空 prefix cache，输入 1280 个相同 token ID 42，seed=42，输出上限 1 token。
交错顺序 v910/v911/v912/v910/v912，status/RPC 和 graph capture 不计入请求 TTFT。

| Prefill / decode | 1280-token TTFT median | 样本数 |
| --- | ---: | ---: |
| v910 / v905 | 16.725 s | 2 |
| v911 / v905 | 16.372 s | 1 |
| v912 / v905 | 16.318 s | 2 |

v912 比同轮 v910 再减少 **2.44%**。v911 单样本只说明方向，不作稳定吞吐保证。
两种新操作和真实完整服务路径都测量过，不能仅凭微 benchmark 宣称加速。

## Decode 与最终驻留状态

v913 的小形状 device profile 已完成，token=2/8；不同精度方向不一致。
使用四次交错 64-token stream 作纯性能测量，不调用评分函数。
两次 median c1：v905 **8.985 tok/s**，v913 **9.452 tok/s**，同轮增加 **5.20%**。
这是固定 64 token 的有限次数 matched measurement，不是所有上下文的吞吐保证。
没有运行模型输出评分，保留 v913 由数值门及 token rate 决定。
最终驻留 **v912 prefill + v913 decode**，服务 unpaused。全四 rank 确认至少两次
640-token graph replay、fallback=0，PID/weight storage digest 与实验前相同。
最终 source SHA256 和每 rank 回执见 `validation.json`，最大 context 设置未改。

服务保持 FULL decode 2/8 与单请求 640-token prefill 的 46 graph segments / 45
attention/indexer/KDA eager breaks。本轮没有把混合 prefill 或 attention 断点称为 full graph。
max model len 311040、max sequences 4；128k/bs16、EP、flashcomm1 不在本轮范围。
模型为文本模型，不涉及多模态。

## 复现与运行

- 恢复 `frozen-source` 的对应版本源文件，保留唯一 append-only build/version。
  v911 使用原宽行构建参数加 `--quad-hidden-quant`；v912 再加
  `--direct-hidden-gather`。v913 在 v905 参数基础上仅加 direct gather。
- `frozen-builds` 保留实际 `.bin`、bridge `.so`、SHA256 provenance、frozen helpers。
  历史 helper `.py.txt` 须恢复 `.py` 并校验原 hash 再导入。
- `protocols/*.py.txt` 为实际控制脚本；纯性能脚本不执行自动质量判定。
- 原始 SDK 日志压缩为 `.log.gz`，解压 hash 位于 `raw-log-hashes.json`。
- 完整 graph 候选在 measurements 的 candidate `.py.txt`，native bundles 用各自
  real-weight gates manifest 加载，再通过 resident controller 重捕获，不能直接导入到
  正在执行的 graph 中。

API：`http://192.168.53.187:8001/v1/models`。
服务日志：
`/home/matteius/experiments/glm-reconstruction-20261006/serve-native-disk-head-nz-recovery-20261007-v2.log`。

idle 后使用 `/pause?mode=wait&clear_cache=true`，从未中断或 abort 用户请求。
独立 gate/profile 后都恢复运行；最终热换由 API 实测选择，四 rank 的 weight storage
摘要与 PID 保持不变。本轮不重启，不重新准备模型权重，也不改永久磁盘默认 bundle。
