# GLM 冷 prefill：gate/up 输入复用

状态：v904/v905 已编译并通过真实 NPU gates；**v905 已热切换到 8001 并 resume**。
本轮只在已有 v903 的 L1 权重缓存和 FP16 route workspace 上增加输入复用。
永久磁盘默认 v56 不变；不会重新量化、准备或加载模型权重。

## 两个开关

`--share-gate-up-input`：gate 读取 route/source-token 与 activation scale 一次，
up 复用同一 row batch 的元数据。不同专家、输出 tile、batch 和 replay 都重新建立。

`--cache-gate-up-activations`：需要上一开关。gate 在 L1 保留完整 K 的 packed
activation tile，up 直接 LoadData 到 L0A，省掉第二次 GM 读取、UB packing 和 UB→L1。
每 block32 group 固定 1024-byte slot；最大 A1 区为 128 KiB；A2、UB 不增大。
A4 paired rows、A8 高低 limb/bias row、K128/K256 wide 布局均保持原逻辑。
同一 batch 的 gate 写入所有将被 up 使用的 slot；不跨 batch/replay 复用旧输入。
数学、scale rounding、FP32 累加次序及 route workspace ABI 不变。

两个选项默认关闭；宏只传给 gate/up（含 W3 specialization），不会改变 down。
manifest 要求匹配 binary/helper hash、六精度真实 multibatch gate 与 changed-input replay。
v904 为仅元数据，v905 为元数据加完整 packed activation。

## 验证

- 定向 CPU UT：113 passed；隔离 review checkout 的 glm_perf UT：435 passed。
- Scoped pre-commit 在隔离 checkout 全部通过；沿用此前三个外围测试排除项。
- v904、v905 各 42 个真实权重精确配对门，对照当前 v903。
  W2/W3/W4 × A4/A8，tokens 2/15/16/17/30/31/640；含输入/route/code/scale
  变化、重复路由、零权重、all-peer 和 graph replay。
- 每个候选另通过 30 个独立 arithmetic/replay + 12 个真实权重门。
- 31/640 行独立 device-event profile：每层一个真实专家，合成激活、top-k1；
  warmups2、samples5。v905 gate/up stage 降低约 7.7–10.4%，v904 基本无收益。
  这是单专家设备 interval，不是完整模型收益，也不是实测 HBM bandwidth。

## 在线方法

保持原 640-token graph、TP4、MTP1、max-seqs4、max-model-len311040。
通过 resident API 加载已签收的新 bundle，保留 API/worker PID 和 weight-storage digest。
每次冷请求前 pause-wait + clear-cache + resume，输入 token ID 固定为 42，
分别 1280/2560 tokens、生成8tokens、temperature0、seed42。
记录流式首 token 时刻、总 API 时长、usage、graph counters、metrics 与 server log。
1280 在两版本各测三次，并在后段重新切回 v903 检查时间漂移。
这些是合成冷 prompt，不能视为 KiloCode 长会话的完整性能评测。
随后沿用既有 26 请求 short-c1/c4、20 quality、tool suite，不放宽质量门。

## 在线结果

| 冷输入 | v903 TTFT | v905 TTFT | 降低 | 每版本次数 |
| --- | ---: | ---: | ---: | ---: |
| 1280 tokens | 22.871 s | 21.793 s | 4.717% | 3，取中位数 |
| 2560 tokens | 48.022 s | 45.921 s | 4.375% | 1 |

1280 对照三次为 22.914/22.865/22.871 s；候选三次为 21.810/21.793/21.712 s。
所有请求均有完整 SSE terminal/usage，生成8tokens；TTFT 不含其后 decode tail。
这是完整模型的合成冷 prompt 实测，不是单算子推算，也不是长上下文吞吐承诺。

随后 c1 **9.449 tok/s**，c4 总量 **14.191 tok/s**；26/26 API 请求有效，
quality **17/20**，tool 通过。三个失败仍为 instr_reverse、instr_first、code_slice；
实际内容为 `pial`、`Red`、反引号加引号的 `lan`。输出细节存在既有运行间波动；
未放宽评分规则。decode/quality 对照使用前轮 v903 suite，不能作严格同期 decode 增益。
没有将候选发布为永久默认，因为完整质量门仍未通过。

API PID 2826435、workers 2827551/2827949/2828351/2828744 及各 weight storage
摘要不变；没有重新准备全模型权重，变换/backup 计数为0。服务已 resume，
候选为 `native_v905_prefill_graph`，source digest 为
`bc8eb2fc66b8a16e791377044a9338780388dd902384d6bda3ed4c829ff02110`。

最终按候选 digest 和至少2次 replay 等待全 rank 回执，四 rank 均为 2 dispatches /
2 replays、46 graph segments / 45 eager breaks、40 MiB FP16 route scratch，
无 graphs_dirty、native_failed、native fallback。逐请求纯 status RPC 的 counters
有旧回执滞后，原始记录保留；不把那些即时值当作四 rank 同期计数。
以 `measurements/v905-verified-final-status.json` 的条件确认结果为最终状态证据。
640 full-prefill 使用分段图；attention/indexer 动态边界及小 prefill 仍有 eager 部分。

保留 frozen-source、build binary/helper/provenance、NPU XML、原始 profile samples、
在线 SSE/metrics/status/server logs 和 controller。归档 `.py.txt` 是历史源码原字节，
重放时还原 `.py` 名称，再核对 provenance hashes；不是运行时可导入的 package。
原始远端 bundle 保留在 `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/build-v904`
和 `build-v905`。源码开关/构建参数见 frozen-source 和 build-v905 provenance。

运行日志仍为：
`/home/matteius/experiments/glm-reconstruction-20261006/serve-native-disk-head-nz-recovery-20261007-v2.log`。

## 下一步

本轮减少了重复 activation packing，但保留每15行一个 batch、逐 block32 的
Cube readback/cast/scale/barriers 和40 MiB route round trip。
下一轮优先测更宽 row batch 与 readback/layout 调度；必须先重排 UB 生命周期，
不能直接把 M16 常量改为 M32。还需重新采集全 rank 冷 prefill attribution，
区分 native MoE、动态 attention/indexer 与 host graph-break 成本。
