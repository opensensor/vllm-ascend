# GLM 冷 prefill：批量 scale products 与 Cube result layout

本轮对照为正在运行的 v905，保留已有权重/输入 L1 复用和 FP16 route workspace。
不改量化码、输入精度、block32 scale、FP32 累加顺序和每15行一个 batch。

## 两个默认关闭的构建开关

`--vector-scale-products`：只对 tokens>16 且 count>4 的 prefill batch 启用。
v906 首次实现用 Gather 将当前 block32 activation scale 广播到每行128列，
速度退化。v909 修正为只 Gather 16 个 scalar（未用行索引 clamp 到合法行），
再 Brcb 为每行一个32-byte block；两个64-lane repeated Mul 将原 weight scale
与该 scalar block 广播到每行128列。A4 bulk 不再执行逐行 factor loop。
仍先算 weight scale × activation scale，再乘 INT4 dot，最后按原次序累加。
row indices 放在既有 mask scratch 的24 KiB 起始处（v909只用64bytes）；
Brcb output 借用已有 productFloat 第二半，和第一半的 factors 不重叠；不新增 UB。
小 decode 与已缓存 combined scales 的 count<=4 路径保留原算术。

`--gather-product-matrix`：prefill count>1 时，用一张 batch-specific row/strip
index table 将 Cube NZ INT32 result Gather 到连续临时区，再 Cast 整个矩阵。
原路径逐 N16 strip 做带 stride 的 Cast。A8 的 high limb/bias row 独立提取。
row index table 放在 mask scratch 的16–24 KiB，和 scale indices 不重叠。
要求 prepared layout，且不允许和 weight decode lookup 或其他 readback schedule 同用。
普通 prepared W2/W3 重建只用 mask 的前16 KiB；不跨 batch/replay 复用旧 row layout。

v906：最初的 matrix-broadcast vector scales；v907：仅 matrix gather；v908：组合。
v909：小 scalar Gather + Brcb，保留原 result readback，不启用 matrix gather。
所有候选保持已有 v905 flags、TP4/MTP1、640-token 分段图与311040 context上限。

## 验证与结果

隔离 CPU glm_perf UT：494 passed（含10个分支/ABI兼容 UT）；scoped pre-commit 通过。
沿用此前三个外围测试排除项。构建 flag 类型、required layout、readback/scratch 冲突、
frozen provenance、CLI 转发和 manifest real-multibatch gate 均有 UT。
每候选42个真实权重配对门；独立 arithmetic/replay 30门 + 真实权重12门。
单专家 stage profile 与完整模型冷 TTFT 分别记录，不从单算子推算整机增益。

## 被拒绝的前三个 schedule

31 行及640行的单专家 profile 都退化。640行 A4 的四 stage median 总和：

| 真实权重 | v905 ms | v906 ms | v907 ms | v908 ms |
| --- | ---: | ---: | ---: | ---: |
| W2 | 55.997 | 65.239 | 81.184 | 95.204 |
| W3 | 56.384 | 64.852 | 81.553 | 95.450 |
| W4 | 56.027 | 65.060 | 81.080 | 95.190 |

这是单专家、合成激活、top-k1 的设备 interval，且表中为分 stage median 总和，
不是完整模型 latency。三个候选未装入 serving，服务恢复 v905。
更少 Cast/Muls 调用不代表更少设备工作；全矩阵 Gather 的 layout 成本盖过收益。

v909 也通过42个真实配对门及30+12独立门。31行 A4 的 stage median 总和下降
16.3–17.2%；640行下降21.0–21.7%。A8 收益约4.3–5.8%。serving 使用 A4。
只选择 v909 做完整模型在线比较；三个慢 schedule 没有装入 resident registry。

v909 的在线配对正在运行；baseline v905 和候选保持相同 cold prompt、context、
cache 清理和 graph 设置，先测1280两次及2560一次，再回切两版本各测1280一次。
baseline 同期测 short c1/c4，候选做完整26请求 short/quality/tool suite。
保留完整 SSE、usage、逐请求 metrics/status 和 server log，不只测单个投影。

## Decode 的独立资源

all-purpose v909 冷1280 TTFT 三次为18.723/18.858/18.722 s；同期 v905 为
22.128/21.947/21.880 s。2560 为39.535 vs45.777 s，分别约14.7%/13.6%改善。
但该轮 short c1为8.638 vs9.472，c4总量12.960 vs19.378；拒绝保留all-purpose v909，
finally 已恢复 v905 并 resume。26请求有效，质量 gate 仍有失败；不声称新增量化误差，
也不将这一轮 decode 差异直接归因为某条指令，需要隔离新 schedule 与运行间波动。

新增 PrefillDecodeNative：tokens>16 使用已签收 v909，tokens<=16 使用原 v905
资源。边界仅读静态 shape，输入、路由和 scale 不变，不使用 device item/CPU拷贝。
两个资源 device/dtype/activation bits/prepared layout/route dtype 必须相同。
各自保留 descriptor/scratch；prefill graph release 同时清理两资源的独立 scratch。
选择发生在 graph capture 中；graph replay 使用当时选择的原始 kernel。
status 分别记录 prefill/decode 资源与 capture/Python call counts，计数不是设备 replay数。

第一次独立/在线 split runner 检测到外部大型请求，未执行门或切换；保留busy日志。
没有取消外部请求。split 随后完成42个真实 paired/replay gates；在线比较又遇到
新请求，因此仅在下一空闲点 apply + recapture + resume，没有发送 benchmark请求。

## 最终在线状态

**split 已运行在8001**：`native_v909_prefill_v905_decode`，source digest
`2c7a5712537c039f4dd7a56d4635172662ab34faeaed84ccff8f82ec9ecfc6ed`。
四 rank 的 capture/Python dispatches 均为 prefill84/decode172，资源明确分别为
reconstruction_v909 / reconstruction_v905；该计数不声称是设备 graph replay次数。
API PID2826435与workers2827551/2827949/2828351/2828744不变，weight storage
摘要不变，无重新准备全模型权重；graphs_dirty/native_failed均为false。服务已resume。
后续按candidate/replay条件等待四rank回执，均至少19次640-token graph replay、
native fallback为0、40 MiB FP16 scratch；原始即时回执仍有不同rank的counter滞后，
不声称逐请求counter完全同步。见measurements/v909-mixed-verified-status.json。

实际用户请求在新 split 上继续运行，观察到640-token chunk约9.7–9.9s，之前请求约
11.2–11.4s。上下文和请求不同，这些观察不构成严格配对。混合split的完整冷API/quality/
decode suite尚未运行，不能把all-purpose v909的18.7s数字当作已测mixed结果。
其prefill原语来自已签收v909，decode原语保留原v905；真实42门验证了这个组合。

永久磁盘默认仍为v56；质量仍有既有17/20失败，没有发布为永久默认。
保留四个build的binary/bridge/helper/provenance、各版本源快照、NPUpaired XML、
原始stage samples、all-purpose在线SSE/metrics/server log、busy检查及mixed签收记录。
历史`.py.txt`文件需还原原名并核对hash后才能用于重放。
三个包含SDK原始拼写或opaque request IDs的机器log使用gzip原字节归档；
raw-log-hashes.json给出解压后SHA256，未改写SDK错误文字或request ID。

运行日志不变：
`/home/matteius/experiments/glm-reconstruction-20261006/serve-native-disk-head-nz-recovery-20261007-v2.log`。
下一轮需要在空闲时完成mixed的受控全模型比较，并继续研究更宽row batch与图的动态边界。
