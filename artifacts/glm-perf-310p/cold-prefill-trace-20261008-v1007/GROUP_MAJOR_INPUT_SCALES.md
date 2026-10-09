# GLM group-major input scales: offline candidate

## US English summary

`--group-major-input-scales` prepares dense routed activation scales once in
`[group, row]` order, so every gate/up output tile can broadcast directly from
contiguous FP32 scalars. It removes the K-loop Gather and its following vector
barrier on the row-accumulator path. Brcb and all weight-scale products, dot
products, scaling, addition, output rounding and routing remain in their
previous order. The optional NZ accumulator copies contiguous row scalars;
it removes the Gather but retains its shared barrier and copy cost.

The producer and consumers switch together at **five expert rows**. At most
four rows retain the old `[row, group]` layout and combined-scale cache.
Inactive dense rows copy row zero, matching the old consumer's clamp. Unused
groups remain zero. This supports A4 bulk prefill with routed raw input scales,
prepared M32 weights and vector scale products. A8 and small-token launches
retain their existing paths. Up recomputes the layout selection even when it
reuses the scale bank already loaded by gate. No host route count or new
CPU/NPU transfer is introduced. Packed weight and activation codes do not change.

The matching producer and every ordinary/specialized gate/up binary receive
one compile define. Down, quantizer and reducer do not. Build provenance records
`input_scale_layout=dense_group_major_sparse_row_major_v1`, hashes both layout
headers and identifies the producer binary. Runtime rejects incompatible
options or layout markers before loading kernels. Admission requires matching
synthetic **and real-weight** W2/W3/W4 A4 graph replays from four to five expert
rows, with changed inputs, routes and weights, followed by all-peer zero output.
These device gates are implemented but **have not run** in this offline phase.

## Work and memory accounting

For hidden width 4,096, intermediate width 2,048 and N128 output tiles, each
dense expert batch has 128 independent K32 scale groups and 16 gate/up tile
pairs. The row-accumulator consumer formerly issued 2 x 16 x 128 = 4,096
scale Gathers. The producer now issues 128 once, leaving **3,968 fewer Gather
calls**. This is a source-level count, not measured NPU cycles or speedup.
Producer work, lane broadcasts and synchronization still have a cost.

The existing packed-code tile and transpose output have disjoint lifetimes.
The producer reuses that allocation, expanding it from 2 KiB to 16 KiB and
adding a 128-byte index vector. With raw scales its declared UB allocation
increases from **18 KiB to 32.125 KiB**, or **14.125 KiB/core**. Consumer UB,
L1, GM scale allocation and GM scale DMA sizes are unchanged: 16 KiB per
batch slot. Descriptor ABI and permanent checkpoint layout are unchanged.
SDK scratch, compiler lowering, device occupancy and runtime peak/reserved
memory still require Ascend validation; this does not qualify larger chunks.

## Offline validation and remaining hardware work

CPU tests compile the actual route-input producer twice and the consumer
broadcast/copy helpers using explicit CPU operator semantics. They compare
raw FP32 bits, including signed zero, infinity and NaN payloads, check exact
packed-code bytes, disjoint slots, one write per scale, untouched peer/hole
slots and declared buffer sizes. Coverage includes 8/16/64/128 groups,
64/640/1,280 tokens, zero through 62 expert rows with 31-row batching,
changed metadata on a second invocation and an all-peer third invocation.
Direct broadcasts and NZ row copies match the legacy helper bit-for-bit.
Tests also enforce paired compile defines, provenance, descriptor stability,
CLI wiring, bad-option rejection and missing real/synthetic boundary gates.
CPU launches and stubs do not emulate ACL graph replay, Cube execution,
Ascend DMA scheduling or device timing.

Run the candidate alone before combining it with compact-down, reducer or
expert-boundary candidates. Build a fresh append-only version from the qualified
v1001 recipe with the additional flag; do not relabel existing binaries.
Then run independent synthetic/loaded-weight arithmetic and changed-route
ACL replay, boundary/all-peer checks, and paired 6,400-token cold requests.
Record native dispatches, per-rank task timing, allocation/reserved peaks and
decode controls. Retain the qualified v1001/v984 control until those pass.

The full offline GLM suite passes **1,921 tests**, with six environment warnings.
Scoped manual pre-commit checks pass; whole-repository checks still fail on
existing archived Python, Markdown and forbidden-import errors outside this
change. Validation logs are saved beside this report.

No SSH, NPU use, server pause/reset, launch or hot swap occurred. No speed gain
or language-quality improvement is claimed. See the current CPU/check results
in [the shared offline handoff](OFFLINE_NEXT.md).

## 中文摘要

新增默认关闭的 `--group-major-input-scales` 离线候选：dense 路由激活 scale
在 producer 一次转为 `[group,row]`，gate/up 各输出 tile 在 K 循环直接 Brcb，
删除反复 Gather 及其后的 vector barrier。权重 scale 乘法、dot、缩放、加法和
输出舍入顺序保留。可选 NZ 累加改为连续行 copy，删除 Gather，但保留 copy
和共同 barrier，不能将其视为同等时延收益。

生产和消费两端以五个专家行为边界共同切换；至多四行保持旧 row-major 布局和
combined-scale cache。dense padding 行复制第零行，符合旧 clamp；未用 group
为零。仅适用于 raw routed A4 bulk、prepared M32 和 vector scale products，
A8 和小 token 路径保留。up 即使复用 gate scale bank，仍在设备上重新确定布局。
无新增主机路由计数或 CPU/NPU 传输，权重及激活 codes、descriptor 和磁盘布局
不变。编译 flag 只给 producer 及普通/W3/W4 gate/up，down/quantizer/reducer
不接收。provenance 记录 `dense_group_major_sparse_row_major_v1` 和 header、
producer 二进制哈希；运行前拒绝选项或 marker 不一致。准入另要求合成及真实
W2/W3/W4 A4 的四行至五行 ACL 重放、输入/路由/权重变化及 all-peer 零输出。
这些设备 gate 已实现，本阶段尚未执行。

GLM 4,096 输入、2,048 intermediate、N128 形状每个 dense 专家批次有 128 个
K32 group、16 对 gate/up tile。消费者原有 4,096 次 scale Gather，生产者新增
一次性 128 次，净减少 3,968 次源码调用。这不是实测周期或加速倍数；Brcb、
生产者重排及同步仍有代价。复用不同生命周期的 code tile 缓冲区，raw producer
声明 UB 从 18 KiB 增至 32.125 KiB，每核增加 14.125 KiB。消费者 UB、L1、GM
scale 分配及 DMA 尺寸不变，每 batch slot 仍为 16 KiB。SDK scratch、编译器
lowering、占用率和完整内存峰值仍需 Ascend 验证，不能据此放大 chunk。

CPU 测试编译实际 producer 及消费者 helper，验证 FP32 位模式（含 signed zero、
infinity、NaN payload）、code 字节、slot 无重叠、scale 单写、peer/hole 未写和
声明 UB 大小。覆盖 8/16/64/128 group，64/640/1,280 token，0–62 专家行的
31 行分批，第二次修改元数据、第三次 all-peer。直接广播及 NZ copy 与旧 helper
位一致。另测试成对编译、provenance、ABI、CLI、坏参数和缺失真实/合成边界 gate。
这些 CPU launch/stub 不模拟 ACL 图、Cube、Ascend DMA 调度或性能。

完整 GLM CPU 测试通过 1,921 项，有六条环境警告。改动文件的 manual pre-commit
检查通过；全仓库检查仍因本次范围外的既有归档 Python、Markdown 和禁止导入
错误失败。验证日志保存在本报告目录。

下一阶段从合格 v1001 recipe 新建 append-only 版本，增加此 flag，先单独跑合成/
真实算术和路由变化 ACL 图、边界及 all-peer，再做配对 6,400-token 冷请求。
记录原生调用、各 rank 时序、allocated/reserved 峰值和 decode 控制，通过之前
保留 v1001/v984 基线。本阶段无 SSH、NPU、pause/reset、launch 或 hot swap；
不宣称速度或语言质量收益。当前 CPU 和检查结果见共同离线 handoff。
