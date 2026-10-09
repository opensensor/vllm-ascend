# GLM 完整专家流式架构提案

![完整流式架构](streaming-architecture.png)

[矢量图](streaming-architecture.svg) · [US English companion](STREAMING.md).

目标覆盖全部路由专家和全部 K32 量化组，贯穿 gate/up、SwiGLU、重新量化、
down、路由合并、共享专家与跨卡归约。分块是存储和调度方式，不是只优化
模型的一部分。图中蓝色为已有阶段，黄色为计划中的调度重构，灰色为明确的
内存或模型边界。此图是设计提案；服务仍使用 prefill v1021、decode v984。

现有实现已经使用原生 INT4 Cube 和向量 scale 运算。W4 直接加载预排布代码，
W2/W3 在核内重建 INT4；大专家批次已经跨行批次复用 L1 权重。
SwiGLU 与重新量化已融合在 gate/up 内核中。下一步重点是操作数加载、Cube
结果读回、布局转换、scale 广播及同步的粒度，而不是声称这些张量目前都
在用标量计算。

## 流式执行与边界

输入每 token 量化一次，按稳定专家路由生成 packed 激活与 scale。缓存专家
输出 tile 后，Cube 计算独立 K32 整数点积，向量单元应用每组独立系数，并按
既有顺序更新核内 FP32 累加器。完整 gate/up 结束后，保留既有 FP16 舍入
边界，执行 SwiGLU，直接写出 down 所需 packed A4 与 group-major scale。
Down 覆盖完整中间维度，然后执行稳定路由归约、共享专家相加与 HCCL。

目标为加载 tile j+1、计算 tile j、消费 tile j−1 的重叠流水线。是否可行必须
证明缓冲区独立所有权、事件依赖及统一 310P 核的实际收益。当前 product
复用失效 decode scratch，factor scratch 复用 `mask_`。不能直接增加第二个
活跃缓冲区而忽略这些生命周期或 UB 容量。

Gate/up 到 down 的紧凑 GM 交接明确保留：不同核生成不同中间列，而 down
需要整个中间维度。取消该交接需要重新设计核间所有权、同步及内存预算。
路由输出到 reducer 的交接及跨卡依赖也仍存在。标量地址计算、指令提交和
事件控制无法全部消失；张量数学应交给向量和 Cube 单元。

## 为什么必须分块

31 行、128 列、K4096 的全部 K32 FP32 部分点积需要
`31 × 128 × 128 × 4 = 2,031,616 bytes = 1.9375 MiB`，尚不含操作数、scale
及累加器。名义 UB 只有 256 KiB，本内核还保留 8 KiB SDK scratch。
现有 M32 product 区为 64 KiB，另有 16 KiB FP32 累加器。消费后复用部分
点积空间即可处理全部组，无需把完整部分结果写入 GM。

不能先把不同 K32 scale 的整数点积相加再乘一个统一 scale。批量读回必须
保留独立系数及 FP32 累加顺序。FMA 或更大量化组属于新的数值候选。

## 当前实测依据

新 v1021 四卡 trace 匹配 576 次 collective，每卡包含 258 次 MoE 层/chunk。
Rank 0 的归因时间为 gate/up 15.258 秒、down 7.013 秒、准备 1.442 秒、
路由归约 1.361 秒。Device stage 显示无重叠通信 11.652 秒、设备空闲
0.154 秒。此 trace 不支持把 host launch 间隙当作冷 prefill 的主要瓶颈。

一个代表性 W4 gate/up task 为 78.727 ms，其中 vector 56.604 ms、scalar
26.077 ms、Cube MAC 1.201 ms。各 pipe 有重叠，不能直接相加，也不能把
单个样本推广成全部任务的比例。它说明应该重构完整读回、scale 和控制调度，
而不是期待 INT4 乘法本身带来四倍端到端加速。

证据见 [trace 归因](critical-attribution.json) 与
[代表性 pipe 数据](critical-pipe-samples.json)。通信 wait 包含依赖和协议成本，
不能直接视为全部可回收时间。本提案尚未实现，没有预测加速倍数。

验收要求：先证明 UB/L1/L0 容量与重叠生命周期；覆盖全部数学阶段、稀疏
尾部、全远端专家与变化输入图回放；使用四卡真实 W2/W3/W4 权重对比；
记录真实 binary hash 与调用；最终评估完整冷请求和 decode，而非仅 matmul。
