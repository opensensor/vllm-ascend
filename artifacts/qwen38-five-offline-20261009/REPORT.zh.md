# Qwen 五项离线优化候选

## 中文摘要

完成分页融合 QSA、解码感知的预填充分块、融合 GDN WY、有效本地 MoE
路由准备及 MTP0/1/2 对比工具。默认配置不变，没有启动或修改服务器，
没有提交 NPU 工作负载。三个原生内核和版本化桥接库已通过 CANN 9.1
主机编译；设备执行、事件依赖、质量、性能和温度仍未验证。

工作从 2026 年 10 月 8 日延续到 9 日。保留共享工作区的其他未提交修改。

## 五项实现与限制

| 项目 | 实现 | 待验证或限制 |
| --- | --- | --- |
| QSA | 显式 `paged_native` 使用现有片上收集与注意力融合内核；不创建无用的并行收集流。 | 没有新增 QSA 内核。需要与批量收集路径进行多请求、长上下文及图像验证；数值舍入可能不同。 |
| 预填充 | 新调度器继承有界 Mamba 前缀策略；优先已有解码请求，限制混合步骤总预算，依据主机步骤时间调整 128 对齐的预算，并恢复配置和 FCFS 顺序。 | 500 毫秒是估计目标，不是硬期限。纯预填充和待处理图像编码保留原预算，图像预填充仍可能阻塞解码。 |
| WY | 原生 FP32 前向代入融合衰减、三角求解、U/W 运算及最终 FP16 写回；U/W 中间结果保留在片上。 | Q/K 布局、分组 Gram 和累积门仍用 torch。求和顺序改变；需要设备 H/O、长序列状态和真实模型质量验证。 |
| MoE | 根据设备端本地路由计数，只收集有效前缀的四个压缩操作数，只对本地行执行原有 SwiGLU 量化器。 | 分配容量不变，投影仍清零无效输出，未融合投影与最终归并。必须显式使用原生 INT4 和 `cann_swiglu_pack`，先与相同精度模式对比。 |
| MTP | 提供 0/1/2 草稿深度配置、持续并发采集和离线匹配比较；推荐须满足三次重复、质量及图像凭据、完整温度记录和每个并发配置至少 600 秒。 | 不自动启动或切换服务。每种深度需要重新规划缓存和捕获图。尚未选定最佳深度。 |

WY 每核使用 83,456 字节逻辑 UB；保留 FP32 循环状态。它减少外部中间张量，
但增加向量求解和片上访问，可能比原有分块求逆更慢。
25,600 路由、隐藏宽度 2,560 时，四个完整操作数收集写入 93.75 MiB
逻辑负载；25% 本地路由时新路径写入 23.4375 MiB。
这不是 DDR 测量、峰值分配或温度预测。

## 离线验证

针对性 CPU 测试 **360 通过，1 项可选 `msmodelslim` 跳过**。
覆盖真实原生向量函数的 CPU 模拟、4/12 和 16/48 头布局、分块 FP32
状态传播、填充、本地路由字节一致性及调度恢复。
CPU 模拟的事件为空操作，不能验证设备同步。调度测试也不能替代完整引擎验证。

三个原生内核和桥接库在独立目录完成主机编译，最终源码指纹匹配
`host-build-provenance.json`，资源名为 `qwen_prefill_v2`。
没有加载或执行这些二进制。针对性格式和 lint 结果保存；全仓库
`format.sh ci` 仍有基线失败，压缩日志保留，未引入无关格式修改。

未进行服务启动、真实权重推理、图像、性能、容量扩展或持续温度验证。
保留现有图像支持，但每个候选仍须重新通过图像门槛。
ACLGraph、EP 和 MTP 组合待验证；FlashComm1 不在本次修改范围。
未新增环境变量或模型运行器行为。

## 后续验证

获得 NPU 授权后按 `RUNBOOK.zh.md` 与 `RUNBOOK.en.md` 分别测试；
`queued-profiles.json` 明确禁止自动启动。保持相同检查点、缓存、激活精度、
请求内容和输出长度，区分冷前缀与缓存命中，再组合通过的候选。
保留 94°C 暂停、85°C 恢复及 96°C 硬停止。
只有实际外部能源测量才报告能耗，不从温度推断冗余传输。

## US English summary

All five offline candidates and comparison tools are implemented. Defaults and
image support are preserved. CPU validation passed 360 checks with one optional
skip; host CANN compilation passed for three kernels and a versioned bridge.
No server or NPU workload was started. Device correctness, real-model quality,
performance and sustained thermals remain pending.
