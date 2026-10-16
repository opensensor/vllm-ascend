# GLM 完整专家流式架构：执行计划摘要

[完整 US English 任务计划](glm-streaming-plan.md) ·
[架构与内存边界](artifacts/glm-perf-310p/scale-layout-live-20261009/STREAMING.zh.md)。

计划已进入执行阶段。T1–T9 已完成，最终 v1111 已热装并通过全部 438 项
native 硬件验证、344 项真实模型 MoE shadow 比较、变化 decode 回放和
prefix/KDA 控制检查。T10/T11 已完成，完整 serving 对比拒绝升级，最终恢复
健康 v1021。所测 c4 generation rate 降低 6.8044%，超过 5% 门槛；基线
language JSON 在完成前达到 192 token 输出上限。
参考版本为 commit `b7d8b8f88`、prefill v1021、decode v984。
完整流程覆盖输入量化与路由、权重复用、gate/up、SwiGLU 与重新量化、紧凑
GM 交接、down、稳定路由合并、共享专家和跨卡归约。

保留混合 W2/W3/W4 永久 checkpoint、FP16 scale marker、K32 独立系数、
FP32 累加顺序、既有 FP16 舍入边界和 packed ABI。初期优化 A4 bulk prefill，
decode v984 保留并作为回归检查；不保证未修改的解码内核会加速。不同核生成
中间列，down 需要完整中间维度，因此本计划明确保留紧凑核间 GM 交接。

## 任务与波次

| 波次 | 任务 | 必要条件 |
| --- | --- | --- |
| 1 | T1 内存与接口证明；T2 冻结基线和工作负载 | 离线或只读，可立即开始 |
| 2 | T3 操作数生产；T4 点积消费；T5 紧凑 epilogue | T1 接口已冻结，三个 worker 文件独立 |
| 3 | T6 内核/构建整合；T7 常驻控制器 | T6 依赖 T1–T5；T7 依赖 T1/T2，可提前 |
| 4 | T8 真实 NPU 内核及变化输入图回放 | T2/T6/T7 完成，设备窗口空闲 |
| 5 | T9 真实模型相同输入 shadow 验证 | T8 完整通过 |
| 6 | T10 冷 prefill、decode、混合请求及归因 | T9 通过，移除 shadow |
| 7 | T11 最终整合、报告与实测升级决定 | 全部证据完成 |

每项任务的精确 `depends_on`、文件所有权和验证命令以英文主计划为准。
最多三个 worker 加 coordinator；所有 NPU 阶段串行，只有 coordinator 可热换
服务。Worker 不能覆盖其他代理的 Qwen 改动，也不要求各自提交 commit。

## 关键验证与升级条件

T1 先证明 UB/L1/L0 容量和重叠生命周期，明确 decode/product scratch、
factor scratch、scale 索引、gate 结果与累加器的所有权。31 行 × 128 列 ×
K4096 的全部 K32 FP32 部分结果需 1.9375 MiB，不能整体物化；必须用有界
tile 覆盖全部计算。缓冲或事件不支持时拒绝该拓扑，不把旧循环标成新流水线。

CPU 验证不能代替真实 NPU 事件/重叠验证。T8 使用真实 W2/W3/W4 bank，
覆盖 A4/A8、稀疏尾部、全远端专家、72 专家、多批次复用、640/1280 chunk
和变化输入图回放。T9 在全部 43 bank/卡比较相同真实 activation；1920 token
两 chunk 的控制请求应有四卡 344 次有限输出的逐位比较，且覆盖所有阶段。

T1 冻结实际 tensor 类型、offset、物理/有效行数、slot 状态和事件所有权；
Cube 指令提交循环由 T6 负责。T2 冻结构建/gate/shadow/trial 的 evidence schema，
避免并行 worker 各自猜测接口。Standalone 可在完整 T8 gate 之前按构建与
contract digest 创建测试实例，但不能据此热装服务；live admission 必须有
完整匹配的 T8 证据。T6 还需验证组合后的真实指令和生命周期顺序。

独立测试前必须计算服务仍已加载时的峰值：原有约 40 GiB/卡 resident 分配、
第二进程/runtime、逐 bank fixture、候选与基线 scratch、输出及验证缓冲。
设备 idle 不会释放原有内存。空间不足时分配前停止并记录 gate 受阻，不能
默默卸载服务或缩小必要覆盖范围。

T10 对主 6400 token 冷请求执行至少三个 ABBA 周期，每版本六次测量。
还检查更长请求、已有上下文深度、prefix 命中、自然语言/tool、c1/c4 decode
和混合请求。两版本先做相同未计时初始化/图 warmup，再清 prefix 测冷请求，
每个 ABBA 周期使用相同 token IDs。TTFT 优先使用请求对应的服务端首生成
token 时间；不可用则明确标注首 token-bearing SSE 延迟，首可见文本另记。
目标为主冷 TTFT 降低至少 10%，不是预测或硬性门槛；更小收益仍需超过
`max(2%, 2 × repeat_spread)`，其中 spread 为两版本各自 `(p75-p25)/median`
的最大值。此规则判断重复性，不代表统计显著性证明。还需正确性通过，且无
可重复超过 5% 的 decode/混合服务退化；失败时保留 v1021。计时与归因分开。

操作前后检查 idle 和 maintenance 所有权。用户有活跃请求则退出，不热换、
不 recapture、不清缓存。Reset/apply RPC 前先记录 mutation intent，超时即使
未返回 receipt 也可能已修改部分 rank。必须确认前一 collective 已完成，才能
对齐四卡状态和恢复；无法有界确认时保留己方 maintenance/paused 状态、报告
恢复失败并停止，不能发第二个 collective 或恢复部分确认的服务。失败与恢复
记录必须保存。源代码或 binary hash 改动使下游 gate 失效，需重新验证。

后续任务在设备不可用时继续离线工作，不把 CPU 或部分诊断当作硬件通过。
历史 T8–T11 已完成；其结论为拒绝 v1111 升级并恢复 v1021。
最终英文和中文报告必须一致，区分离线设计、CPU 验证、真实硬件验证和服务
实测，写明最终运行版本、配置、日志、hash、内存变化及已知限制。

独立 plan-only reviewer 已检查依赖、文件所有权、内存和恢复流程，确认最多
三个 worker 加 coordinator 的波次可执行；上述接口、证据、内存、恢复及性能
规则包含其六项修改建议。最终 v1111 硬件与实际输入证据见英文主计划。
三个 ABBA 周期（每版本六次）已完成：6400 token 冷 TTFT 中位数从基线
50.9959 秒降至候选 49.6796 秒，降低 2.5811%；其他长上下文、decode、混合
请求和质量控制已完成，按冻结规则拒绝升级。独立四 rank 归因 trace 也已完成。
最终 CPU suite 为 3951 通过、48 跳过；scoped hooks 与 git diff --check 通过。
最终四 rank generation 为 3f16017ebb4a49eab6469e2ad5ab1672，graph 正常，
maintenance 关闭，health 200，完整八 token serving smoke 通过。
Scalar 时间从初始流式版本显著下降，
但仍高于基线，接近零及 10% 冷 prefill 改进目标均未达到。

## 后续 scalar 优化状态

T12、T13 已完成各自限定范围，T14 正在整合。它们不改变历史 T1–T11
结论：v1111 的 c4 generation rate 退化约 6.8%，因此没有升级，已恢复
健康 v1021。新的候选仍需独立 native、真实输入和 serving 验证。

- **T12：Completed**。每 pair 的 Cast mask 设置从十六次降至两次，CPU
  精确检查、安装版本 dav-2002 编译及五项选定 native case 通过；不是完整
  qualification。单个 W4/A4/31 行诊断中 gate/up 为 50.432 对 52.298 ms，
  down 为 22.612 对 23.808 ms（参考 v1021）。Scalar 归因为
  31.465/15.495 对 16.878/7.437 ms，与此前 v1111 基本不变。没有 live
  qualification 或热换。[英文](artifacts/glm-perf-310p/streaming-upgrade/T12/README.md) ·
  [中文](artifacts/glm-perf-310p/streaming-upgrade/T12/REPORT.zh.md)。
- **T13：Completed**。公开 Brcb/Gather helper 完成 94 项 CPU 测试及
  ASan/UBSan，保护完整 route cache 并保持任意 FP32 位模式。仅审核 SDK
  源码支持，尚无 T13 SDK 编译、NPU 执行或 live 集成。
  [报告](artifacts/glm-perf-310p/streaming-upgrade/T13/REPORT.zh.md)。
- **T14：In Progress**。版本 2 内存合同记录 expert ends
  [24704,27008)、扩展 Gather 索引 [27008,28032)、固定 group-major 源
  [28032,28160)，全部阶段保持存活，mask 仍为 32768 字节。冻结 v1 证明及
  真实 v1111 admission 保持有效。当前 CPU suite 为 4506 通过、48 跳过、
  6 个警告，耗时 74.02 秒。独立 NPU helper 的 186 个行数/步长/重放用例
  对全部测试位模式逐位通过；这证明尺度复制，不证明非有限专家算术。
  初始 v1130 的入口长度检查未纳入新增配置尾部而返回、未写输出，已被
  拒绝；这不是 Gather 数学错误。
  v1131 修正该检查后，完整 438 项 native 矩阵通过。Standalone gate/up
  为 47.608 对 52.381 ms，down 为 21.796 对 23.858 ms；scalar 归因为
  31.155/14.266 对 16.875/7.434 ms（参考 v1021）。Pipe 计数相互重叠，
  不能相加解释为 E2E 可回收时间。目前尚无完整模型 E2E 收益或 live
  promotion 结论；真实输入 shadow 与 serving 资格仍由 coordinator 验证。
  [T14 报告](artifacts/glm-perf-310p/streaming-upgrade/T14/REPORT.zh.md)。

### T15：离线探查 SDK tensor descriptor 的 scalar 开销

- **depends_on**：[T13]
- **agent_type**：worker
- **ownership**：新增 `glm_streaming_products_pointer.h`、专用 CPU stubs、
  checker/tests 和 `T15/` 英中报告；不修改共享内核。
- **description**：准备锁定 CANN 9.1/dav-2002 的编译器实验。通过公开
  tensor API 一次取得 UB base pointer，再用编译器 vector builtin 执行
  Cast 和数学操作，去除逐 strip tensor slicing。保留两个 slot、独立
  K32 group 的顺序、精确 FP32 运算和 mask。此 builtin 路径不是可移植
  的公开 SDK API 实现；不引入私有 SDK 头文件，记录 SDK hash 和必要
  编译目标。不能把该目标上的源码支持推断为其他 CANN/芯片兼容性。
- **acceptance_criteria**：实际 helper CPU 测试覆盖每个 count/layout、
  正负零、抵消、mask 效果及保留区。CPU 结果不能说明 native 位语义、
  设备事件合法性、serving 收益或升级资格。后续 SDK/NPU 验证归 root。
- **validation**：CPU 指令/位模式/canary/生命周期测试、sanitizer 和只读
  安装版本 SDK 核查；后续不可变 hardware probe 归 coordinator。
- **status**：Completed
- **log**：离线 pointer consumer 完成 187 项 CPU 测试、30 项 ASan/UBSan
  边界检查及七项负向 fault。七个 UB base 仅查询一次，不使用 descriptor
  slicing；保留十六次 Cast、六次乘法、两次有序加法、十次 vector fence
  及全 mask 退出。SDK/compiler 调用点哈希已固定于 T15 英中报告。尚无
  SDK 编译、native 验证或性能结论；root 在当前 E2E 测量结束后决定硬件
  验证。[报告](artifacts/glm-perf-310p/streaming-upgrade/T15/REPORT.zh.md)。
- **files edited/created**：`glm_streaming_products_pointer.h`、专用 CPU
  stubs/checker/tests 与 `T15/` 报告；精确文件清单见英文主计划和报告。

### T16：审计下一候选的 decode 路由元数据

- **depends_on**：[T11]
- **agent_type**：worker
- **ownership**：仅 `T16/` 英中报告，不修改共享源码。
- **description**：核实实际 v984 decode bundle 与调用路径，审计输出 tile
  扫描中重复的 expert-end GM 标量读取。证明 M16 保留 scratch 生命周期
  与一次 launch-prologue DMA cache 的可行性，记录 sparse/A8/W2/W3 和
  图重放风险以及后续 profiling gate。
- **acceptance_criteria**：有冻结源码支撑的 wiring/offset 证明和限定实验
  方案；明确区分推测的 scalar 开销与实际测量归因。
- **validation**：冻结源码/provenance 审查与既有 CPU 测试；root 保留
  NPU 访问，任务不编译、不修改运行时、不宣称加速。
- **status**：Completed
- **log**：已核实实际 v984 的 M16/wideK128/FP16-scale 冻结源码及旧
  binary wiring，其中没有后续 expert-end cache。H4096/I2048/E72 的
  gate/up 与 down projection pair 从源码可推导 3456 次 GM end 读取，
  这是循环计数而非硬件 traffic/counter 或成本测量。Prepared/no-LUT
  分支允许在既有 mask [24640,26944) 保留完整 288×INT64 ends，UB 不增长。
  现有 bulk/vector-scale cache flag 不能直接复用。八项既有 admission/CLI
  CPU 测试通过；编译用例按只读审计范围排除。这不验证新候选或性能。
  [报告](artifacts/glm-perf-310p/streaming-upgrade/T16/REPORT.zh.md)。
- **files edited/created**：`T16/` 英中报告及只读 provenance 记录。

### T17：离线实现 decode expert-end cache helper

- **depends_on**：[T16]
- **agent_type**：worker
- **ownership**：新增 `glm_decode_end_cache.h`、专用 CPU stubs/checker/tests
  和 `T17/` 英中报告；不改共享内核或 builder。
- **description**：使用公开 SDK API 的 M16 helper，在每次 launch 开始
  加载一次对齐的 INT64 expert ends，尾部最多三项安全 GM 标量读取，
  后续 lookup 从保留 UB 读取。保留元数据位模式和全部 mask 邻域，不新增
  UB/HBM 分配。每次图重放重新加载，不依赖旧 cache 或 Python 计数。
- **acceptance_criteria**：覆盖每个 1–288 专家长度、变更输入 replay、尾部、
  单调/空专家/全远端边界；GM DMA 不越界，没有新分配。仍保留 UB scalar
  读取、扫描与分支，不能称为消除全部 scalar 工作。
- **validation**：实际 helper CPU 语义、canary 和 sanitizer；后续安装
  SDK 编译、native parity、decode profiling 由 root 负责。
- **status**：Completed
- **log**：离线公共SDK缓存辅助路径通过872个CPU测试（全部1..288长度、重放、7个故障及sanitizer）。无新分配；每次launch刷新全部有效INT64边界。共享源码/builder保持不变。SDK编译、原生一致性、分析及在线测试等待T14结束。见T17双语报告。
- **files edited/created**：实现与验证完成后记录，以英文主计划为准。

### T18：补充受响应预算限制的质量控制

- **depends_on**：[T14]
- **agent_type**：worker；硬件由 coordinator 执行
- **ownership**：新增 quality-repair 控制器、专用 CPU 测试及 T18 英中报告。
- **description**：保持原提示与性能规则，为两种版本的语言和工具控制
  预先声明 768-token 响应上限；保留原 192-token 失败结果。
- **acceptance_criteria**：全部原生、shadow 和非质量门槛必须通过；仅允许
  原记录证明为预算截断的补充检查，不得绕过性能回退门槛。
- **validation**：36 项专用测试和 159 项相关测试通过；真实补充检查等待
  父试验结束及非质量准入。根代理复核并重新执行专用测试。
- **status**：In Progress
- **log**：离线实现完成，未执行硬件补充、未推广候选。
- **files edited/created**：详见 T18 英中报告与英文主计划。

### T19：集成 pointer consumer 并验证标量削减

- **depends_on**：[T14, T15]
- **agent_type**：worker 集成；coordinator 硬件
- **ownership**：共享 consumer include/call、调用点及来源完整性测试、T19 报告。
- **description**：替换 v1131 consumer，保留 Gather、独立 K32 算术、mask、
  scratch、事件及 ABI。T14 在线测量结束之前不运行编译器或 NPU。
- **acceptance_criteria**：冻结来源包含实际 pointer header；SDK 编译、
  原生一致性及新模型 shadow 先于在线性能测量。
- **validation**：1070 项 CPU 检查通过、48 项既有不适用组合跳过；
  实际调用点/signature 的 ASan/UBSan 与 lint 通过。硬件验证尚未执行。
- **status**：Completed
- **log**：SDK编译、完整438项原生用例及两组各344项真实输入shadow通过。
  部分cold6400中位数51.144→41.594秒。用户要求编码使用，九个主样本后
  guarded停止benchmark并释放维护，再安装1140到四个rank。服务健康、
  未暂停，shadow/profiler关闭。完整性能认证未完成，不声明benchmark推广。
- **files edited/created**：详见 T19 英中报告与英文主计划。

### T20：审计 RTC 优化选项与标量指令生成

- **depends_on**：[T11]
- **agent_type**：explorer
- **ownership**：仅 T20 英中报告及 evidence.json。
- **description**：只读核查实际 RTC argv、SDK 及冻结二进制；区分 host
  O2 与 device 优化，检查缺失编译选项是否值得独立实验。
- **acceptance_criteria**：记录 SDK 原始证据；不得从选项或二进制大小
  推断加速或标量归零。
- **validation**：冻结 ELF 哈希核实。RTC 默认优化等级未确认；SDK 独立
  Release 路径使用 O3。安装的反汇编器无法解码设备指令。
- **status**：Completed
- **log**：只读审计完成；未编译、未使用 NPU、未改运行时。
- **files edited/created**：T20/README.md、REPORT.zh.md、evidence.json。

### T21：分析并发 decode 控制回退

- **depends_on**：[T14]
- **agent_type**：explorer
- **ownership**：仅 T21 英中报告。
- **description**：核查原始 c4 样本、静态 dispatch、调度及推测解码日志；
  decode v984 未改，需区分实证与可能的 admission、调度、温度影响。
- **acceptance_criteria**：保留现有 5% 回退门槛；不将相关性作为内核归因。
- **validation**：只读源代码、日志及完整试验记录；不增加在线请求或测量。
- **status**：Completed
- **log**：完整v1131的c4中位数5.158261→4.694975t/s，回退8.98145%。
  前两组引擎迭代更慢、接受率更低，第三组恢复。根因未隔离，未绕过门槛。
- **files edited/created**：完成后记录，以英文主计划为准。

### T22：离线集成独立 decode boundary-cache 候选

- **depends_on**：[T16, T17]
- **agent_type**：worker
- **ownership**：新独立候选builder、专用CPU测试/stubs及T22英中报告。
- **description**：冻结实际v984源码/argv，仅改变expert-end读取；每次launch
  初始化缓存，保持M16数学、ABI、原flags与FP16scale契约。
- **acceptance_criteria**：严格来源/替换/别名/重放证明；默认离线，不修改
  共享源码，不因缓存而启用vector-scale产品路径。
- **validation**：实际变换Run的2296项CPU测试，加helper共3168项通过，
  包括sanitizer、Ruff、格式与来源哈希验证。
- **status**：Completed
- **log**：v1150冻结输入准备完毕；未SDK编译、NPU执行、改服务器或保留
  版本号，不声明原生认证/加速。用户编码期间保持离线。
- **files edited/created**：详见T22英中报告及英文主计划。
