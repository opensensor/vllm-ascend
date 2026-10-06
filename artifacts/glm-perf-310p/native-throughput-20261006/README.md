# GLM 310P：更宽 INT4 Cube、decode 与冷 prefill

延续同一 TP4 热服务。最终服务为 `native_v56_prefill_graph`，四 worker PID
327411/327821/328263/328726 不变；全部权重银行继续使用永久 Cube 布局。
磁盘 `native-layout.json` 已指向 `native-kernels-v56`，权重内容、scale 和索引布局
没有改写，在线权重转换为 0。冻结发布包另通过 12 个独立新进程算术/重放门。

## 实测与发布判断

相同 64 输出 token 的 API suite；每轮 26 请求均有效。质量分母为 20，另有 tool 门。
以下为单轮观察，四并发吞吐受入队顺序、prefill 插入和公平性影响。

| 版本 | c1 tok/s | c4 总 tok/s | 质量 | 判断 |
| --- | ---: | ---: | ---: | --- |
| v45，上一发布版 | 8.091 | 12.510 | 18/20 | 历史参考 |
| v56，K128 + M32 | 9.082 | 13.584 | 18/20 | 已发布，失败项与参考相同 |
| v58，K256 | 9.299 | 14.059 | 17/20 | 不发布 |
| v59，再加双行隐藏量化 | 8.954 | 13.898 | 17/20 | 不发布 |
| v60，合并源码 K128 | 8.796 | 13.373 | 18/20 | 通过发布门，保留候选 |
| v61，逐字段 lookup | 6.069 | 11.444 | 17/20 | 重构更慢 |
| v64，K256/K128 自适应 | 9.412 | 13.805 | 17/20 | 最快 c1，仍为实验 |
| v65，双字节 lookup | 6.203 | 9.399 | 17/20 | 重构更慢 |
| v66，稀疏结果条带复制 | 8.865 | 14.677 | 17/20 | 不发布 |
| v67，最终源码 K128 | 8.759 | 13.865 | 17/20 | 保留原始失败记录 |
| v56 + 原生坐标除法 | 8.844 | 19.100 | 17/20 | 峰值不稳定，不发布 |

v56 发布门保留 `instr_first`、`code_slice` 两项原有格式失败，tool 1/1。
多个后续批次额外失败 `instr_reverse`；没有用 17/20 放宽发布条件。
旧轮次已显示 batching 会影响结果。K256 与 K128 的真实 live MoE 输出各 rank
128 次逐位相同，但不能以此取代整个模型的质量门。尚未证明 c1 达到 10 tok/s。

坐标除法三次 c4 重测为 13.605/13.634/18.980，总量变化伴随公平性变化，
不能把首次 19.100 视为稳定内核收益。原始报告均保留。

最终 v56、无 profiler 的冷输入 1280、输出 1：API 时长 **24.684 s**。
冷检索 chat 输入 **8199 token**，正确返回 `BLUE-ORCHID-7319-8`，
TTFT **165.669 s**，全 rank 确认至少新增 12 次 640-token 图重放。
首次状态回执混入旧计数；第二次确认同时校验 generation 与最低 replay 计数。
原始回执和失败断言日志保留，最终报告只在确认全部门后标记 complete。

上一轮匹配 CANN 测量的冷 1280/输出 1：v41 32.708 s → v45 25.992 s，
降低 20.5%。它们开启 profiler，不能与本轮无 profiler 的 24.684 s 当作同口径配对。

## 代码与数值

- A4 prefill count 大于 8 时用 M32 放置两个独立 block32 点积；
  K128 将四组、K256 将八组点积放在空闲行，减少 Cube 发射和读回。
  K256 变体在三/四行时退到 K128；每组 scale 和 FP32 相加顺序不变。
- W2/W3 软件展开为 INT4；W4 直接读取永久 packed bank。乘法是原生
  INT4×INT4→INT32，scale、SwiGLU 与归并保留 FP32；没有 FP16 权重 GM 往返。
- 当前主源码通过 `--wide-cube-k 128/256`、`--pair-prefill-scale-groups`
  显式实验，默认不启用。`--weight-decode-lut`、`--strided-product-copy`
  保留已测的实验实现，不能默认视为提速开关。
- 两种 lookup 均通过 262144 字节的 W2/W3 精确验证。第二种直接产出 packed
  字节，仍因 Gather 等代价变慢。W3 编译期定宽另通过 30 个独立门及真实
  2/640-token 六精度组合逐位比较，matched stage 时长没有实质改善。
- 310P SDK 的 ShiftRight 为不支持的实现，早期整数移位探针失败，未用于服务。
  v55 L0B 排列失败、v57/v62 编译失败，均未加载。
- profile utility 重用每 stage 两个 event，避免 640-token 多 stage 超过事件限制。
  sample 同步后重用；launch interval 包括 host 间隙，不能当作纯内核时钟。
- resume 使用各 rank 的有效状态确认，拒绝旧准备回执；不重发有副作用的操作。

## Decode trace 与缓存

v64 的 CANN c1/c4 trace、host step 标记、设备 step 边界和源摘要位于
`trace-v64-decode/attribution.json`。c1 rank0、16 step：native expert 1088.8 ms，
matmul TransData 210.3 ms，AI-CPU FloorDiv 68.5 ms；AI-CPU Cast 为 0。
原生 expert 的时长加权 pipe 指标：vec 60.8%、scalar 53.4%、mac 0.95%；
这些 pipe 可重叠，不能相加。Cube 活跃时的 utilization 高，不代表整个任务
主要耗时在 Cube。未报告任务区间不能解释为硬件或 CPU 空闲。

小型 INT64 坐标除法探针覆盖负数、INT64 极值及修改输入后重放；作为独立
AI-Core 操作已集成到实验 pool writer，保持 dtype 与向下取整语义。
仅处理最多 16 个元素，其他形状沿用原实现；没有改变服务默认配置。

相同 2055-token 检索 cold/warm TTFT 为 **38.466/3.848 s**，均回答正确，
metrics 确认命中 **1920 token**。APC 已能工作；640-token 边界、选择性状态
保留、清缓存以及前缀变化仍会影响实际请求，不能保证 KiloCode 同样命中。

## 重放与边界

源 `/srv/ai/src/glm-selective-w3-nz-test-20261004`，venv
`/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python`；需先 source
`/srv/ai/bin/ascend-env.sh`。服务 API `http://127.0.0.1:8001`。
永久 model `/srv/ai/models/GLM-5.3-Flash-selective-W3-native-int4-310p`。
图仍是 resident 策略：46 段、45 个动态状态断点；640 单请求分段图与 FULL
2/8 decode 并存。其他尺寸/混合 prefill 沿用原路径，重启后须重新应用和验证策略。
完整候选在 `graph-qualified-v56-v1/candidate.txt`，原始 v56 CPP 在 `v56-source.cpp.txt`。

隔离 CPU UT 298 项通过；另一次扩大范围遇到既有模块缺失和六个旧接口不匹配，
未将那些用户工作纳入修改。全部模型签收使用真实权重，不以 dummy 替代。

后续优先削减 block32 scale/归并和结果布局开销，以及非专家投影的重复格式转换。
更大 scale 分组需离线校准、重新量化和严格模型门；不能在线把两组 scale 合并。
输入量化已复用且占比很小。EP、flashcomm、bs16、混合 prefill 图与异步 serving
尚未验证；现有 TP4/MTP1/max-seqs4/max-len311040 配置保持原值。
