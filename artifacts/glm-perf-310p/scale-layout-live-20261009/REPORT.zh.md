# GLM 常驻内核布局实验：2026 年 10 月 9 日

本轮在原有四个 310P 工作进程中热加载六个独立候选内核，不重启服务，
不重新加载模型权重。保留解码 v984、131072 上下文上限、1280 token 调度预算、
MTP1 和仅解码 ACL 图配置。测试完成后再依据实测结果选择运行版本。

新增 down scale 布局：密集专家批次在 gate/up 输出阶段整理 scale，
down 阶段直接读取连续列；稀疏批次保留原布局。复用已失效的输入 scale
临时区，不增加张量存储，不修改量化与累加顺序。部分密集 tile 的 scale
写入增加至 512 字节，因此不能仅凭减少 Gather 次数断言端到端加速。

Rank 0 的热加载日志显示完整 prompt token 数、入队时复用位置、当前 chunk、
已调度位置及剩余 token。只读取 CPU 调度元数据，无设备同步。永久 scheduler
补丁在下次启动后还会记录请求入队及命中缓存的 token 数。这里记录的是调度
进度，不是设备完成进度；完成耗时仍以原有 iteration 日志为准。

最终对比额外记录真实 prefill bridge 的 operator namespace、各阶段内核对象的
binary hash 和提交次数，排除准备与验证调用；这些记录不是设备指令计时。
最终图捕获前释放退役基线引用的 scratch，避免验证保留的临时区影响内存。

末尾两个 640 token chunk 是 prefix checkpoint 的要求：共同缓存块为 640，
6400 token prompt 的相同请求回放需要保留末 token 重算，因此必须保存
5760 位置的 KDA state。直接合并末 1280 的执行需要支持中间 recurrent state
checkpoint，不能只删除调度 split。

六个候选均已通过真实 CANN 编译，以及 W2/W3/W4、A4/A8 的模拟输入、
真实 checkpoint、输入和路由变化后的图回放与全远端专家测试。常驻安装
额外使用每个 rank 已加载的 72 专家 bank，与 v1001 逐位比较结果。
服务计时每次清空 prefix cache，使用同一重复 token 输入；这不是自然语言
准确率评估，也不代表大上下文或并发场景下必然获得相同收益。

CPU GLM 测试共 1998 项通过，改动文件的 manual hooks 全部通过。
全仓库格式检查仍有本次改动之外的既有错误；完整日志已保存。
格式检查在隔离 worktree 中进行，未修改其他代理的 Qwen 文件。

最初两次控制脚本在安装候选前失败：名称含不允许的连字符，以及替换完整
资源字典导致 normalization 条目丢失。修正后只替换 MoE 条目，两次失败均
恢复基线。日志首次使用未配置的 logger，现改用服务 logger，已确认输出。

最终服务数据与运行版本见 `serving-summary.json`；构建及独立设备验证见
`build-results.json`、`gate-results.json`。原始控制器保存在
`live-queue-r3.py.txt`。服务与实验日志路径见英文报告。

[US English companion](README.md).

## 服务实测与最终运行版本

独立候选冷请求均值：v1001 51.79 秒；v1020 51.10；v1021 50.76；
v1022 51.30；v1023 51.13；v1024 52.19；v1025 64.85。
最终带提交审计的对比为 v1001 52.277 秒、v1021 50.930 秒，减少 2.576%。
两者在每个 rank 都提交 258 套 MoE 阶段，operator namespace 分别为
`glm_reconstruction_v1001.launch`、`glm_reconstruction_v1021.launch`，
gate/up 与 down binary hash 不同，无 fallback。这证明实际加载及调用了新内核，
但不是数量级加速。

首次严格文本 gate 因输出 `q` 与 `b` 不同恢复了基线；同一 prompt 的基线此前
输出 `L`，因此本轮文本 gate 自身不稳定。随后用 1920 token shadow 请求，
在每个 MoE 层对相同的真实 activation 和 route 分别运行两个内核，覆盖全部
43 个 bank、1280 与 640 两种 chunk。四卡共 344 次逐位比较全部一致。
移除 shadow 后保留 v1021 prefill、v984 decode，图完整，权重和进程未更换。
这不解释基线输出波动，也不代表任意自然语言请求的质量通过。

`serving-summary.json` 保存初始文本 guard 恢复基线的决定；
`shadow-summary.json` 保存之后真实输入对比及最终升级决定。
升级后 health 为 200，服务未暂停，maintenance 已关闭。
最终基线短请求 decode 测得 10.14 token/s；本轮未修改 decode 内核。

新 v1021 四卡 profile 已完成，6400 token 冷请求为 50.909 秒。这是包含
profiler 开销的归因运行，不是新的成对加速结果。全部 576 次 collective
匹配四个 rank。Rank 0 的 gate/up、down、准备和路由归约分别为 15.258、
7.013、1.442、1.361 秒；device stage 的无重叠通信为 11.652 秒，设备空闲
为 0.154 秒。归因数据不能直接相加宣称收益；通信 wait 包含依赖和协议成本，
不等同于全部专家负载失衡。

代表性 W4 gate/up task 为 78.727 ms，其中 vector 56.604、scalar 26.077、
Cube MAC 1.201 ms。各 pipe 有重叠，不能相加或推广成所有 task 的比例。
[完整流式架构提案](STREAMING.zh.md) 覆盖全部数学阶段，明确已有实现、
计划调度、内存边界和验收标准。现有大专家批次已经跨行批次复用 L1 权重。

Profiler 已停止，普通 v1021 已恢复，服务已重新开放；trace 离线导出。
每卡有 38 条不完整内存记录，来自 trace 开始前的分配，不能据此做完整
内存证明。完整原表保留在远端 `critical-tables.tar.gz`，每表 provenance
见 `critical-table-hashes.json`，精简归因与代表性 counters 已保存至本目录。

离线 worksheet 发现混合批次 cliff：1280 总预算扣除一个 MTP decode 的
2 token 后只剩 1278，内部 prefill 对齐降为 640。1296 总预算加 1280 prefill
上限可留出 decode 空间；已知共享 MoE scratch 每卡增加 1.914 MiB，不含
其他模型/runtime 分配。该候选尚未启动，完整内存与启动检查待完成，见
`headroom-analysis.json`。

首次 profile idle 检查遇到用户活跃请求，但 cleanup 不必要地重新应用了
相同候选，已向用户说明。修正后只有取得 maintenance 才执行恢复。
新架构图完全离线生成，不修改当前运行服务。
