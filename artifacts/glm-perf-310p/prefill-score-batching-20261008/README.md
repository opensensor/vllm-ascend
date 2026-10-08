# GLM 冷预填充行级向量批处理

新增 `GLM_KDA_SCORE_ROW_BATCH`，与已验证的列缓存、FP32 beta 乘积、分数归约、
尾块 W/U 向量化和 gated-key 复用组合。对完整 64 行分数块，保持门值差的 FP16
舍入，随后对整行批量执行缩放、截断、指数和 key 乘法。K×K 与 Q×K 仍分开，
每个通道保留原有两棵 64 元素 FP32 归约树及 +0/p0/p1 顺序。

复用原有 128 KiB 向量 arena 的空闲区：16 KiB 起存 FP16 gate/product，
32 KiB 起存 FP32 product，末端不超过 64 KiB；不可变列缓存从 80 KiB 起。
本次新增 UB 分配为零。仅 SAFE_GATE、FP16、K128、BT64 且完整列缓存启用时
采用新路径，尾块继续使用原有已验证实现。

## 验证

17 个生产 safe-gate 完整算子用例全部通过：每例 12 个输出逐字节一致，
包括递归 carry；输入保持不变，重复输出一致。覆盖 BSND/BNSD、尾块、多块及
不同门值。另两个 nonsafe 父实现用例已非有限值，仅保留诊断，不作为资格认证。

首次编译失败是 tensor 临时切片不能绑定非 const 引用；v2 增加命名别名，
主机语法回归测试覆盖两种归约分支。首次在线旁路测试在设备上下文初始化失败
（507033/E39007），尚未执行候选算子。芯片接近满额保留，可能无空间启动另一
上下文，但日志不能证明精确内存原因。确认服务空闲及进程身份后停止 GLM 树，
候选在释放的 NPU 上通过，随后重载真实权重并恢复八个 native 资源。

直接使用已冻结的 tail-W/U v2 输出参考，父 gated-key reuse v2 已在同 17 个
用例匹配参考，记录参考 SHA；没有重新跑基线。640 token 完整 KDA 为
17.70/17.49 ms（BSND/BNSD），已存档父实现为 26.14/25.84 ms。
这是候选空闲设备测量与存档父时间的比较，不能当作新配对 E2E 加速结论。

## 服务范围

继续使用永久 FP16 scale 磁盘模型、target/draft A4、MTP1、TP4、四槽、
640 token chunk、prefix cache，decode 图大小 2/8。仅替换 KDA OPP vendor。
预填充仍有 attention/indexer eager 边界，不是全融合图。配置 context 为
311040，本批不验证最大长度或扩大到 skill 的 16 槽容量基线。
EP/FlashComm1 和多模态不新增资格认证；图像/视频关闭。未使用 dummy 权重。
文本质量由用户评估。

日志：`/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/score-row-batch-server-20261008.log`。
E2E 测量和最终检查附于下方。参见 [US English](README.en.md)、
[运行说明](RUNBOOK.md) 与 [服务路径审计](SERVING-SCAN.md)。

## E2E 结果

| 输入 token | 存档父版本 TTFT | 新版 TTFT |
| --- | --- | --- |
| 640 | 5.63 s | 5.40 / 5.36 s |
| 1280 | 12.32 s | 11.68 / 11.43 s |
| 6400 | 68.17 s | 65.36 s |

均为冷的 synthetic token-ID completion，每次重置 prefix，只生成一个 token。
父时间为存档，未新增配对基线；新版 640/1280 有重复，6400 只有一次。
6400 比存档快约 2.8 秒。63 token 为 1.26–1.30 秒，639 为 6.17 秒。

C1 生成 9.83/9.62 tok/s；C4 含预填充总吞吐 9.58/10.37/10.84 tok/s，
本次不能证明 decode 加速。计时包含请求完成尾部，另存可见片段时间。
四 rank 图干净，expert fallback 为零；真实 chat smoke HTTP 200，返回连贯的
光合作用说明。仅功能检查，不是文本准确率认证。

## 本地检查

新增 9 项测试通过；GLM performance 1503 项、可运行 GLM W2 339 项，合计
1842 项通过。现有 assembly/grouped-gate-up/kpool-ops 三个文件依赖本地缺少的
torch_npu/upstream vLLM 模块，失败日志保留并在 CPU gate 中排除。默认顶层
conftest 也缺运行时依赖，定向 CPU 测试使用 confcutdir；真实 NPU gate 独立验证。

已在隔离 checkout 运行要求的 `bash format.sh ci`，仓库已有格式问题及自动改动，
无关改动已恢复。owned 文件 scoped hooks 通过。编译及 lint 日志保持原字节压缩，
避免 cache SHA 和原有错误引文被误报为凭据或拼写错误。

Prefix 复用验证：1280 token 冷 11.37 秒，同样输入重复 6.06 秒，实际命中
640 token；末尾 640 token 仍计算。一个真实文本 direct completion 全程
9.03 tok/s，仅计时，不是 chat-template 或文本质量评估。

## 新鲜 serving profile 与后续目标

CANN 覆盖四 rank 的 decode 和冷 1280 输入的两个 640 chunk。末尾 rank0
step 中，按源码确认的九次 launch 顺序归因：stage7 分数整理 34 层合计
417.11 ms，存档父 gated-key 版本为 702.97 ms。完整 KDA task 合计
552.58 ms，step 时长 6151.34 ms，存档为 6515.00 ms。profiling 有开销，
任务合计不能相加当作关键路径，也不是新配对基线。decode 为 210.35 ms，
存档 205.66 ms，本批未证明 decode 提升。

剩余 task 合计：expert gate/up 1661.68 ms，down 849.94 ms，QSA 758.36 ms，
route reduction 133.04 ms，pack 115.76 ms。仍有 AI_CPU：ScatterElements
42 次 8.25 ms，FloorDiv 24 次 1.71 ms，ReduceSum cast 两次 0.36 ms 等。
并未消除全部 AI_CPU。混合 cast 类 Matmul/Cast 为 34.85 ms、266 次，
InplaceCopy/Cast 为 22.69 ms、1376 次，不能直接映射到 grep 行。

下一步重点为重复 expert 重建/layout 和 QSA cache 流量。KDA 批量矩阵也可
研究 repeat 指令代替逐列 Sub/Mul 及两棵归约树；本批未实现或认证这个后续候选。
仍须保持精度/carry gate，并用完整 prompt 时间评估。profile 后服务健康、
未暂停，source switch 保留 worker 和权重身份。四 rank OPP 均采用新 KDA 包。
原始 rank0 CSV、全 rank 归因、profile receipts 和冻结源保留于本目录。
头文件及 binary SHA 见 `measurements/build.json`，Python factory 保持
`33b6d50f3ffc2bf6212a3f968082fc5cf5dc3f85f0ee6da3601f9f08d0b35a49`。
