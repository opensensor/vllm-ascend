# GLM 专家 offset 预计算与 QSA 共享缓存复用

已实现并装入真实 GLM 服务的候选：decode v984 / prefill v985 在 CPU 上准备
不变 offset 表；QSA v991 在 K/V 地址相同时复用 L1 tile，并批处理 NZ block 的 accumulator/output
向量操作。精度、数学次序、
永久 packed 权重及 `fp16_storage_fp32_compute_v1` FP16 scale 格式保持一致。

## 实现与验证

- `--prepared-offset-tables` 用对齐 DMA 替代每个专家 task 的标量 UB 表构建。
  descriptor 按 geometry 缓存，前八个 INT64 字段和 launch 参数数目不变。
  CPU 测试编译原始表生成函数，逐字节比较多个 K 和调度；不增加 UB/L1。
- QSA staging 校验 parent SHA 和 DMA 锚点。AI core 比较 K/V 原始地址；
  相同则省去第二份 value gather 和第二次 UB→L1 拷贝，不同则保留原始路径。
  本轮保留 UB 分配，不声称节约 KV 存储。线上每个 rank 审计出 12 对共享缓存。
- 独立 QSA parent/candidate 使用相同私有 wrapper；ACLRTC 需要显式读取
  原始 14 INT64 tiling 和 310P `KERNEL_TYPE_AICORE` 标注。30 个完整案例
  与安装中的 operator 及 compiled parent 位精确一致，每例三次修改输入、
  缓存和 metadata 的 graph replay；包含共享/独立缓存和 FP32 独立参考。
  四个真实 worker 各通过 18 个 admission 案例，之后重捕获并恢复服务。
- v984、v985 各在 device 0 通过 30 个独立数学案例、12 个真实权重案例、
  60 个边界配对案例及三次修改 replay；四 rank 各通过 12 个有界 admission。
  覆盖 W2/W3/W4 与 A4/A8，未采用 dummy-only 验证。

## 性能与边界

专家表优化尚未证明 prefill 提速。冷 640 / 1280 / 6400 token 分别约
5.36 / 11.39 / 65.13 秒，C1 synthetic-token generation 为 9.53–10.05 tok/s。
共享 K/V 复用单独（v989）没有证明端到端提速。相同 wrapper 两次 parent 冷测试
1280 / 6400 token 平均 11.433 / 64.644 秒；shared-only 为 11.508 / 64.722 秒。

更强的 v991 批处理 FP32 accumulator rescale/add 及最终 scale/cast。乘法和加法
仍分别执行，每 lane 次序相同，最终 cast 仍为 FP16；512 维时每个 head/tile
省去原有 32 次独立小循环、命令及 barrier。独立编译选项为 `--vector-output`
和 `--vector-accumulate`，不增加 UB/模型内存；flag-off 代码与原始一致。
v990（output）与 v991（output + accumulator）均通过 30 个完整位精确案例及
每个真实 rank 的 18 个 admission。v991 dense 640-row fixture 从 12.264 降至
4.799 ms；sparse 从 97.809 降至 28.205 ms，latency 少 71.2%。这是单个 attention
kernel 测量，不能称整个模型快 3.5 倍。

首次 v991 线上冷 1280 token 为 10.95–11.05 秒、6400 为 57.70 秒；对照为
11.433 / 64.644 秒，分别减少约 3–4% / 10.7%。额外重测 1280 / 6400 为 10.75 / 57.93 秒；
两次长冷 latency 少 10.4–10.7%，记录在 `qsa-v991-repeat-serving.json`。短 640 token 仍约 5.42 秒，未证明提速。
最新 C1/C4 保存在 `qsa-v991-final-serving.json`，C1 为 9.77–10.03 tok/s，C4 全请求 aggregate 为 10.28 tok/s；
本轮没有证明持续 decode 提速。

真实 HTTP 文本响应只证明功能可用；synthetic token-ID 是计时 fixture，
不是质量评估。未进行完整质量评测。MTP 接受率会影响 decode。

四张 310P，TP4/EP、FlashComm1、MTP1、prefix cache、chunk 640、decode graph
2/8 均保留。配置 context 311040、并发 4；本轮只测到 6400 token，不认证完整容量。
640 prefill 仍有约 46 段图、45 个 eager break。多模态输入按原 launcher 禁用。
这是已有模型的增量优化，没有新增环境变量、生产 patch 或 runner 行为。

保留失败证据：首次专家 inventory 请求不存在的 decode route-input binary；
后续可选审计误认 type metadata 为字符串。两次失败 native session 均通过
核对 PID/create-time、暂停且空闲后重建 worker，再继续 admission。
QSA v987/v988 编译分别遇到 tiling macro/core annotation 问题；v989 完整通过。
首次自由设备 QSA probe 缺少 serving libopapi 栈，已恢复服务并用原栈重测通过。

CPU tooling 1555 个测试通过，使用 `--confcutdir=tests/ut/glm_perf`；顶层 UT conftest 缺少
上游 Flash Linear Attention 包。格式检查、日志、冻结 bundle、真实权重门禁、
实际执行脚本和 SHA 均保存；大 JSON 以 `.json.gz` 保存，冻结证据不被 formatter 改写。

配套 [US English 报告](README.en.md) 和 [运行说明](RUNBOOK.md)。

Scoped Ruff/manual hooks 通过。按要求执行的 `bash format.sh ci` 因已有全仓
lint/format/import 问题失败；隔离 checkout 中 145 个无关 formatter 改动已恢复，
完整日志和清单均保留。
