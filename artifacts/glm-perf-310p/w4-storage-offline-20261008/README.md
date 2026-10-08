# 无损 W4 存储候选：2026 年 10 月 8 日

[US English summary and commands](README.en.md)

本次继续离线优化 GLM 冷预填充与解码。新增 CPU 导出器将选定的 W2/W3
专家码无损扩展为磁盘上的原生 W4 布局，保留有符号整数值和权重尺度。
`glm_fused_moe.cpp::Decode` 的 W4 分支直接复制 INT4 字节，跳过每次执行的
W2/W3 重建、符号扩展与类型转换。现有 W3 重建后已经使用 INT4 Cube 数学；
本候选不减少 Cube 运算次数，也不改变算术精度。增加的权重读取可能抵消收益。

没有操作服务器、选择 NPU、加载内核、捕获图或转换实际模型。没有测得新的
延迟或 tok/s 收益。上一轮永久预舍入尺度候选仍独立保留；新导出器可与其组合，
并保持尺度标记及配套内核契约。

## 内存代价

磁盘上的永久检查点为
`/srv/ai/models/GLM-5.3-Flash-selective-W3-native-int4-310p`，288 个专家、
四个 rank、每 rank 72 个专家。43 个路由层（含 draft 层）中，22 层已经为
W4、8 层为 W3、13 层为 W2。W3 层为 8、9、10、12、13、14、16、17。
许多层已经使用 W4 直接复制路径。

| 扩展的 W3 bank | 每 rank 新增驻留码内存 | 四个 rank 新增磁盘 payload |
| --- | ---: | ---: |
| 单层 gate/up | 144 MiB | 2.25 GiB |
| 单层 down | 72 MiB | 1.125 GiB |
| 完整一层 | 216 MiB | 3.375 GiB |
| 完整四层 | 864 MiB | 13.5 GiB |
| 全部八层 | 1.6875 GiB | 27 GiB |

这些是由形状计算的准确字节数，并非实测剩余 HBM。必须先为 KV cache、
图池和 workspace 预留空间，再显式提供新增驻留码内存预算；工具不推断空闲
内存，也不修改上下文设置。[预算示例](memory-budget-examples.json)中的
1 GiB / 2 GiB 仅为示例：八层不满足前者、满足后者。尚无实测性能证明应选哪些层。
W3 扩展增加三分之一码内存；W2 扩展使码内存翻倍。

## 代码与验证

- `w4_storage_checkpoint.py`：仅读头的规划，以及按完整 gate/up 或 down bank
  导出全部专家/rank。CPU 直接扩展码域并验证逆变换；硬链接未修改分片，保留
  尺度、dense、draft、indexer，复制已核验的配套内核，最后发布 complete 标记。
- `w4_storage_probe.py`：待执行的完整原生路由 MoE 对照。输入量化、路由、
  gate/up、SwiGLU/隐藏量化、down、路由归约都包含在图重放中。分别比较只扩展
  gate/up、只扩展 down、同时扩展两者；使用相同构建和独立 scratch。
- 原生 loader：加载时校验新增码分片 checksum；运行与捕图路径不新增转换。

探针覆盖 A4/A8、2/8/17/640 tokens；要求有限 FP32 输出逐位相同。变更
activation、码、尺度、路由后重放，并检查全 peer / 零权重情况无陈旧结果。
按交替顺序计时。实际专家选项使用完整矩阵维度及同一 rank 的少量连续专家；
它是算子验证，不等于完整 rank、通信、长上下文 attention 或模型推理验证。

离线证据：

- [检查点清单](real-checkpoint-inventory.json)：只读 safetensors 头与索引，
  不映射 payload、不查询设备。
- [实际码 CPU gate](real-code-cpu-gate.json)：第 8、17 层专家 0–2 的 gate/up
  和 down，共 **150,994,944** 个有符号码值，独立解码后完全一致。
  尺度源摘要不变；没有写检查点、加载内核。
- [兼容 GLM 回归](cpu-tests.txt)：**1,576 passed**；继续排除三个本机缺少
  upstream/NPU 依赖的文件：`test_glm5next_w2_assembly.py`、
  `test_grouped_gate_up.py`、`test_kpool_ops.py`。
- [最终回归](final-regression-tests.txt)：增量导出完整性修改后 **64 passed**，
  覆盖 signed/unsigned 字节、所有 W2/W3 码、多个 tile/专家、准确预算、实际
  loader/bank 宽度、中断状态、checksum 拒绝、永久预舍入尺度组合。
- [实际 CPU gate 源快照](real-codes-cpu-source.tar.gz)：远端 CPU 实验使用的
  精确源代码与脚本；后续导出器 lineage 修改由最终回归验证。摘要见
  `evidence-manifest.json`。

远端 CPU 命令使用 PyTorch 已有的 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，
禁用 backend 自动导入；没有新增 Ascend 环境变量。按照用户要求，本轮
ACLGraph、实际模型 decode/cold prefill、EP/通信、MTP、多模态和容量测试
均未执行，`real_model_evaluated=false`。

## 操作说明

[英文报告](README.en.md)提供完整 plan/export 命令和待执行的图重放命令。
第 8 层两 bank 的示例预算为 226492416 bytes，即 216 MiB；它不代表运行时
存在这些空闲内存。实际模型尚未执行 export。目标目录必须不存在且位于同一
文件系统。磁盘 payload 大于驻留增量，因为原文件保留为硬链接。

同一层的不同候选须从同一 base 导出到独立目录；已存在目标及冲突分片会被拒绝。
不同层的增量导出保留此前完整性记录，新增预算相对于输入 source 计算。

恢复设备验证且有独占访问时，使用 prefill build957 测试 17/640 tokens，
decode build956 测试 2/8 tokens，先验证完整 MoE 的逐位重放与内存/计时。
探针在选择设备前校验全部配套二进制和 helper；W3 专用分派保持原行为，
W4 使用同一构建的通用直接复制分支。然后通过现有 `glm_native_int4` loader
验证导出后的完整 target/draft 模型、相同 KV/图/上下文设置、冷长 prompt TTFT
与 decode tok/s。CPU 码一致、synthetic 计时或启动成功都不能作为性能交付依据。
