# W4 scratch 缩减：2026 年 10 月 8 日离线交付

[US English summary and commands](README.en.md)

新增成对的 gate/up 与 down W4 入口。原生 W4 已直接复制 nibble，但通用
内核仍分配 W2/W3 重建用的大缓冲区。本候选删除其无用容量，保留所有存活用途。
永久检查点已有 22 层 W4，这些层无需扩展权重存储或重新量化即可使用本入口。

| 排程 | raw scratch | decoded scratch | gathered scratch | 每内核实例分配缩减 |
| --- | ---: | ---: | ---: | ---: |
| 通用入口 | 16,448 B | 65,536 B | 32,768 B | 对照 |
| W4 decode，M16/K128 | 1,088 B | 4,096 B | 8,192 B | 99 KiB |
| W4 decode，M16/K256 | 2,112 B | 4,096 B | 8,192 B | 98 KiB |
| W4 prefill，M32/K64 | 2,112 B | 65,536 B | 8,192 B | 38 KiB |

这些是编译实际内核 C++ header 得到的 **UB 分配减少**，不是释放的 HBM/KV，
也不是实测带宽、延迟或吞吐。增加排程空间的收益仍需硬件 profiling。

## 生命周期及完整路径

保留 raw 的 `ActivationWide` 打包容量。M32 的 decoded 缓冲区仍需完整 64 KiB，
用于两组 INT32 readback 及两组 FP32 结果；M16 保留八个 FP32 权重广播向量。
W4 将 scale factor 移到 gathered 起点，并保留最大 K、四个缓存行需要的 8 KiB。
mask、量化临时区、保留的行索引保持原样。FP32 乘积、尺度数学、归约顺序、输出
转换与量化不变。

显式 builder 参数 `--compact-w4-scratch` 新增两个 W4 二进制，入口固定接收
W4，直接复制权重。W2/W3 继续使用通用或 W3 专用入口；混合 gate/up、down 位宽
也按各自位宽分派。旧 bundle 保持原行为。必须使用 prepared 布局，不支持 LUT。

代码覆盖 builder、dispatch、冻结 profiler 校验、real-weight gate 报告、
resident 资格 manifest、永久检查点复制及成对完整 MoE 图验证。缺失或未声明的
W4 pair 在加载前被拒绝，所有配套 binary/helper checksum 必须通过。
profiler 能识别 W3/W4 专用句柄并归入正确 pipeline 阶段。

## 离线证据

- Decode **v958** 沿用 v956 排程，prefill **v959** 沿用 v957；均成功编译
  为 dav-2002，保留上一轮 rounded scale 和 native route column 候选。
  [编译日志](compile-only.txt)记录两次完成。
- Raw-scale decode **v960** 沿用 v952，raw-scale prefill **v961** 沿用 v953，
  均编译完成，见 [日志](compile-only-raw.txt)。这两个版本接受现有 FP32 raw scales，
  保留原尺度 cast，隔离 W4 scratch 对照不需要先转换检查点。
- [CPU suite](cpu-tests.txt)：**1,622 passed**；继续排除三个本机依赖不齐的
  文件：`test_glm5next_w2_assembly.py`、`test_grouped_gate_up.py`、
  `test_kpool_ops.py`。
- 测试编译实际 scratch header、检查边界、核验通用/W3/W4 builder 产物，
  以 A4/A8 和 2/17/640 tokens 测混合 bank 的 host dispatch，拒绝不完整或
  被篡改的 bundle，并要求显式 device flag、real-weight 资格标记。
- [最终 dispatch 回归](final-dispatch-regressions.txt)：并行扫描增加的真实 GLM
  method-selection 检查后 **39 passed**。
- [Scratch audit](scratch-audit.json)及保存的 C++ 源可用 host compiler 重现容量。
- [归档](w4-compact-v958-v961-20261008.tar.gz)保存 **84 个逐文件核验的文件**：
  binary、bridge、冻结 helper、参数、编译日志和源快照。
  [Manifest](compiled-artifact-manifest.json)及
  [归档摘要](archive-sha256.json)记录 SHA256；交付源码与编译 provenance 一致。

SDK compiler 为编译调用 ACL 初始化，但没有选择设备、加载 kernel 或提交推理。
远端 CPU 编译禁用 PyTorch backend 自动导入。没有修改 serving 文件、检查点、
图设置或服务状态。候选尚未经过 NPU 验证；按用户设备延期要求，本轮未运行
ACLGraph、EP/通信、MTP、多模态、容量、冷 prefill 或 decode 性能测试。

## 后续验证

[英文报告](README.en.md)提供尚未执行的完整 MoE 逐位图重放和交替计时命令。
远端 ROOT 为 `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007`；
现有 raw-scale 检查点先对照 `build-v952` / `build-v960-w4-compact`、
`build-v953` / `build-v961-w4-compact`；rounded-scale 则对照 v956/v958、v957/v959，feature 为 `compact_w4_scratch`。

恢复独占设备访问后先测 W2/W3/W4 × A4/A8 × 2/8/17/640 tokens 的完整原生
路由 MoE pipeline。算子门不等于完整模型门；仍需实际权重（含 W4）、target/draft
图重放及相同上下文设置下的冷长 prompt TTFT、c1/c4 decode 对照。

v958/v959 同时要求上一轮永久 rounded-scale 契约，不能接受 raw resident scales。
CPU `prerounded_scale_checkpoint` 导出器可使用本轮 bundle，复制路径已涵盖两个
W4 入口。现有 W4 bank 不需要码转换；上一轮无损 W3→W4 导出仍为独立候选，
需要其显式内存预算。本轮没有导出真实检查点，也没有启动模型服务。

## 并行代码扫描的分派核对

`glm5next_w2/model.py::_bind_eager_kda_forward` 对 NPU 输入选择
`run_stateful_kda_310`，prefill 调用 `chunk_kda_fwd`；`kda.py` 中 Python
逐 token 循环属于 CPU oracle/fallback。函数名中的 eager 不代表设备上
执行 Python recurrence。本轮没有查询实际 worker 分派。

实际 decoder 的 mHC 走 patched `MHCPreOp` / `MHCFusedPostPreOp`，
`glm5next/model.py` 的 FN、base、scale 参数本来就是 FP32，因此 host
reference 中的 `.to(float32)` 不能证明每次 forward 都分配权重副本。
`kda_310.py::prepare_kda_gate_weights` 也已经在加载后准备固定 gate scale/bias。
下一轮应先沿真实 native/patch 路径和检查点 flags 核对剩余 activation cast、
归约与 FP32 projection，再排列优化优先级。

永久 `glm_native_int4` loader 将 `module._method` 设为 `NativeInt4MoEMethod`，
真实 `Glm5NextW2MoE.method` property 使用此绑定。其调用使用驻留 bank 并返回
FP32，绕过旧 FP64 host oracle、逐专家 CPU staging 及旧 Python SwiGLU。
prepared 几何不支持时会抛错，不会静默进入旧 scheme。其他配置仍须检查旧分支；
本轮最终 CPU 回归通过实际 GLM routed-forward 验证了该分派。

共享 `csrc/gmm/w2_blocked_dequant_matmul_v310` 路径先扩展 FP16 权重再执行
Cube，其 GM workspace 模型不能归到独立的 `tools/glm_perf/glm_fused_moe.cpp`。
后者将 W2/W3 重建为有符号 INT4、直接复制 W4，再执行 INT4 Cube。
之前的尺度优化也在后者 `PrepareScales`，由 `GLM_PREROUNDED_WEIGHT_SCALES`
及永久 `fp16_rounded_fp32_v1` 标记控制。CPU 导出器把 `scale.half().float()`
保存为 FP32，config、manifest、safetensors metadata 和 bank 标记一致。
尺度乘法与累加保持 FP32；这不能说明 Qwen 格式或尺度契约相同。
