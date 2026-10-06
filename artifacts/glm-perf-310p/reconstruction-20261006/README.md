# GLM 原生低位 MoE 融合实验：2026-10-06

已实现 v13：每个原始 token 只量化一次，两个原生 INT4 内核分别融合
**gate/up→SwiGLU→hidden 量化**和 **down→weighted route reduction**。
不生成 GM FP16 gate/up、hidden activation、routed-down 矩阵；down 在 UB
累加，只写一次最终 FP32 token output。所有现有 W2/W3/W4 routed banks
均走 INT4 Cube，包含 42 个主模型 sparse layers 和一个 MTP routed layer。
两种激活精度均已完成真实 GLM 在线请求、前后 baseline 和恢复。

融合及覆盖已验证，但当前原型仍慢于 baseline，没有得到目标加速。
routing 使用原有 device sort/count，attention、dense/shared 分支维持当前
模型实现。所有改动都是实验入口，默认分发没有启用候选。

## 端到端结果

同一热服务、TP4、MTP1、full ACLGraph sizes 2/8、seed=42。每种候选包括
五个 64-token short、20 个 128-token exact-answer 和一个 128-token tool
请求。所有 52 个候选请求 valid，short 均达到预算；c4 为四流 aggregate。

| Candidate | c1 tok/s | c4 aggregate tok/s | 凭据 |
| --- | ---: | ---: | --- |
| v13 前 baseline | 5.174 | 17.006 | [before](live-fused-v13-baseline-before.json) |
| Fused INT4 / A8 | 1.602 | 3.759 | [trial A8](live-fused-v13-fused_int4a8-trial.json) |
| A8 后 baseline | 5.173 | 18.152 | [after A8](live-fused-v13-baseline-after-fused_int4a8.json) |
| Fused INT4 / A4 | 1.793 | 3.108 | [trial A4](live-fused-v13-fused_int4a4-trial.json) |
| A4 后 baseline | 4.998 | 12.342 | [after A4](live-fused-v13-baseline-after-fused_int4a4.json) |

baseline c4 有波动，两个候选仍低于全部 controls，不能宣称 A4 端到端
性能更优或实现已达到生产要求。c1 的 A4 改善也不能外推为四倍加速。

每个 worker 的 capture 为 344 native、0 fallback、172 kernel_fused_calls；
完整请求结束为 **1032 native、0 fallback、516 kernel_fused_calls**。W2/W3/W4
的 gate/up 和 down 几何全部出现，包含未捕获 prefill。Python counter
记录 capture 和 prefill 调用，graph replay 不会重复增加 counter。

## 质量 gate

baseline、A8、A4 均为 **17/20 exact-answer，tool 1/1**，失败 case IDs 相同：

| Case | Expected | Baseline | A8 | A4 |
| --- | --- | --- | --- | --- |
| instr_reverse | pmal | pail | pial | pma |
| instr_first | red | Red | Red | Red |
| code_slice | lan | 带反引号的 lan | 带反引号的 lan | 带粗体标记的 lan |

baseline 记录见 [完整 26 请求控制](live-allbits-v9-baseline-before.json)。
reverse 是真实答错，另外两题违反严格格式；没有修改 workload 或宽松计分。
strict gate 因这些失败报错，两次都保留 trial_error 并恢复服务。同分、同失败
IDs 不表示输出或完整模型质量不变；没有评估大规模准确率或长上下文质量。

## 原生内核与独立验证

[完整 v13 gates](fused-v13-full-gates.json)覆盖两种激活精度×三种权重位宽，
tokens=1/2/3/17/128 的 30 个完整 MoE case，另有六个真实 expert gate：
layer 10/11/33、expert 0 的 gate/up/down 权重。每个 case 捕获后修改激活、
route ids/weights 与权重，再测试全 peer 输出严格为零。
[四 worker 注册](native-fused-v13-load-receipts.json)各自验证六个完整 MoE case。
[构建哈希](native-fused-v13-provenance.json)与 [manifest](native-manifest-fused-v13.json)
绑定 exact binary 和 immutable private helpers，不覆盖既有 OPP 或模块。

W2/W3 权重在 UB 精确扩展为 signed INT4。A8 将每 32 元素激活量化为
INT8，再用 `a8 = lo + 16*hi + 8` 的两项 INT4 产品及权重和修正；A4 直接
使用单个 signed INT4 产品。SwiGLU、scale 和累加仍为浮点数学。nibble
重排含精确 FP16 归一化，矩阵乘使用 INT4，不经过 FP16 权重 GM workspace。

## 剩余成本的证据

[Device events](native-fused-v13-stage-events.json)使用同一真实 W4 expert、
两行合成激活、预先准备的 routing，三次 warmup 后五次采样中位数。
这些是 standalone launch 区间，不能直接等同完整模型 profiler 归因。

| 激活精度 | input quant ms | gate/up+SwiGLU+quant ms | down+reduce ms |
| --- | ---: | ---: | ---: |
| A8 | 0.036 | 2.708 | 1.344 |
| A4 | 0.034 | 2.457 | 1.212 |

输入量化已很小；去掉 A8 limb 后，两个 compute 区间只改善约 9–10%。
实现仍保留每 K=32 group 的 K=64 padding、INT32 partial readback、scale
修正、weight repack 与保守 barriers。A8 两个 limb 加 padding 消耗理想
低位吞吐收益；A4 仍承担每组开销。上述成本可在源码看到，但尚未将这些
compute 区间细分为各类指令的因果占比，不能只靠 bit 数推断 tok/s。

## 失败迭代及早期对照

- v12 已融合两个 compute kernels，但每个 output tile 重做输入量化，
  破坏 reuse；停止该测试 client，drain 并 [恢复 baseline](fused-v12-stopped-restoration.json)。
  未完成的 serving 测量不作结果。v13 改为每个原始 token 一次量化，量化
  buffer 在所有 expert/output tiles 之间复用；新增 regression 防止重复。
- v9 只有独立 projection/SwiGLU/combine 的组合，没有内核融合。
  [全 bank trial](live-native-allbits-v9-trial.json)为 c1 1.667、c4 3.374。
- v8 的 uint8 W2 banks 曾导致每 worker 八次 fallback，严格 capture gate
  在请求前拒绝，见 [失败](live-native-allbits-v8-trial.json)与
  [诊断](native-geometry-failures.json)。现已接受 int8/uint8 raw byte 容器。
- dav_m200 SDK 的 ShiftLeft/ShiftRight 编译为 unsupported stub，7<<4 返回 7；
  [device 证据](vector-shift-evidence.json)与 [v4 失败](native-pipeline-v4-failed-gates.json)
  保留该发现。改用精确归一化后才通过独立参考。
- 早期仅 W4 的 N16/N64/N128 及 W3 GM/L1 对照留在目录。N128 v5 为
  [c1 2.891、c4 6.688](live-native-pipeline-v5-trial.json)，仍慢于其 controls。
  部分 bank 的结果不能代替 v13 的完整 routed pipeline。

## 恢复及交付

[最终状态](resident-status-fused-v13-after.json)为 decode_combine、graphs clean、
native_failed=false、未暂停，四个原 worker PID 和 weight storage digest
保持不变，reconstruction audit 消失。没有重启服务或改写 checkpoint。

设备为四个 310P3 worker（两块双芯片卡），served model 为
`glm53-flash-selective-w3`、port=8001、既有 max context=311040。
baseline binding hash `9b61b2069b44638193918de0ba76e546e1003dae535a7c27414a7f4ee0924d73`；
launcher 的 `e229e66dbe61649b446b9fd9de9ccaed93685a8314d421831d107fefd646de87`
属于另一文件 glm_decode_flags.so，提供旧 SwiGLU/combine operators。
它不是 baseline binding 的另一版本；v13 不调用这些旧 fusion operators。

92 个 CPU 测试通过；本改动全部 scoped manual hooks 通过。独立 review
worktree 完整 bash format.sh ci 因 HEAD 已有 Ruff undefined names、拼写、
Markdown 及 forbidden imports 失败，没有修复或提交这些无关文件。
复现命令及离线校准见 [runbook](../../../tools/glm_perf/reconstruction_experiments.md)。
