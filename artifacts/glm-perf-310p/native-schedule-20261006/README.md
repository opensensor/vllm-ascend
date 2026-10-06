# GLM 原生 INT4 调度实验（2026-10-06）

v22 改善完整 routed MoE 原生 INT4 路径，但仍慢于在线 FP16 Cube
baseline。两种激活精度均完成 26 个真实请求，四个 worker 全 bank
覆盖、零 fallback；未提升为默认实现。最终恢复 `decode_combine`、
干净 ACLGraph 和运行状态，PID 与 weight-storage identity 均保持。

## 在线结果

Ascend 310P3 ×4（两张双芯片卡），TP4、MTP1、ACLGraph sizes=2/8、
context=311040，checkpoint 与 OPP/binding 沿用
[前轮记录](../reconstruction-20261006/README.md)。short 请求各 64 tokens，
temperature=0、seed=42；c4 为四请求 aggregate decode throughput。

| 实现 | c1 tok/s | c4 aggregate tok/s | 严格质量 | tool |
| --- | ---: | ---: | ---: | ---: |
| 本轮 baseline before | 5.001 | 11.910 | 未重复 | 未重复 |
| v22 INT4 / A8 | 2.683 | 6.021 | 17/20 | 1/1 |
| baseline after A8 | 5.221 | 12.416 | 未重复 | 未重复 |
| v22 INT4 / A4 | 3.068 | 5.379 | 18/20 | 1/1 |
| baseline after A4 | 5.179 | 12.452 | 未重复 | 未重复 |

前轮 v13 A8 为 c1 1.602 / c4 3.759；A4 为 1.793 / 3.108。
本轮 c1 分别提高约 67.5% / 71.1%，但跨轮比较受运行条件影响；
本轮前后 baseline 仍明显更快。A4 本轮 c1 比 A8 快约 14.3%，
c4 更慢，不能将单 expert 的 device interval 加速等同于完整服务收益。

[A8 全链路请求](live-fused-v22-fused_int4a8-trial.json)与
[A4 全链路请求](live-fused-v22-fused_int4a4-trial.json)均为 26/26 有效；
十个 short 请求均完成 64-token budget。两种 profile 捕获后每 rank
为 344 native / 0 fallback / 172 kernel_fused_calls；完整请求后为
1032 / 0 / 516。覆盖 42 个主模型 sparse layers 与一个 MTP sparse layer
的 W2/W3/W4 gate/up/down，包括 prefill 与 decode。

严格 gate 未通过，controller 保留 `trial_error` 与完整结果后恢复 baseline。
A8 的失败为 `instr_reverse=pial`、`instr_first=Red`、
`code_slice=\`'lan'\``；A4 为`instr_first=Red`、`code_slice=\`lan\``。
前轮 baseline 为 17/20；A4 的 18/20 不构成广泛质量认证，未放宽检查。

最终 [worker 状态](resident-status-fused-v22-after.json)：rank 0..3 PID
2876823 / 2877237 / 2877601 / 2878087，baseline digest
`4a40c9fbba8af6f3bb2c093a871a0bcf028be3a18af39813f982a2ec51b63da9`，
`graphs_dirty=false`、`native_failed=false`、`is_paused=false`。
没有重启 worker、改写 checkpoint 或重排 resident weight banks。

## 调度改动

保留两个融合计算阶段：gate/up + UB SwiGLU + hidden quant，
down + UB FP32 weighted reduction。原始 token 先量化一次；没有 GM
FP16 gate/up、hidden 或 routed-down 矩阵，没有 FP16 weight GEMM。
attention、dense、shared expert 分支沿用现有模型。

v22 按 byte plane 批量解码，减少每 strip 的 scalar/vector 设置。
signed codes 在 UB 转成 FP16，仅用于格式转置与 INT4 pack；Cube
仍执行 INT4×INT4→INT32。完整 K=64 权重 tile 由相邻两个 K=32
scale groups 复用，激活及 A8 bias row 只启用各自半区。
W3 四个 field 在 UB 通过 bulk copy 重排，以四次 repeat 完成转置，
避免大量小转置。weight scales、activation scales 和 token indices
在一次 projection 内缓存；仍保留原有 32×32 scales 和数值参考。

kernel 中无 host route 读取或 NPU `.item()`。移除 CPU gather-offset
metadata；所有 weight-layout scratch 留在 UB，checkpoint packing 不变。

## 独立证据

[v13→v22 device events](native-fused-v13-v22-stage-events.json)使用同一
真实 expert、两行合成激活、预先准备的 topk1 routing；warmup=3，
samples=9，报告 NPU event 中位数。interval 包含 native launch 间隔，
不是完整 serving 吞吐或精确 Cube-only 时间。

| 权重 / 激活 | v13 gate+up ms | v22 gate+up ms | v13 down ms | v22 down ms |
| --- | ---: | ---: | ---: | ---: |
| W3 / A8 | 4.422 | 2.007 | 2.199 | 1.003 |
| W3 / A4 | 4.170 | 1.715 | 2.069 | 0.858 |
| W4 / A8 | 2.708 | 1.755 | 1.344 | 0.874 |
| W4 / A4 | 2.457 | 1.454 | 1.213 | 0.728 |
| W2 / A8 | 2.770 | 1.760 | 1.374 | 0.876 |
| W2 / A4 | 2.515 | 1.457 | 1.243 | 0.730 |

W3 stages 提高约 2.2–2.4×，W4 约 1.5–1.7×。A4 减少 limb arithmetic
后同权重 stage 仍只提高约 15–20%，说明 INT4 峰值吞吐不是当前
端到端速度的充分解释。K=32 scale groups 仍要求逐组读取 INT32
部分积并执行浮点缩放，且每次 token 仍转换权重布局；需要继续测量
Cube/vector/MTE overlap、layout cache 和硬件对齐 quantization groups。
更改 group scale 或 quantizer 会改变数值，必须重新通过质量 gate。

[v13 消融](schedule-ablation-v13.json)分别抑制 decode、group math、
epilogue，显示 decode 消融将 W4 intervals 减少约 60–70%、W3 约
75–80%。消融输出刻意无效，未加载服务；这些差值是诊断，不是
测得的 HBM traffic 或严格可加的时间拆分。
[诊断源快照](schedule-ablation-source.txt)可复现该冷测试。

v14 的 table decoder 通过 gate，但 W4 仅改善约 7–8%；v15 的第一版
转置通过 gate 却更慢。v17 / v19 在 A8 numerical gate 失败，v18
编译失败，均未加载服务。v20 是正确的 K=64 schedule；v21 把转置
拆小后回退，完成 native registration 与一个 c1 sample 后主动终止，
没有完整 benchmark 结论，见 [恢复记录](fused-v21-stopped-restoration.json)。
v22 冷 gate 30 个 synthetic + 6 个 real expert 均通过；graph replay
修改激活、ids、weights、gate/down codes，覆盖 tokens=1/2/3/17/128
与全 peer 零输出。每个 worker 注册时另验证六个完整 MoE cases。

额外的 [v13/v22 精确对照](native-fused-v13-v22-exactness.json)使用相同
真实 W2/W3/W4 experts、两行合成激活与 topk1 routing。两种激活精度的
六组完整 MoE 输出各 8192 elements 全部 bitwise equal，最大差值为零。
这验证样本中的调度等价，不等同于所有输入或完整服务的 bitwise 保证；
在线 batching/graph 执行与独立 expert 测试范围不同。
[对照源快照](exactness-source.txt)在冷 profiler 外补充该比较。

## 复现与交付

构建和热切换见 [runbook](../../../tools/glm_perf/reconstruction_experiments.md)。
每个 build 使用唯一 namespace 与 frozen helpers，所有已加载资产保持
append-only；[v22 provenance](build-native-fused-v22/provenance.json)、
[manifest](native-manifest-fused-v22.json)、
[device gates](native-fused-v22-gates.json)绑定 binary/helper hashes。

新增 `tools/glm_perf/fused_moe_profile.py` 校验 frozen sources 与 binaries，
在独立进程中顺序比较真实专家的 device intervals；不会操作 resident
server。仅在 serving 无请求时运行，以避免设备竞争。

```bash
python -m tools.glm_perf.fused_moe_profile \
  --build-dir ../build-native-fused-v13 \
  --build-dir ../build-native-fused-v22 \
  --checkpoint /srv/ai/models/GLM-5.3-Flash-selective-W3-310p \
  --output ../new-stage-events.json
```

CPU：99 tests passed（`--noconftest` 的四个原有测试文件及新增 profiler
测试）；包括 signed codes、adjacent scale groups、peer/duplicate routes、
完整 MoE wrapper、immutable manifest、失败恢复与 profiler cleanup。
标准根 conftest 依赖当前本地缺失的 vLLM flash-linear-attention；未宣称
全项目 suite 通过。相关文件的 manual pre-commit hooks 通过。
