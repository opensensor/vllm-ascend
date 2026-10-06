# GLM 310P 原生低位 MoE 实验

完整硬件及在线结果见 [实验记录](../../artifacts/glm-perf-310p/reconstruction-20261006/README.md)。
后续布局调度及全链路结果见 [原生 INT4 调度实验](../../artifacts/glm-perf-310p/native-schedule-20261006/README.md)。
无损权重准备、后续在线结果及 prefix cache 检查见
[准备布局实验](../../artifacts/glm-perf-310p/native-prepared-20261006/README.md)。

## 内核融合边界

`fused_int4a8` 与 `fused_int4a4` 使用两个原生内核：

原始 token 先量化一次，packed 输入在所有 expert/output tiles 间复用。

1. packed 激活→INT4 gate/up→UB SwiGLU→量化 hidden scratch。
2. 量化 hidden→INT4 down→UB FP32 route-weight reduction→最终 token output。

不生成 GM FP16 gate/up、hidden activation 或 routed-down 矩阵，不再调用
独立的 SwiGLU、hidden activation pack 或 combine kernel。down 的 token chunks 在 UB
中按 expert 顺序累加，每个 output column tile 只有一个 core owner，最终
只写一次 FP32 token output，不需要浮点 atomic 或 GM 部分和 round-trip。
routing 元数据仍由原有 device-resident sort/count 构造；attention、dense
及 shared expert 分支沿用当前模型实现。

两种模式保留 checkpoint 的 W2/W3/W4 signed codes 与 32×32 scales。
W2/W3 在 UB 精确扩展为 INT4；UB nibble 提取含精确 FP16 归一化，Cube
矩阵乘均为 INT4。A8 每组激活量化为 INT8，以 `a8 = lo + 16*hi + 8` 的
两个 INT4 limb 和权重和修正计算；A4 直接使用一个 signed INT4 limb。
SwiGLU 和 scale/累加仍使用浮点数学。A4 引入更大的激活量化误差，两种
模式均需要完整模型质量 gate，不能用算子参考通过代替。

v22 在 UB 按 byte plane 批量解码，用完整 K=64 tile 与相邻两个 K=32
scale groups 共享权重。激活及 A8 bias row 只启用对应的 K 半区，仍保留
原有每组 scale 与数值参考。W3 planes 在 UB 重排，使用四次 repeat 的
转置，避免重复的小转置。FP16 signed codes 只用于 UB 中的格式转换与
INT4 pack；矩阵乘是 INT4。scale、token index 在单次 projection 内缓存。

`--prepared-weight-layout` 在暂停 worker 时按 expert 将原始 NZ bank 无损重排：
W4 直接存储 Cube INT4 bytes；W2/W3 保留原位宽，推理时只扩展 signed codes，
不再转置。shape、storage pointer、quantized codes 与 scales 均不变。
CPU 保存原 bank bytes，用四个 CPU threads 直接转换 byte fields，每个 expert
核对 inverse 后只按整 bank 做一次 H2D；完整 D2H SHA256 验证 device bytes。任何部分失败
先恢复。切出 candidate 前逐 bank 还原并核对 SHA256，之后才移除 hook、
捕获 baseline graph。备份要求所有 rank 总 bank bytes 加 8 GiB host reserve；
不保留第二份 device bank。`weight-layout-rank*.json` 按 bank 写入完成数、耗时、
原始/准备后/还原 SHA256；写文件使用 atomic replace，监测无需额外 RPC。
布局准备完成后不允许 fallback 到旧 NZ kernel，未支持 geometry 立即拒绝。

A8 在 sparse batch 将两个 INT4 activation limb 和 bias row 放入同一 M=16
fractal。`--pair-scale-groups` 再将相邻两个 K32 group 放到未使用的 Cube rows，
共享一次 K64 Mmad、readback、cast。两组各自的 integer dot、scale 和 FP32
累加顺序保留；A8 count≤3、A4 count≤8 使用该调度，更大 batch 使用原路径。
这与增大 quantization group 或合并 scales 不同，不引入新的量化误差。
后续内核将 even/odd channel 两个半区的同一 block32 scale
改为一条 repeated vector instruction，并批量累加当前的 active rows。
每个 group 的 scale product 与 FP32 加法顺序保持不变。

`int4a8` 是早期组合入口，使用独立 projection、SwiGLU 和 combine；
`w4a8` 仅覆盖 W4。它们不能充当上述内核融合实现的结果。
`gm`、`l1`、`l1_singleton` 保留 W3 FP16 reconstruction 调度筛选。

## 构建与独立 gate

使用服务匹配的 CANN、torch-npu；每个新版本采用未使用的 namespace 和
目录。v28/v29 已构建，下面以 v30 为新构建例子。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m tools.glm_perf.build_reconstruction \
  --build-dir /tmp/glm-fused-v30 --version 30 \
  --output-columns 128 --tile-pipeline --all-bits --fused-moe \
  --prepared-weight-layout --pair-scale-groups
```

```python
from pathlib import Path
from tools.glm_perf.fused_moe_probe import run

run(
    Path("/tmp/glm-fused-v30"),
    Path("/tmp/glm-fused-v30-gates.json"),
    checkpoint=Path("/srv/ai/models/GLM-5.3-Flash-selective-W3-310p"),
    real_prefixes=tuple(
        f"model.language_model.layers.{layer}.mlp.experts.0."
        for layer in (10, 11, 33)
    ),
)
```

两种激活精度×三种权重位宽×tokens=1/2/3/17/128，共 30 个完整 MoE
参考与 graph replay；另有两种激活精度×三个真实 expert 的六个 gate。
捕获后修改激活、route ids/weights 与 gate/down 权重，并测试全 peer 输出
严格为零。参考使用独立 CPU quantization/整数部分积/FP32 reduction。
真实权重 gate 使用合成激活，在线 workload 才检查模型输出。

这些直接内核不依赖早期 `glm_decode_flags.so` 的 SwiGLU/combine operators。
旧 `reconstruction_probe.run` 若使用 `--all-bits` 而没有 `--fused-moe`，仍需
原服务 baseline binding、OPP stack 和 `--support-library`；该旧路径的 gate
报告与 fused gate 格式不同。

## 热切换与恢复

```bash
python -m tools.glm_perf.reconstruction_control manifest \
  --build-dir /tmp/glm-fused-v30 --gate-report /tmp/glm-fused-v30-gates.json \
  --output /tmp/glm-fused-v30-manifest.json
python -m tools.glm_perf.resident_harness --base-url http://127.0.0.1:8001 \
  load-native /tmp/glm-fused-v30-manifest.json
python -m tools.glm_perf.reconstruction_control trial \
  --base-url http://127.0.0.1:8001 \
  --base-source /home/matteius/experiments/glm-decode-flags-20261006/resident-combine.py \
  --profile fused_int4a8 --native-resource reconstruction_v30 \
  --max-tokens 64 --require-full-coverage --quality --timeout 3600 \
  --output /tmp/glm-fused-v30-a8-trial.json
```

再用 `--profile fused_int4a4`、新的 output 路径测试 A4。manifest 要求六种
精度组合的真实权重、prefill、变更输入/路由/权重 replay 通过，且 binary 与
immutable helper hashes 匹配。每个 worker 注册前独立验证六个完整 MoE
case。注册是 append-only；已经加载的 binary/helper 目录不得覆盖。

prepared-layout trial 的控制 RPC 使用 3600 秒 deadline。不要在重排时另外提交
worker RPC，也不要在原 RPC timeout 后凭晚到的 status receipt 恢复 serving：
旧 executor 会留下未消费回复，后续 model call 可能读到 status dict。
已有事故记录及修正见 native-prepared 实验报告。

trial 验证 exact base source、graph、native resources、pause 状态。
严格模式要求每个 worker capture native>0、fallback=0，完成请求后再次
要求 fallback=0。bank_coverage 记录位宽和完整几何，kernel_fused_calls
区分真实融合入口。Python counter 记录 capture 和未捕获 prefill 调用；
graph replay 不会增加 Python counter。首次新 prefill shape 的 metadata
只使用 CPU-visible shapes，不读取 device routes，不增加 `.item()`。

测试包含五个 64-token short、20 个 128-token exact-answer 和一个
128-token tool 请求，seed=42；baseline 使用同一 workload。任何 strict
失败均保留原始输出及 trial_error，不能把 baseline 原有失败当成新增退化，
也不能把短题结果当成长上下文或大型准确率评测。

结束或失败后，在 finally drain/pause，先清除 candidate，再从 exact base
source 构造原有 fusions 和 graph，验证 PID、weight storage/source digest、
graphs clean、reconstruction audit 消失，再恢复服务。这样避免 factory
将 native wrapper 嵌入看似已经恢复的 baseline。默认模型分发不启用实验。

## 离线校准与 CPU 测试

`calibrate_projection` 用校准激活的 32×32 covariance 搜索 scale，并用
held-out 激活报告输出误差。它是 block loss surrogate，没有 GPTQ error
feedback，不修改在线 checkpoint 或自动分配位宽。输入 safetensors 键为
有限 CPU 浮点的 `weight[N,K]`、`calibration[samples,K]`、`validation[heldout,K]`。

```bash
python -m tools.glm_w2.calibrate_projection \
  --input /tmp/projection-calibration.safetensors \
  --bits 3 --output /tmp/calibrated-projection.safetensors
python -m pytest --noconftest -q \
  tests/ut/glm_perf/test_fused_weight_layout.py \
  tests/ut/glm_perf/test_fused_moe_profile.py \
  tests/ut/glm_perf/test_fused_moe.py \
  tests/ut/glm_perf/test_glm_int4.py \
  tests/ut/glm_perf/test_reconstruction_control.py \
  tests/ut/glm_w2/test_calibrate_projection.py
```

校准输出独立 codes/scale artifact 与 JSON，完整模型速度及质量必须另行验证。

## 永久 Cube checkpoint

当前完整布局位于 Threadripper 的
`/srv/ai/models/GLM-5.3-Flash-selective-W3-native-int4-310p`。
原 checkpoint 保持独立；未变的 dense/scale shards 使用 hard links，native code
shards、config/index、kernel bundle 与 complete manifest 是新文件。

启动使用原 server command，仅替换模型目录及两个 load formats：

```text
--load-format glm_native_int4
--speculative-config '{"method":"mtp","num_speculative_tokens":1,"draft_load_config":{"load_format":"glm_native_int4"}}'
```

要求启动日志同时出现主模型与 draft 的 `GLM native INT4 loaded directly`，
`transformed_code_tensors=0`、`layout_backup_bytes=0`。loader 按 index 只读本 rank
的 native codes，逐 tensor CPU staging 是字节复制，不是量化或布局转换。
此 checkpoint 必须使用 native loader；普通 safetensors loader 不理解 Cube bytes。
这是为避免 Ascend driver 对 file-backed safetensors 页的长期 pinning stall。

重新导出时用 `python -m tools.glm_perf.native_checkpoint initialize` 创建新输出，
paused worker 的 `export_rank` 逐 bank 独立对照 canonical checkpoint，再用
`finalize` 检查四 rank 与全部 code keys。没有 complete manifest 不能直接载入。
MTP 源 bank 的 layout 不能从主模型推断：本次实际 MTP 是 canonical，而主模型
是 NZ。若独立对照失败必须从 canonical 源生成并做 full inverse check。

更多 profiling 使用直接 CANN lifecycle，start/stop 各提交一次并以只读 status
确认；必须在 graph recapture 前 stop/finalize。profiled latency 不用于 tok/s 声明。
