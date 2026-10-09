# 后续设备验证说明

[英文完整命令](RUNBOOK.en.md) · [中文报告](REPORT.zh.md)。

目前仍不启动服务或执行 NPU 工作。以下说明用于未来再次授权后验证。
使用完整提交的 runtime，保留图像开启、TP4/EP4、模型、缓存容量、graph
尺寸与温控策略，不向运行中的服务零散复制文件。

## 主机准备

在匹配服务的 Python 环境中运行 `tools.qwen4exp.build_native_transfer`，
指定完整 runtime、CANN 根目录、新的 append-only build 目录和 `--version 1`。
再用 `tools.qwen4exp.prepare_native_transfer` 生成仅含指纹的 manifest。
这两步只编译与检查文件，不执行设备代码。

主机已成功生成
`/home/matteius/experiments/qwen-transfer-next-20261009/build-v1c`，命名空间
为 `qwen_transfer_v1`。旁边 `source` 仅是编译输入，不是服务 runtime。
已加载过的二进制若变化，必须使用新版本并更新绑定，不覆盖旧记录。

## 再次授权之后

先在独立诊断设备执行 `tools.qwen4exp.benchmark_transfer_next_310`。
需要一致的 FP32 GDN/W4 OPP。验证 state gather/scatter、4/12 与 16/48
头的完整 H/O 输出和最终状态、W4 gate/up 与 down、非完整行块、空专家、
peer tail。该命令会执行 NPU，目前不得运行；不证明速度、图像、容量或热稳定。

诊断服务使用
`tools.qwen4exp.resident_worker.QwenResidentExtension` 作为 worker extension。
现有 resident harness 可在请求排空后加载 manifest、切换一个候选、回滚和
重新捕获。它可能清除 prefix cache，不要用于当前 demo 或热暂停期间。
保持四个 rank 的完整回执，不改变 OPP 搜索路径或模型权重。

候选位于 `tools/qwen4exp/resident_candidates/`：

- `transfer_audit.py`：有界时间明细与 runner/W4 记录，prefix 累计计数已可用。
- `prefix_phase_batching.py`：同一设备各 tier 的失效/CoW 与 admission 分阶段同步。
- `native_state_layout.py`：需 `qwen_transfer_v1`、FP32 状态与合法唯一写回槽。
- `cached_w4_metadata.py`：需同一资源、grouped native INT4、128 本地专家与八 lanes。
- `tp_prefill_pipeline.py`：需 TP-sharded shared expert，所有 rank 分块与通信顺序一致。

修正后的 fused-WY 继续使用旧 build 的 `qwen_prefill_v2`，但应重新生成包含
本次模型文件的 manifest。切换会解开同方法的旧候选，不能自动组合优化。
安装 telemetry 后重新保存 baseline，比较器拒绝跨配置切换的样本。

## 保存统计与真实负载

以后授权的诊断服务可通过 resident harness 的 `status` 保存 before/after
JSON，中间执行相同且配置不变的工作负载。再离线运行
`tools.qwen4exp.transfer_snapshot --before ... --after ... --output ...`。
要求四个 rank、PID/配置/存储不变；缓存重置不清除累计计数，须根据同步原因
排除管理操作。事件有截断，总计数保留；这些字节不是 DDR 总线实测。

验证独立 cold 与已证明命中的 warm prefix、C1/C2/C3、混合预填充/解码、
fresh/cached image、thinking/tool call、CoW、取消与回滚。核对图像、容量与
MTP 接受率，并同步采集四 rank 的 memcpy 字节/方向、事件、HCCL、MTE 和温度。

保持 94°C 暂停、全部传感器不高于 85°C 才恢复、96°C 独立 cutoff，暂停归属
和缺少传感器的行为不变。持续热稳定验证通过前不得推荐性能或部署。各候选
单独合格后，组合还须重新完整验证；现在仍不得启动服务。
