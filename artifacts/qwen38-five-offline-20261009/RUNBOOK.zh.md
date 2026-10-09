# Qwen 候选的延后验证步骤

## 当前状态

继续暂缓启动服务器及 NPU 测试。五个候选均关闭，保留图像及原有容量。
使用包含前一批 P1 修复的完整源码快照，不能把个别文件混入旧服务目录。
`queued-profiles.json` 只有局部覆盖，明确禁止自动启动。

开发主机已有主机编译产物：

```text
/home/matteius/experiments/qwen-five-offline-20261008/build-v2
```

三个内核、`qwen_prefill_v2.so` 和指纹均已保存，但没有加载或执行。
需要重新编译时，使用新目录和唯一版本，并使两个候选中的
`RESOURCE_NAME` 与版本一致。清单创建会检查版本、接口及源码和二进制指纹。

```bash
python -m tools.qwen4exp.prepare_native_prefill \
  --build /path/to/build-v2 --runtime /path/to/complete/snapshot \
  --output /path/to/new/native-manifest.json
python -m tools.qwen4exp.speculation_sweep > mtp-arms.json
```

以上命令仅生成文件；清单加载是另一个设备操作，目前不得执行。

## 再次获得 NPU 授权后

先独立验证前一批 P1 修改。保持 94°C 暂停、85°C 恢复和 96°C 硬停止，
检查所有设备的温度记录。温控不能中断正在执行的单个内核。
组件测试需要空闲 NPU 和外部温控监督，具体命令见 `RUNBOOK.en.md`。

1. 对比 `paged_native` 和批量收集 QSA，检查长上下文、多请求、尾部与图像。
2. 验证融合 WY 的 4/12、16/48 头布局、完整 H/O 及 FP32 最终状态。
3. 确认本地路由操作数与原量化器字节一致，并使用相同激活精度对比。
4. 在独立诊断服务中，用现有 resident 控制器加载清单，一次只切换一个候选。
5. 真正通过文本、图像、状态及温度测试后，才组合候选。

本地路由必须使用原生 INT4 和 `cann_swiglu_pack`。
它与服务中 `cann_builtin_fp16` 的差异须单独进行质量和性能对比。
控制器执行既有排空、重置和图捕获流程；请求因温度暂停时不要切换配置。

预填充分块调度器通过启动参数显式选择：

```text
--scheduler-cls vllm_ascend.core.qwen_prefill_scheduler.QwenPrefillPacedScheduler
--additional-config '{"qwen_prefill_pacing":{"target_step_ms":500}}'
```

这不是当前的启动授权。保持图像和有界前缀缓存；验证新图像、缓存图像、
取消、前缀复用及解码期间到来的长预填充。记录解码间隔、排队时间、
缓存溢出及温度。500 毫秒是估计目标，不保证单步耗时。

## MTP0/1/2 对比

每种深度需要独立的缓存规划和两个图尺寸；MTP0 完全移除推测解码配置。
不能通过修改标量在运行中切换深度。核对实际启动配置、服务别名和深度，
并保留启动配置校验和。

采集器在服务主机上运行，要求正确 API PID、相同请求 JSON、独立质量和
图像凭据，以及外部温控 JSONL。默认测试并发 1/2/3，每个并发配置至少
600 秒，分三次重复窗口。它不会启动、暂停或重配置服务器。
API 身份改变、步骤未完成、温度记录缺失或过早恢复会使结果不能入选。

合并三个深度的采集文件后离线比较：

```bash
cat mtp0.jsonl mtp1.jsonl mtp2.jsonl > matched-arms.jsonl
python -m tools.qwen4exp.speculation_sweep --records matched-arms.jsonl > comparison.json
```

客户端流块只近似解码计时；包括预填充、排队和冷却的墙钟吞吐优先用于选择。
核对相同请求、输出长度和冷热缓存协议，结合服务日志判断前缀是否命中。
质量和图像通过来自独立验证，不由性能采集器自行声明。
没有实际能源测量时，能耗保持未知。

## US English summary

Startup and NPU work remain deferred. After authorization, gate preceding fixes,
then each candidate independently with images and FP32 state preserved. Keep
94/85 thermal hysteresis and the 96-C hard stop. MTP comparisons require matched
requests, separate cache/graph plans, three repeats and sustained sensor coverage.
