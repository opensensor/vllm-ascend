# Qwen 310P 有界 prefix Mamba 保留策略

本配置最初仅完成离线验证。用户于10月8日重新授权NPU测试，现使用
8001独立测试服务；没有修改Kilo配置。实机结果和最终服务状态见
[验证报告](../../../../artifacts/qwen38-prefix-npu-20261008/REPORT.zh.md)。
[US English companion](qwen38_310p_bounded_prefix_runbook.en.md)提供一致说明。
图片能力已确认，但长会话 checkpoint 保留量和 attention KV token 容量是
两个独立预算。旧服务在 KV usage约10%时仍耗尽 primary63与archive175–194，
发生NPU→CPU spill；只读状态当时所有group的restore_count仍为0。

## 修复范围

显式选择 `PrefixMambaBoundedScheduler`。scheduler 在统一决策后，先通过
BlockPool移除旧Mamba prefix hashes，令后续hybrid lookup回退到仍完整的
较短prefix或重新prefill；再向每个worker发送仍有效的block ID集合。
worker只回收未被请求持有、没有可命中hash、也不是本步CoW source的state。
不能独立在worker删除LRU bytes，否则KV命中仍可能引用丢失的recurrent state。

所有Mamba group按相同scheduler快照处理。保留canonical hash与partial
aliases；attention hashes不受此预算限制。tracked metadata只扫描有界
checkpoint集合和请求Mamba表，不在每个decode step遍历全部attention KV。

primary pool与null slot、archive storage、FP32 recurrent精度及ACL graph
tensor addresses保持既有布局。回收复用前同步pending graph writers；没有
通过CPU snapshot实现回收。现有scheduler默认行为不变，新增接口是可选的。

## 下一次启动

沿用[完整runtime runbook](qwen38_310p_runtime_runbook.md)的CANN/OPP顺序。
在独立完整快照中同时带入以下文件的本轮变更，不覆盖其它已资格验证组件：

- `vllm_ascend/core/prefix_mamba_scheduler.py`
- `vllm_ascend/_310p/prefix_mamba_state.py`
- `vllm_ascend/_310p/model_runner_310p.py`中的可选snapshot处理

在image-enabled recovery launcher的`serve_cmd`中添加：

```bash
--scheduler-cls vllm_ascend.core.prefix_mamba_scheduler.PrefixMambaBoundedScheduler
```

仅支持310P Qwen4Exp、standalone、同步调度、align-mode prefix caching。
初始测试使用三个slots、MTP2、`[3,9]`、1024-token scheduler batch、一张图片。
实机验证后按用户要求切到2560-token batch与native HC residual，图片保持
启用；最终cold/repeat、transfer及thermal结果见上述双语报告。
其它profile见[恢复runbook](../../../../artifacts/qwen38-serving-gate-20261007/RUNBOOK.md)。
预算从既有compact primary pool扣除每请求四个speculative工作窗口：
three-slot/MTP2为每group最多27个cached checkpoints；four-slot/MTP2为15。
active和CoW窗口另保留。更老会话的prefix可能需要重算，不能宣称所有历史
都常驻或始终零spill，也不能由KV planner推断完整长窗口质量或吞吐。

## 验证与交付边界

host regression使用与线上相同的OpenSensor vLLM `3ab5dda29`源码，在隔离
CPU平台环境验证真实BlockPool的hash eviction、partial aliases、CoW保护、
IPC snapshot、group校验、slot复用与失败时保留metadata。
64项host checks通过；所有本轮文件通过scoped格式与静态检查。
三请求400步state隔离测试超过现有archive容量，旧hash无法再命中，最新
prefix值精确，且未触发host snapshot或restore；这不是NPU性能benchmark。

```bash
PYTHONPATH=/path/to/pinned-vllm VLLM_TARGET_DEVICE=cpu VLLM_PLUGINS='' \
python -m pytest --noconftest -q \
  tests/ut/_310p/test_prefix_mamba_scheduler.py \
  tests/ut/_310p/test_prefix_mamba_state.py
```

本轮新增diagnostic cleanup和真实CPU tier压力回归后，共78项host checks通过。
实机测试采用实际模型checkpoint tensors比较原策略与有界策略，单独记录
值一致性、storage addresses与spill/restore/retirement counters。该RPC不是
语言模型准确率或token throughput benchmark；实模型图片与并发推理另测。
实机通过范围以报告中的已完成结果为准，不能把启动成功当作推理通过。
不要通过清空live prefix cache来代替验证有界保留策略。
