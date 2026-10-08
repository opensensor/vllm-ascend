# Qwen有界checkpoint实机验证与图片服务

[US English report](REPORT.en.md)提供相同结果与限制。

## 初始1024-token配置

用户10月8日重新授权NPU测试。完整recovery snapshot复制到
`/srv/ai/src/qwen38-prefix-bounded-runtime-20261008`，回移已签署commit的
有界scheduler、tier helper及runner `_update_states`方法和对应import。
四个已资格验证model源码保持byte-identical；vLLM固定`3ab5dda29`，
没有升级依赖。使用实际W4权重、四个310P device、TP4/EP4、MTP2、
`[3,9]` decode graphs、三个slots、262144最大context、一张图片、
built-in FP16 SwiGLU与CANN finalizer。FlashComm1沿用关闭配置。
没有使用dummy weights；端口8001，Kilo配置未改。

## 验证结果

78项focused host regression全部通过。本轮code、scripts、runbooks的
scoped pre-commit hooks通过。隔离worktree运行`bash format.sh ci`，
被既有无关Ruff、codespell、typos、clang-format、markdownlint和
forbidden-import问题阻止；自动修改仅留在该worktree。最终全量日志为`full-format-ci-complete.log`。
原始terminal/lint记录保留既有拼写与carriage returns；codespell和typos
会标记其中复制的历史lint诊断与opaque IDs，仅raw evidence跳过这两个
spelling hooks。本轮全部source和reports的这两个hooks均通过。

实际模型text/tool smoke为6/7，与此前baseline一致；已知ASCEND反转
应为DNECSA，仍输出DNESCA。其它五个文本项与tool call通过。
全新红色、蓝色和DEMO42 OCR图片三项均通过；这不是全面准确率验收。

独占测试服务drain并清理scheduler及worker cache后，使用实际模型
checkpoint tensors，执行96轮、每轮三个changing state IDs。
两种策略都保持最新state值与primary/archive/swap地址；baseline还
验证最早spill state恢复后的值。四rank均返回确认。

| Diagnostic策略 | 全rank/group spill | restore | 最慢rank计时 |
| --- | ---: | ---: | ---: |
| Baseline保留 | 495 | 12 | 2.187 s |
| 有界回收 | 0 | 0 | 0.739 s |

每group checkpoint为9744384 bytes；baseline累计spill payload
4823470080 bytes，即4.492 GiB。有界策略回收3096 states。
计时不含correctness probes；counter包含最早state restore probe及其
admission操作。仅一对有序microbenchmark，四rank并非独立重复样本。
不能把此结果当作LLM token throughput提升或thermal因果证明。

256-token冷prompt生成256 tokens，decode为27.412 tok/s。
三个独立16384-token冷prompt各生成128 tokens；server prefill分别
43.712、51.023、50.422秒，排除queue，约375、321、325 prompt tok/s。
client TTFT为43.735、94.756、142.439秒，后两条包含等待。
decode分别1.258、2.388、17.707 tok/s，前两条与其它冷prefill重叠。
整批149.619秒完成；每rank新增spill/restore均为0，三group合计
新增retirement 194，weights和clean graphs保持不变。

用户要interactive使用时终止16K repeat批次，不能计为完成的prefix复用
验收。mixed-image/decode probe已准备但未执行。初始phase最高温80°C，
没有thermal shutdown，96°C watchdog保持启用。这不是持续thermal或
三个完整256K窗口的资格验证。

## 历史比较与用户要求的切换

历史native HC residual三次8192-token冷样本median 19.973秒，
约410 prompt tok/s；后续23410-token residual control median
59.400秒，约394 prompt tok/s。因此初始image recovery profile
没有证明仍是最快long-prefill。长度、并发及设备初始条件不同。

用户要求live切到更接近该配置且保持图片。replacement使用2560-token
scheduler batch与native HC residual，保留三个slots、MTP2、`[3,9]`、
有界checkpoint及图片。batch输入buffer按startup容量分配，需一次
engine restart；随后HC用既有resident load/switch/recapture事务选择。
完整graph/KV-budget热重配置尚未实现。最终资格结果与服务状态见下方。

## 最终image-enabled配置

2560-token scheduler batch成功加载；resident controls选择
`native_hc_residual`。native library和kernel与已资格验证SHA256
manifest一致；四worker native值probe通过，切换过程中PID和
weight-storage digest保持不变，两个graphs重新捕获且status clean。
首次controller因generation非UUID而在修改Python dispatch前拒绝，
修正后安全复用已加载native资源，完成切换。

实模型smoke仍为6/7，失败仍是同一反转问题。红色、蓝色与OCR图片
再测3/3通过。23410-token prompt使用历史
`qwen-latest-matched-23410-v1`标签。server冷prefill **59.688秒**，
约**392.2 prompt tok/s**，cached=0；client TTFT **59.716秒**。
立即repeat命中**23296 tokens**；server prefill **1.900秒**，
client TTFT **1.917秒**。接近历史59.400秒median，没有证明新纪录。
cold/repeat output hashes不同，不是exact output parity gate；历史
unchanged-runtime trial同样有text drift。

这对请求每rank新增spill/restore均为0，三group合计retirement 49，
graphs clean且weights未改。此phase最高温**76°C**，测试结束
74/73/76/75°C，无thermal shutdown。没有验收持续thermal、完整窗口
质量、三个并发256K或最终配置的并发冷prefill吞吐。handoff后不再提交
测试负载。

服务留在running/unpaused供用户使用：
`http://192.168.53.187:8001/v1`，model ID
`qwen38-prefix-bounded-validation`，watchdog 96°C保持启用。
历史diagnostic脚本需要独占/drained测试服务；当前interactive使用时
不要运行其cache-clearing操作。

复现使用`start-fast-image.sh`，随后`select-hc.py`选择native HC，
再执行`validate.py`图片和cold-prefix checks。
主要receipt为`fast-image-results.json`、`fast-prefill.json`、
`fast-image-hc-switch.json`、`fast-image-native-load.json`、
`fast-image-vision.json`、`fast-image-smoke.json`、`runtime-provenance.json`
和两份thermal记录。本轮交付diagnostic worker RPC、四项cleanup/pressure
回归、双语runbooks、启动/切换脚本与验证记录。无关既有实验未纳入；
此前已发布的HF图片能力model cards本轮未更改。
