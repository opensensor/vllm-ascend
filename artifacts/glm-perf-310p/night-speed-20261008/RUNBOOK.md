# GLM 310P3 性能实验运行手册

本手册对应用户已有 NPU host 与端口 8001；不使用通用 A2/A3 Docker 路径。
英文结果见 [README.en.md](README.en.md)，中文报告见 [README.md](README.md)。

## 当前配置

- TP4 / MTP1、四个 310P3 chips，4 request slots，640-token chunk。
- 配置 context 311,040；本轮仅验证至 6,400-token prompt。
- decode full graphs 2/8；640 prefill 为 46 segments、45 eager boundaries。
- permanent FP16 scales，INT4 Cube、FP32 accumulation，safe KDA gate。
- API `http://192.168.53.187:8001`，模型 `glm53-flash-selective-w3`。
- runtime `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/runtime-fp16-scales-20261008`。
- 当前 checkpoint `/srv/ai/models/GLM-5.3-Flash-native-int4-fp16-scales-20261008`。
- 当前日志 `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/gated-key-reuse-server-20261008.log`。
- API PID 2275553；engine PID 见当前 process / log；具体 process identity 见
  `measurements/gated-key-reuse-server-process-20261008.json`。
- KDA vendor `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/night-kda-gated-key-reuse-full-v2/opp-gated-key-reuse/vendors/custom_transformer`。
- target/draft activation 均 A4。未保留 draft A8 或额外 W4 down7 memory。

启动时先 `source /srv/ai/bin/ascend-env.sh`，不要在它之前启用 `set -u`。
Python 使用 `/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python`。
未升级任何依赖。

## 源码与 native gate

默认 production header 不变。将已合格 source 复制到新目录，用 staging tool
与明确的 parent SHA 生成新 header；例如：

```bash
python -m tools.glm_perf.stage_kda_beta_scale_vector \
  --source /path/to/qualified-parent.h \
  --destination /path/to/NEW-candidate.h \
  --expected-sha256 ac922b4b01240a51757f0cf0ee6fa0d1b6937373e346e12318deb07945f36740 \
  --report /path/to/stage.json
```

完整 build protocol 位于 `protocols/glm-night-build-kda-beta-full.py.txt`。
在既有 `GLM_KDA_SCORE_CACHE_COLUMNS` 编译选项后添加
`GLM_KDA_VECTOR_BETA_SCALE`；再使用合格的 score reduction、tail W/U 与
`GLM_KDA_GATED_KEY_REUSE` build。所有 stage 需匹配其明确 parent SHA。score-reduction 与 tail W/U 各使用独立 staging
工具及新 source/package 目录；不覆盖已加载 manifest 或 binaries。tail-cache
完整 gate 失败，工具标为 `serving_eligible=false`，不得装入服务。

独立 diagnostic 使用 `build_kda_beta_scale_vector`；其 compiler 参数是既有
`compile-reconstruction` wrapper，不是直接传 `bisheng`。fresh v978 probe 需显式
`--allow-device-gate`。完整 operator gate 使用已归档 fixture，比较 12 个返回值、
recurrent state、重复执行及输入不变。不能以 compile-only 或独立 row kernel
通过代替完整 operator / serving 结果。

## 替换及启动

最终 memory cleanup restart / restore 与 frozen launch 见：

- `protocols/glm-night-gated-key-reuse-restart.py.txt`
- `protocols/glm-night-gated-key-reuse-restore.py.txt`
- `protocols/gated-key-reuse-launch.py.txt` / `gated-key-vector-candidate.py.txt`

先检查新 package 的 SHA 与完整 safe-gate JSON。drain 服务，仅终止 process
receipt 的 PID/create-time 对应 GLM 子树；保留无关进程。替换原 `opp-score-cache`
vendor slot，保持其它 OPP 与 `LD_LIBRARY_PATH` 的 host libraries 不变。
process JSON 与 launch protocol 保存完整 direct `vllm serve` 参数，不能简化掉
resident middleware、loader 或旧 hf overrides。

冷启动后加载合格的八个 native resources，再应用完整 frozen combined source、
重新捕获 target/draft decode 与 640-token prefill graphs，最后 resume。
检查四 rank `graphs_dirty=false`、`native_failed=false`、FP16 scale dtype、每 rank
43 banks / 152,174,592 scale bytes，以及各融合 native counters。
source-only hot swap 验证 worker PIDs 与 weight-storage digests 不变；OPP restart
会更换 PIDs。不要在服务中重新 repack 全模型。

```bash
curl --fail http://192.168.53.187:8001/health
curl --fail http://192.168.53.187:8001/is_paused
curl --fail http://192.168.53.187:8001/v1/models
curl --fail http://192.168.53.187:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash-selective-w3","messages":[{"role":"user","content":"Explain why plants need sunlight."}],"max_tokens":512,"temperature":0}'
```

检查实际请求正常结束，而非仅 startup ready。语言质量由用户判断。
此模型 tutorial 指出 `enable_thinking=false` 不受支持，不以该参数判断质量。
本轮未启用 eager 或 TorchDynamo isolation fallback。

## 测量与恢复

`protocols/glm-night-vector-bench.py.txt`、`glm-night-kda-beta-bench.py.txt` 保存
完整 TTFT / generation / C4 方法与 cold-prefix 清理。直接文本与 prefix 请求见
`glm-night-real-prefix.py.txt` / `glm-night-kda-beta-real-prefix.py.txt`。synthetic
prompt 的约 10 tok/s 不能替代真实文本结果。

Profiler controller 使用 direct CANN 生命周期，停止/finalize 后才恢复候选。
profiles 中包含 rank-zero 图表及四 rank JSON attribution；原始 traces 保留在
远程。trace 的 task sums 不能作为 wall critical path。

保持最后一个合格完整候选 unpaused。未来实验失败时，使用已归档 source 与
新 generation ID recapture；不要复用部分失败的 graph。OPP 恢复也需要 drain
并重新冷启动，同时恢复所有 native resources。

## 最终验证

最终 serving 与三次 C4 repeat 见 `protocols/glm-night-gated-key-reuse-bench.py.txt`；
直接文本与 1,280-token prefix replay 见 `glm-night-gated-key-reuse-real-prefix.py.txt`。
部分 tail 的比对使用 `tail-before-serving.json` 与 `tail-vector-serving.json`；
新 CANN stage 4 attribution 见 `profiles/vector-tails-kda-stage-attribution.jsonl.gz`。

CPU 检查日志 `measurements/glm-night-cpu-final-tests.log.txt`，1,833 passed；
scoped hooks 日志随 final evidence 保存。全仓 `format.sh ci` 原有问题见英文
报告，不将其描述为通过。使用 signed scoped commit，不暂存并行 Qwen 修改。

最后替换的是 `night-kda-gated-key-reuse-full-v2`；额外 16 KiB UB 缓存同一
score row 的 gated keys，保留独立 score passes。完整 17-case gate 全部通过，
不能改用未测试的初始 reuse build。之前 `final-night-speed-*` receipts 对应
smaller-checkpoint / tail-vector 选择过程，当前执行路径以 gated-key receipts 为准。

最终 trace 的 task sums 与 stage 顺序 attribution 见
`profiles/gated-key-all-op-attribution.jsonl.gz` 和
`profiles/gated-key-kda-stage-attribution.jsonl.gz`；current health / package / source
验证在 `measurements/running-state-final.json`。原始 traces 仍留在远程。
real-text rate 另计算到完整 request completion，避免遗漏最后 visible text 后的
耗时；推导记录在 `measurements/full-generation-timing-summary.json`。
