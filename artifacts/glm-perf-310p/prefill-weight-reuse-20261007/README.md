# GLM prefill：L1 权重复用与半精度 route workspace

状态：三个候选已通过 CANN 编译和真实 NPU gates；**v903 组合已热切换到 8001**。
API PID 2826435、四 worker PID 2827551/2827949/2828351/2828744 及所有 weight storage
摘要不变，没有重新加载或重新准备权重。最大模型长度 311040、TP4、MTP1、max-seqs4、
640-token prefill 沿用原值。永久磁盘默认专家 bundle 仍为 v56，v903 为 resident 实验。
来源为[内存审计](../prefill-memory-audit-20261007/README.md)，该审计使用历史 trace。

## 在线结果

v903 全套 26 请求均有效；c1 **9.257 tok/s**，c4 总量 **14.149 tok/s**，
工具调用成功；质量 **17/20**。失败为 `instr_reverse` 的 `pail`、`instr_first` 的
`Red`、`code_slice` 的反引号 `lan`；前一轮已经记录同样的 reverse 不稳定和格式失败。
没有放宽质量门，也没有将本轮候选发布为永久默认。

两次清缓存、1280 输入 token、8 completion tokens 的 API 时长为
**23.950/24.124 s**；四 rank 每次均确认至少两次 640-token graph replay。
没有完成同期 v56 配对，因此不能把历史 26.398 s 与本轮值作为严格速度收益。
用户开始自行检查后，中断了待执行的配对 benchmark，finally 恢复 v903、确认
所有 rank 图完整、无 native failure/fallback、服务已 resume，之后不再发测试请求。

所有 rank 实际 scratch 均为 `[5120,4096]`、FP16、**41943040 bytes（40 MiB）**。
逻辑流量估计与实测 allocated workspace 大小分别报告；没有 HBM counter 证据。
原始在线请求和恢复/交接记录保留在 `measurements/`。

## 两个独立开关

`--prefill-weight-cache`：只在 prefill 且同一个 expert/output tile 有超过 15 行时启用。
先把该 tile 的全 K 权重转换为 packed INT4 并放入 L1，再处理各 15-row batch。
gate/up 各保留独立权重区和经过原 FP16 rounding 的 scale；down 保留一个区。
每次 kernel launch（含 graph replay）、每个 expert/tile 都重新准备，绝不把旧权重跨
请求缓存。少行及 decode 保留原路径，没有合并 block32 scale 或改变 FP32 加法顺序。

对于 640 行落在一个专家的示例，原来每个 projection/tile 准备 43 次权重，新分支
准备一次。此数字是静态循环次数，既不是实测专家分布，也不是 43 倍加速预测。
最大 gate/up L1 权重区为 512 KiB，down 为 256 KiB；另有 activation L1。
原有 UB 数组不增加：L1 cache 生效后已闲置的 `packedB_` 保存 up 的 FP32 scale。
L1 容量与 LoadData 地址布局已通过 310P 编译及真实权重配对门。
目前不支持与 decode lookup table 同时启用；builder 会提前拒绝该组合。

`--fp16-route-workspace`：down 已有 `FP32 accumulator → FP16` 边界。
原路径将该 half 值转回 FP32、乘 route weight、写入 FP32 workspace。
新路径直接保存同一个 half 值，reducer 再转 FP32、乘原始 route weight、按同一稳定
专家顺序累加。没有增加一次舍入，也没有将已乘权重的 FP32 值压到 FP16。
peer/零权重后缀仍先判 ownership，再读取 workspace。

640 × 8 × 4096 的 scratch 从 **80 MiB 降到 40 MiB/rank**；42 个 target MoE
层的 write+read 从 **6.5625 GiB 降到 3.28125 GiB/rank/chunk**，仅为逻辑字节模型。
没有消除整个 route round trip；没有测量 HBM 流量。在线 latency 另见上表。
最终输出与所有 route multiply/accumulate 仍为 FP32；小 decode 使用原 FP32 输出。

两开关默认关闭，可分别测试、再测试组合。half reducer 有独立入口
`glm_fused_reduce_half_v1`，增加 order/weights 参数；永久 loader 也传递 half workspace 选项；constructor 拒绝 dtype 与
编译 provenance 不一致的 bundle。manifest 要求六个 W2/W3/W4 × A4/A8 组合的
真实 multibatch gates，核对 half workspace 字节，不能沿用原 FP32 gate 报告。
同时修正 builder CLI：此前解析了 W3 两个选项，却没有把它们传入 build。

## 本地验证

- 定向 CPU UT：116 passed；涵盖 routing ABI、40/80 MiB scratch、已有 half boundary、
  稳定归约、配置拒绝、build/CLI flags 与 manifest。
- 隔离 review checkout 的 `tests/ut/glm_perf`：418 passed，沿用此前三个外围测试排除项。
- 共享工作区同范围：749 passed（修改 loader 前）、4 个既有 KDA header hash fixture errors，未改那些文件。
- Python lint/format、两个 C++ 文件 clang-format 与 scoped diff whitespace 检查通过。
  两个仓库级 hook 指向已有 logger/meta 未提交改动；隔离 review checkout 中均通过。
- 新 NPU 文件包含 42 个真实权重配对门：W2/W3/W4、A4/A8、2/15/16/17/30/31/640
  行；覆盖 changed input/route/code/scale、重复路由、零权重及 all-peer replay。
  v901（L1）、v902（half route）、v903（组合）各 **42 passed**，共 126 个配对门。
  另三个 build 各通过 30 个独立 arithmetic/replay + 12 个 real-weight 门。
  这些测试先 warm descriptor，不声称首次冷 capture 已验证。

## 重放与边界

从 serving 使用的源码 checkout 执行，先 source `/srv/ai/bin/ascend-env.sh`，使用既有
serving venv。不要停止服务、重新量化或重新加载全模型。独立门依次运行，不与在线
benchmark 同时争用 NPU。版本号、输出目录和 namespace 必须选择尚未存在的值。

```bash
python -m tools.glm_perf.build_reconstruction \
  --build-dir /tmp/glm-prefill-reuse-v901 --version 901 \
  --output-columns 128 --tile-pipeline --all-bits --fused-moe \
  --prepared-weight-layout --pair-scale-groups --wide-cube-k 128 \
  --pair-prefill-scale-groups --prefill-weight-cache
```

再用不同版本分别编译仅 `--fp16-route-workspace` 和两个开关同时启用的候选。
不要加入未经配对的 W3 specialization、FP16 SwiGLU 或其他实验。
对照采用既有、已签收的 v56 bundle，而不是重新编译一个不同 schedule 作对照。

```bash
python tests/e2e/nightly/310p/single_node/ops/test_prefill_memory_reuse_310.py \
  --baseline-build-dir /srv/ai/models/GLM-5.3-Flash-selective-W3-native-int4-310p/native-kernels-v56 \
  --candidate-build-dir /tmp/glm-prefill-reuse-v901 \
  --checkpoint /srv/ai/models/GLM-5.3-Flash-selective-W3-310p
```

随后使用 `fused_moe_probe.run` 生成与新 binary/helper hash 完全匹配的独立 arithmetic
及真实权重报告；新选项额外执行 31-token real gates。通过后才创建 manifest、用当前
resident 候选组合进行在线切换。配对门不替代全模型质量/tool gate。

在线比较保持 prompt/context/cache 条件相同，记录逐专家行数、640-token chunk 时长、
graph replay、c1/c4、worker PID、weight storage digest、fallback 和候选 provenance。
本轮三个候选都通过独立门，只有 v903 做完整在线 suite，未测 v901/v902 在线时长。
当前 v903 保持运行，用户正在自行检查；尚未达到 c1 10 tok/s。

第一次 paired runner 因把关闭的 lookup 选项传给老 v56 constructor，42 个用例在发射
kernel 前失败；修正为只传开启选项后，三个 build 各 42 passed。第一次 live smoke
通过后，suite 发现 source snapshot 漏打包 workloads.json，自动恢复了 v56；补齐文件
后重新应用 v903 并完成 26 请求。失败日志没有删除，也没有声称那两次 runner 失败是
算术失败。源 snapshot、已编译 helpers、provenance 与 corrected protocol 都保留。
