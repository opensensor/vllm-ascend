# GLM 永久舍入 scale 与离线融合候选

现有 expert kernel 的 `PrepareScales` 每次从 GM 读取 FP32 后都会执行
FP32→FP16→FP32，再参与原有 FP32 比例乘法与累加。本轮增加显式
`fp16_rounded_fp32_v1` 磁盘标记，将同一舍入步骤放到 CPU 导出阶段。
scale 保持 FP32 存储，不改量化位数、code 字节或 route 的累加顺序。

## 完整路径

- CPU exporter 从完整 native checkpoint 克隆独立目标目录，只重写每个
  rank/layer 的 scale shard；未变化的模型 shard 用硬链接共享。
  不解码、不重新打包 code，也不调用 NPU。中断时保持 complete=false。
- 原 loader 直接读取新 FP32 scale，校验配置/manifest/构建标记和 shard
  checksum，向目标及 draft 的 bank 写入 host layout 属性。
  没有加载后、capture 中或 forward 中的 scale 转换。
- permanent method 与 resident wrapper 都拒绝将未标记的旧 bank 发给
  跳过舍入的 kernel；不能直接把此候选热换到旧 FP32 scale bank。
- gate/up 和 down 的构建选项配套，删除每次 scale 准备中的两个 vector
  Cast 与其同步步骤。scale scratch 从 4096 降到 2560 字节，节省
  1536 字节/kernel instance；GM scale 字节数保持不变。
- 同时补齐 checkpoint bundle 对 routed-input producer 的复制与 hash
  校验，防止永久 prefill bundle 缺失 pipeline 的一端。
- 独立算术门和性能 profiler 在计时外准备 CPU scale fixture；真实模型
  路径要求直接从磁盘加载。不得将 fixture 准备冒充生产路径。

## 离线证据

v956 在 v952 decode 排程上启用新标记，v957 在 v953 prefill 排程上启用。
两者均由 CPU SDK 成功编译为 dav-2002，没有选择设备、加载 kernel、
提交推理、恢复服务或热替换。真实 checkpoint 没有被转换或改写。
两个候选继续保留上一轮的 native-column stable route reduction。

兼容本机依赖的 GLM CPU 测试 1524 项通过；assembly、grouped gate/up、
KPool ops 三个需要本机缺失 upstream/NPU 模块的测试文件仍排除。
新增验证覆盖所有有限 FP16 位模式、signed zero、舍入 ties、overflow、
interrupted export、共享原始 code inode、权威 scale index、实际 loader
iterator、checksum 破坏和 raw-bank 拒绝。scoped hooks 通过。
编译与 CPU oracle 不能证明 NPU Cast 的舍入行为相同，也不能证明加速。

压缩包保存 44 个构建/源码/日志文件；manifest 提供逐文件 SHA256。
编译时的 source snapshot 保留原样，用于解释 provenance 的 source hash。

## 后续命令（尚未运行）

用户要求继续离线；下面只记录重新获准 NPU 工作后的步骤。

```bash
# CPU 导出到新目录；本轮只运行了小型测试 fixture，没有转换实际模型。
python -m tools.glm_perf.prerounded_scale_checkpoint \
  --source /path/to/permanent-native-checkpoint \
  --output /path/to/new-rounded-checkpoint \
  --kernel-bundle "$ROOT/build-v956"

# 完整 MoE baseline/candidate，变化输入 graph replay，逐位比较与交替计时。
python -m tools.glm_perf.route_columns_probe \
  --baseline "$ROOT/build-v952" --candidate "$ROOT/build-v956" \
  --feature prerounded_weight_scales \
  --output rounded-decode-pair.json --allow-device-gate
python -m tools.glm_perf.route_columns_probe \
  --baseline "$ROOT/build-v953" --candidate "$ROOT/build-v957" \
  --feature prerounded_weight_scales \
  --output rounded-prefill-pair.json --allow-device-gate
```

ROOT 是远端 `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007`。
必须先通过 W2/W3/W4×A4/A8×2/8/17/640 tokens 的完整 pipeline 位级门，
再执行原有 real-weight binary gates、加载新 checkpoint、验证 target/draft
graph replay，以及冷 prefill 和 c1/c4 的端到端配对。没有吞吐或 TTFT 新数据。
独立 probe 的 manifest 字段 complete 只表示导出或探针完成，不表示真实
模型已通过性能与质量门。语言输出质量由用户检查。

US English 摘要见 [README.en.md](README.en.md)。
