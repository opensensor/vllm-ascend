# Qwen 310P 实机复测

主机：`matteius@192.168.53.187`。使用完整快照
`/srv/ai/src/qwen38-demo-recovery-runtime-20261008`，不混用主工作树中的
GLM 修改。模型是本机真实 W4 权重，TP4/EP4、MTP2、1024-token scheduler batch；expert chunk cap2560。

## 最终演示配置

当前保留 `[3,9]`：C1 使用3-token图，C2填充到9，C3使用9-token图；
最多三个 active requests。此前四请求实例触发热保护后已退出。
不要同时添加第三个图尺寸；已有 310P HCCL event-id 限制。
实际模型 ID 为 `qwen38-flashnext-latest-gate`，API 为
`http://192.168.53.187:8000/v1`。录制期间保持当前进程，不复跑 benchmark。

确认旧服务和客户端已退出、NPU 空闲后，在 SSH 会话启动：

```bash
qwen_test_dir=/home/matteius/experiments/qwen38-serving-gate-20261007
PORT=8000 \
SERVED_MODEL_NAME=qwen38-flashnext-latest-gate \
KV_CACHE_FRACTION=0.70 \
QWEN38_AFFINITY_HELPER="$qwen_test_dir/affinity-port8000.py" \
LOG="$qwen_test_dir/server-recovery.log" \
WATCHDOG_LOG="$qwen_test_dir/watchdog-recovery.log" \
AFFINITY_LOG="$qwen_test_dir/affinity-recovery.log" \
nohup bash "$qwen_test_dir/start-demo-recovery.sh" \
  > "$qwen_test_dir/launcher-recovery.log" 2>&1 < /dev/null &
```

启动脚本保留原硬件环境和完整 OPP 顺序，去掉固定 KV budget 以执行新 profiling。
当前 receipt：997,292 cache tokens、3.80 × 256K planner 容量、三个 active
requests；rank0 archive194slots、post-capture free9.30GiB、reserve4.00GiB。允许一张图片/prompt，最多 1,048,576 pixels。
用户自行进行图片测试，agent 未运行图片语义或完整长窗口压力测试。
原 affinity helper 只接受端口 8001；本目录的副本只将身份校验端口改成
8000。目标/草稿加载、图捕获和 HTTP 生成都完成后才能验收。

## 可选驻留 residual 候选

以下是此前 text-only A/B 的可选候选，不是当前图片实例已应用的配置。
每次重新启动后，都需重新加载并验证 native resource，再独立切换候选：

```bash
qwen_python=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
export PYTHONPATH=/srv/ai/src/qwen38-demo-recovery-runtime-20261008
qwen_previous=/home/matteius/experiments/qwen38-decode-next-20261005
"$qwen_python" -m tools.glm_perf.resident_harness \
  --base-url http://127.0.0.1:8000 load-native \
  "$qwen_previous/native-hc-residual/native-v1.json"
"$qwen_python" -m tools.glm_perf.resident_harness \
  --base-url http://127.0.0.1:8000 switch --mode graph --recapture \
  --candidate /srv/ai/src/qwen38-demo-recovery-runtime-20261008/tools/qwen4exp/resident_candidates/native_hc_residual.py \
  --candidate-name native_residual
curl -fsS http://127.0.0.1:8000/v1/models
```

冷前缀测量前使用 `tools.qwen4exp.resident_reset`，同时清理 scheduler
prefix cache 和 worker Mamba checkpoint tiers，并确认 `cached_tokens=0`。
反复发送同一前缀时，只有响应或 cache 指标证明命中，才称为暖缓存。

原 `[3,6]`、四请求 `[9,12]`、此前 `[3,12]` 和当前 `[3,9]` 必须分别统计。前者的 C4 eager
警告是实际性能限制，不能标作 C4 图测试通过。完整 A/B 在同一次权重
加载中切换并重捕获；各轮 PID 和 weight-storage digest 必须一致。

`run_gate.executed.txt` 保存首轮实际执行代码。`replay_gate.py` 是后续
重放版本，显式传递 client，并在计时外等待 11 秒，使周期统计发布完成。
`run_c4_gate.py` 使用后者。复跑须使用新的证据目录，避免覆盖历史结果。

当前 resident harness 支持候选算子切换，不支持 graph profile 或 KV budget
热切换。这些调整不能仅修改 worker 字段；需同步 engine scheduler、cache
planner 与所有 rank 的 cache/capture。该后续功能尚未实现或实机验证。

`query_lens_capacity` 是独立 metadata 热修复：扩大旧 snapshot 的 CPU
qLens buffer 后再写入，避免 4→12 行的 out-view 自动 resize。复现及应用
证据见 `metadata-capacity-*.json`；切换会 drain、清理 prefix cache、重捕获
图并保留 worker PID/storage。当前 recovery 快照已在磁盘中包含同等修复，不需再次切换此候选；
切换会清空 prefix cache 并影响在途会话，不要为只读检查调用 reset/switch。
