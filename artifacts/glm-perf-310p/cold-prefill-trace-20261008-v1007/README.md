# GLM cold-prefill trace and candidate tradeoffs

Historical snapshot: pending-validation statements and test counts below are
superseded by [the current bilingual handoff](OFFLINE_NEXT.md).

历史快照：下文待验证状态及测试计数已由[最新中英文交接](OFFLINE_NEXT.md)更新。

## US English summary

The four-rank trace covers one cold 6,400-token request and its first output
token. It is attribution evidence, not an unprofiled speed comparison. Rank 0
spent 23.049 seconds in native expert tasks; its device stage included
10.671 seconds of communication without overlap. Task durations can overlap
and must not be added as independent potential speedups.

The trace recorded a 480.627 MiB `aten::bmm` temporary in the final standalone
mHC mixer and approximately 220 MiB matmul temporaries. Allocations preceding
the trace are absent; each rank has 38 incomplete memory records. These records
do not establish the complete model footprint.

The opt-in fused expert scale-accumulation candidate v1008 passed 30 synthetic
and 12 loaded-expert cases, including changed-input graph replay. Its matched,
unprofiled cold requests averaged 52.105 seconds versus 51.418 seconds for
v1001, approximately 1.34% slower. It removes one vector operation and barrier
but retains the allocated scratch. It therefore has no demonstrated allocated
memory benefit. The qualified v1001 dispatch was restored after this trial.

The prepared native mHC projection v1011 is a separate candidate. It uses
FP16 inputs and prepared weights with FP32 Cube accumulation and output. Its
K=1,024 tile uses 198 KiB of declared UB buffers per core, plus SDK overhead,
and each prepared FN weight occupies 1 MiB. This replaces a whole-batch
projection cast/layout path with bounded tiles. Standalone arithmetic through
1,280 rows and changed-input graph replay passed. Loaded-weight epilogue gates
accepted 81–82 weights per rank with a limit of two FP16 ULP. These gates do
not establish language quality.

The full-model projection trial deferred when a serving request became active;
no benchmark requests completed in that trial. Net peak-memory savings and
end-to-end latency remain unmeasured. A lower peak-memory result is useful
even if latency is flat or slightly slower: it could make a larger prefill
chunk feasible. That conclusion requires measurements including prepared
weight storage and a new capacity analysis. No larger-chunk feasibility or
prefill-graph speedup is claimed here.

Raw CSV tables and their hash are recorded in `SUMMARY.json`. The trace stays
on the NPU host under `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/`.

## Maintenance and restart qualification

The r4 server failed after a recovery graph capture reset worker request state
while new public requests were arriving. The observed exception was a missing
request ID in `GPUModelRunner._update_states`, not a reported kernel arithmetic
error. Public ingress must be held out throughout a resident experiment.

The resident middleware now exposes loopback-only `/resident/maintenance`:
POST enables the gate, GET reports it, and DELETE reopens public inference.
During maintenance, new non-loopback completion requests receive HTTP 503
with `Retry-After: 30`; health and loopback benchmarks remain available.
After enabling the gate, finish draining existing requests before any worker
reset or graph capture. Never issue a competing resume while another
controller is waiting for that drain. Clear the gate only after all four
workers acknowledge complete graphs and the server has resumed.

The r5 restart uses the same checkpoint, context and memory budgets. It reran
the real-weight qualification on all new worker PIDs, accepting 84–85 branches
per rank. Previous process-specific qualifications are not portable across
restarts. The current local GLM suite passed 1,795 tests; this includes the
maintenance gate and the offline final-mixer contracts.

The next final-mixer build has an explicit `--no-state-rounding` option and a
separate `NativeMhcFinalPost` helper. It must preserve unrounded FP32 output;
the existing intermediate mixer retains its state-rounding behavior. CPU
dispatch/build tests pass. Device and full-model validation remain pending.

### 中文：维护与重启验证

r4 恢复过程中，新公开请求进入了 worker reset/图捕获阶段，随后在
`GPUModelRunner._update_states` 出现请求 ID 缺失。日志未报告内核算术错误。
实验期间必须持续阻止新的公开推理请求，并先等待已有请求完成。

新增 loopback-only `/resident/maintenance`：POST 开启，GET 查询，DELETE
恢复公开推理。维护期间外部 completion 请求返回 HTTP 503 和 `Retry-After: 30`，
健康检查与本地 benchmark 可继续使用。不得在另一个控制器等待 drain 时并发 resume。
必须等四个 worker 确认图完整并恢复运行后才关闭维护 gate。

r5 保持相同 checkpoint、上下文及内存预算，已在新 worker PID 上重新执行真实权重
检查，每卡接受 84–85 个分支。旧进程的验证不能跨重启复用。本地 GLM 测试
1,795 项通过，包含维护 gate 和下一版 final mixer 的离线契约测试。

下一版 final mixer 使用显式 `--no-state-rounding` 与独立 `NativeMhcFinalPost`，
保留未舍入的 FP32 输出；中间 mixer 的舍入语义不变。CPU 测试通过，设备和完整模型
验证尚待完成。

## 中文摘要

四卡分析覆盖一次冷启动的 6,400-token 请求及首个输出 token，属于性能归因，
不是无分析器开销的加速测试。Rank 0 的原生专家任务耗时合计 23.049 秒，
设备阶段包含 10.671 秒未重叠通信。任务可能重叠，不能将它们相加作为可实现的加速。
最终独立 mHC mixer 出现 480.627 MiB 的 bmm 临时分配；matmul 临时分配约
220 MiB。分析开始前的分配缺失，每卡有 38 条不完整内存记录。

v1008 融合专家 scale 累加通过了 30 个合成用例、12 个真实专家用例及输入变化的
图重放检查，但冷请求平均 52.105 秒，基线为 51.418 秒，约慢 1.34%。它减少一条
向量操作和一个 barrier，但未减少已分配 scratch。测试后已恢复 v1001。

v1011 原生 mHC projection 是另一候选：使用 FP16 输入和预处理权重，保留 FP32
Cube 累加与输出。K=1,024 时每核声明 UB 为 198 KiB，另有 SDK 开销；每份 FN
预处理权重占 1 MiB。它以有界 tile 代替整批 projection 的 cast/layout 路径。
独立算术测试覆盖至 1,280 行，图重放通过。真实权重 epilogue 检查每卡接受
81–82 份权重，限制为两个 FP16 ULP；这不代表语言质量通过。

完整模型测试因服务请求活动而延期，该轮尚无完成的性能请求。净峰值内存收益及
端到端耗时仍待测量。即使延迟相近或略慢，降低峰值内存也有价值，可能使更大的
prefill chunk 可行；必须把预处理权重计入并重新进行容量分析。本报告不宣称更大
chunk 或 prefill graph 已得到验证。
