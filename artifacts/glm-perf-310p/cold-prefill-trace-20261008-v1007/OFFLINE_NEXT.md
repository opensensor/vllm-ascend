# GLM offline optimization handoff

## US English summary

Cold prefill remains the priority: the qualified 6,400-token baseline is about
51 seconds. The trace attributes 23.049 seconds of summed native expert tasks
to rank 0, with vector/scalar work dominant; these durations may overlap.
The final-mixer operator gain did not resolve the full-model bottleneck.

The second offline candidate, `--cache-expert-ends`, loads cumulative expert
boundaries into retained mask scratch once per launch, replacing per-output-tile
scalar GM reads. It adds no UB allocation. On M32, cached boundaries start at
byte 24,704 after reconstruction and row indices; even 288 INT64 entries end
at byte 27,008 within the existing 32 KiB buffer. DMA reads only aligned valid
entries, with at most three scalar tail loads. Decode keeps its existing reads;
prefill reloads metadata on every graph replay. Arithmetic and route order are
unchanged. Host C++ layout tests cover M16/M32 and every expert count 1–288.
Matching synthetic and real-weight changed-route replay gates are required
before admission. Compilation, 310P execution and performance remain untested.

The new offline candidate adds a density threshold to NZ MoE accumulation.
M32 carries at most 31 expert rows in the current route ABI. Setting
`--nz-prefill-accumulator --nz-prefill-min-rows 31` selects NZ accumulation
only for full usable batches; smaller tails retain row accumulation. Zero
preserves the old 17-row threshold. Prepared descriptors, weight packing,
route buffers and arithmetic order are unchanged by this selection.

Build provenance and admission checks require matching thresholds plus
synthetic and real-weight changed-input replay just below and at the boundary.
The candidate is **not compiled or NPU-qualified**. Earlier unrestricted NZ
candidate v920 was slower; no speed gain is claimed for this narrower selection.
The current GLM CPU suite passed **1,861 tests**, with six environment warnings,
using `python -m pytest --confcutdir=tests/ut/glm_perf -q tests/ut/glm_perf`.
The ordinary repository conftest cannot load the local incomplete vLLM FLA
package; the isolated tool suite bypasses that unrelated NPU setup.

Scoped manual pre-commit checks pass. The required whole-repository
`bash format.sh ci` ran in an isolated snapshot and fails on existing archived
Python lint, Markdown and forbidden-import errors outside this delivery.
Only owned-file formatting was carried back. Raw profiler kernel identifiers
were preserved through three exact typos exceptions; the evidence was not edited.

### Memory evidence

[The allocator audit](memory-peak-audit.json) reads archived CSV only. This
archive contains one rank-0 allocator table; four-rank kernel summaries do not
imply four memory tables.

| Recorded quantity | MiB |
| --- | ---: |
| Peak allocated memory, during a large bmm | 41,023.808 |
| That bmm temporary | 480.627 |
| Peak outside four complete large-bmm lifetimes | 40,937.075 |
| Ideal peak reduction after removing those buffers with timing unchanged | 86.732 |
| Matmul temporary at the remaining peak | 220.002 |

Replacement buffers, scheduling changes, static caches, fragmentation and graph
memory are excluded from that counterfactual. Twenty incomplete allocation/free
lifetimes are never subtracted; allocations preceding capture are missing.
Allocated and reserved memory are distinct. This is not a larger-chunk capacity
qualification. The next memory target is the 220 MiB matmul call site, which
requires source attribution before adding another prepared-weight cache.

```bash
python -m tools.glm_perf.memory_peak_audit \
  artifacts/glm-perf-310p/cold-prefill-trace-20261008-v1007/cold-prefill-v1007-tables.tar.gz \
  --output /tmp/glm-memory-peak-audit.json
```

### Earlier hardware trials

| Candidate | Cold 6,400-token result | Other evidence | Disposition |
| --- | --- | --- | --- |
| v1008 scale FMA | Mean 52.105 s versus paired baseline 51.418 s | Synthetic/loaded-expert replay gates; same allocated scratch | Opt-in, about 1.34% slower |
| v1011 tiled projection | 52.050 / 52.773 s versus baseline 51.236 s | Serving native-call counters verified; additional static weights 90 MiB/rank; peak increased | Opt-in |
| v1012 unrounded final mixer | 51.609 / 51.577 s | 1,280-row operator about 6.7x faster and 480.627 MiB less temporary allocation; model peak largely unchanged | Opt-in; decode discrepancy unresolved |

Saved evidence: [projection serving](mhc-projection-v1011-serving-r6.json),
[final-mixer operator gates](final-mixer-v1012-gates.json). An earlier projection
trial had zero native calls and is excluded from performance conclusions.
Final-mixer serving/control reports remain on the NPU host as
`final-mixer-v1012-serving.json` and `final-mixer-v1012-decode-control.json` under
`/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/`; they were not retrieved
during this offline phase.

Final-mixer decode controls returned 9.103 / 9.447 tok/s and different text after
explicit worker-cache zeroing. Baseline controls returned 9.786 / 9.814 tok/s
with identical text. The final mixer does not select its native path for those
small rows; a binding/graph-state interaction remains unresolved. Zeroing is
a diagnostic, not a serving workaround. Repeated-character prompts do not
validate language quality.

The last confirmed service state is the qualified r5 baseline: v1001 prefill,
v984 decode, scheduler budget 1,280, context 131,072, TP4, MTP1, decode graph
sizes 2/8, KV allocation 2.25 GiB/rank. Projection caches were cleared, the final
mixer disabled, health 200, unpaused, public inference open. **No NPU test or
server operation occurred in this offline phase.** Larger chunks and prefill
graphs remain unqualified; partial-prefix output divergence is unresolved.

Lifecycle fixes gate external completions during maintenance and invoke the
installed successor's preparation hook before capture. Existing requests must
drain before worker reset. Fresh r5 real-weight checks accepted 84–85 branches
per rank; qualifications cannot be reused across process restarts.

## 中文摘要

冷 prefill 仍是重点：合格 6,400-token 基线约 51 秒，rank 0 原生专家任务累计
23.049 秒，vector/scalar 工作占主导；任务可能重叠。final mixer 单算子收益未
解决完整模型瓶颈。第二个离线候选 `--cache-expert-ends` 每次 launch 将累计
专家边界读入保留的 mask scratch，替代每个输出 tile 的标量 GM 读取，不新增 UB
分配。M32 缓存从字节 24,704 开始，288 个 INT64 结束于 27,008，位于既有
32 KiB 缓冲区内。DMA 不越界，最多三个尾部标量读取。decode 保留旧路径，prefill
每次图重放更新缓存。算术和路由顺序不变。主机 C++ 布局测试覆盖 M16/M32 和
1–288 专家；准入要求合成及真实权重的路由变化图重放。尚未编译、310P 验证或测速。

新增离线 NZ MoE 密度门槛候选。M32 当前路由 ABI 最多容纳 31 个专家行，
`--nz-prefill-accumulator --nz-prefill-min-rows 31` 仅对完整可用批次启用 NZ
累加，尾批保留行布局；默认零保持旧版 17 行门槛。描述符、权重布局、路由缓冲区
及算术顺序不因门槛选择改变。构建记录和准入检查要求门槛一致，并要求合成及真实
权重在门槛前后完成输入变化图重放。候选尚未编译或 NPU 验证；旧版 v920 较慢，
本次不宣称加速。隔离 GLM CPU 测试通过 1,861 项，有六条环境警告。普通仓库
conftest 因本地 vLLM FLA 包不完整无法加载，隔离工具测试绕过该无关 NPU 初始化。
改动文件的 manual pre-commit 检查通过；隔离快照运行全仓库 `bash format.sh ci`
仍因既有归档 Python、Markdown 和禁止导入错误失败。仅同步本次文件格式，未修改
并行 Qwen 工作；通过三个精确 typos 例外保留原始 profiler 内核名称，证据未改写。

归档仅含 rank-0 内存表。峰值分配 41,023.808 MiB，其中 bmm 临时分配为
480.627 MiB；四个完整大 bmm 生命周期以外仍有 40,937.075 MiB 峰值。
保持时序不变并理想删除这些缓冲区，峰值仅降低 86.732 MiB。剩余峰值处 matmul
临时分配 220.002 MiB，应先定位调用点再考虑缓存。20 条不完整分配/释放生命周期
不扣减，采集前分配缺失；替代缓冲区、时序变化、静态缓存、碎片及图内存未计入。
allocated 与 reserved 分开报告，此估算不证明更大 chunk 的容量。

此前硬件结果：v1008 冷请求平均 52.105 秒，配对基线 51.418 秒，约慢 1.34%，
分配 scratch 不变；v1011 为 52.050 / 52.773 秒，基线 51.236 秒，每卡额外权重
缓存 90 MiB，完整模型峰值增加；v1012 单算子约快 6.7 倍并少分配 480.627 MiB，
但完整冷请求仍为 51.609 / 51.577 秒，模型峰值基本不变。零原生调用的早期
projection 试验不计入性能结果。final-mixer 完整服务/控制报告仍保存在 NPU 主机，
本阶段未远程取回。

v1012 decode 控制为 9.103 / 9.447 tok/s，显式清零 worker cache 后文本变化；
基线为 9.786 / 9.814 tok/s，文本一致。小行数未启用原生 final mixer，绑定/图
状态交互尚待定位，故未提升为默认。清零仅用于诊断，重复字符不验证语言质量。

最后确认服务为合格 r5 基线：v1001 prefill、v984 decode、budget 1,280、上下文
131,072、TP4、MTP1、decode graph 2/8、KV 2.25 GiB/卡。projection 缓存
已清理、final mixer 已关闭，health 200、未暂停、公开推理开放。本阶段无 NPU
测试或服务操作；更大 chunk、prefill graph 尚未验证，部分 prefix 文本差异未解决。
维护期间应先阻止新外部请求并 drain；新 generation 的准备 hook 必须在 capture
前执行。新 r5 进程真实权重检查每卡接受 84–85 分支，验证不可跨进程复用。
