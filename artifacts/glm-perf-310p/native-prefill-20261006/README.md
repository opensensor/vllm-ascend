# GLM 310P：原生 INT4 冷 prefill 与分段图实测

在既有 TP4 服务热换模块，四 worker PID 与权重存储摘要保持不变。
沿用永久 `cube_n128_k256_v1` 权重布局，没有重新转换权重银行。
本目录记录较早的 v45 + BF16 indexer v5 阶段；最新磁盘发布与后续实验见
[throughput 实测](../native-throughput-20261006/README.md)。图策略仍需重启后重新应用。

## 实现

- 大于 16 token 的 down 投影按完整专家路由批处理，避免原先每 16 token
  重走所有专家。写出加权 FP32 路由行，再由 AI-Core 按稳定专家顺序归并。
  640 token、top-k 8、hidden 4096 的临时区为 80 MiB；没有 FP16 权重 GM 工作区。
- 将标量 scale 操作改为向量广播和整矩阵乘法，保留原 FP32 运算顺序。
  稀疏行缓存组合 scale。双行隐藏量化通过 `--pair-hidden-quant` 显式启用，
  默认关闭，须另过模型质量门。
- 仅 640 bucket 共享 MoE 临时区；decode 和其他形状保留原分配方式，
  返回结果独立分配。同步 serving stream 已验证，异步、多 stream 未验证。
- 主模型单请求的完整 640-token chunk 使用 PIECEWISE 图：每 rank 46 图段、
  45 个 attention/indexer/KDA 动态状态断点；FULL decode 2/8 保留。
  捕获后恢复原 compilation config 与 dispatcher，仅 640 分派使用独立副本。
  MTP draft 保留原 dispatcher，其他 prefill 尺寸及混合请求保留原分派。
- 捕获前同步 stream，在 target/draft 旧图均清空后回收实验资源 scratch
  和 allocator 缓存；旧图仍存在时拒绝清理。重复热换曾导致 OOM，清理后重捕获成功。
- 原生加载只执行一次；相同 digest 的准备回执不能证明设备验证通过。
  使用各 rank 已验证注册表确认加载结果。

## 性能与瓶颈

CANN 实验使用同一真实权重服务、两个冷 640-token chunk，总输入 1280、输出 1。
profiling 会扰动延迟；这些是有限次数的匹配测量，不能当作稳定吞吐保证。

| 原生版本 | API 时长 | CANN step 1 | CANN step 2 |
| --- | ---: | ---: | ---: |
| v41 | 32.708 s | 15.861 s | 17.298 s |
| v44：完整专家批处理、scale 广播 | 27.788 s | 13.376 s | 14.832 s |
| v45：整矩阵 scale | 25.992 s | 12.533 s | 13.836 s |
| v47：双行隐藏量化、分段图（未发布） | 25.511 s | 12.381 s | 13.520 s |

v41→v45 API 时长降低约 20.5%。v45 每步 gate/up 约 4.42–4.45 s，
down 约 2.27–2.29 s，KDA 约 1.31 s。四 rank prefill AI-CPU Cast 均为 0。
输入量化约 115.5 ms/step，少于 1%，已经按输入 token 量化一次并在专家间复用。

原始 v45 完整 API suite 26/26 有效、质量 18/20、tool 1/1；固定输出 64 的
c1 8.091 tok/s、c4 总量 12.510 tok/s。v2 图适配器单次为 7.365/13.390；
不能据此证明 decode 改善，尚未达到 c1 10 tok/s。

图自身收益有限：相同 v47 内核的长检索 graph/direct TTFT 为 76.908/77.292 s，
约 0.5% 差异，两者回答正确。图 trace 的 op-summary 覆盖率不同，v47 rank0
每步约 3.0–3.3 s 没有报告任务，不能直接归为 CPU 空闲或与 direct 同口径比较。

## 质量与图门

最新 v4 图适配器复用 v45 原生内核对象、配置及非 640 路径，仅用 v53 helper
为 640 提供共享临时区。v52/v53 四个计算二进制与 v45 SHA-256 完全相同，
见 `default-compute-binary-equivalence.json`。

暂停服务后提交同一四请求批次，原 v41 与 v4 都为 17/20；失败项相同：
`instr_reverse`、`instr_first`、`code_slice`。两者质量 prefill 路由形状集合一致。
v3 曾额外错误回答 `arith_mod`；v4 隔离普通请求的 dispatcher/config 后恢复该题。
原始输出保留于 `queued-quality.json`，判读见 `matched-quality-assessment.json`。
暂停期间 scheduler metrics 不刷新，旧的 waiting=0 不能证明队列为空。
历史完整套件的 18/20 发布门保留，没有用本次 17/20 放宽磁盘发布条件。

v4 的 1280-token 请求确认全部四 rank 至少两次真实图重放，随后 decode 正常。
冷检索原 prompt 3833 token、chat 输入 3845 token，正确返回 `BLUE-ORCHID-7319`；
TTFT 77.699 s，全部 rank 在原两次基础上新增至少六次重放。
见 `graph-qualified-v45-v4/qualification.json` 与 `final-retrieval-v4.json`。
早期 v3 检索虽正确，第一次状态读取未满足全 rank replay 断言；失败日志也保留。

36 个原生算术/重放门与 18 个真实专家逐位门通过。相同 live 激活与路由下，
v48 与 v46 MoE 输出逐位相同（各 rank 214/128/128/128 次）；此检查不是 v41 对照。
CPU 隔离测试 187 项通过，scoped pre-commit 全部通过。全部模型质量与性能结果
使用真实权重，没有以 dummy 签收。

## 重放

源目录 `/srv/ai/src/glm-selective-w3-nz-test-20261004`，Python 为
`/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python`，API `http://127.0.0.1:8001`。
先准备对应 `reconstruction_v45`、`bf16_cast_v5` 资源和冻结 v53 helper，
核对来源摘要，再应用完整候选：

```bash
python -m tools.glm_perf.resident_harness --base-url http://127.0.0.1:8001 \
  switch --mode graph \
  --candidate /srv/ai/artifacts/glm-native-prefill-20261006/graph-qualified-v45-v4/candidate.py \
  --candidate-name native_v45_qualified_prefill_graph_v4
python -m tools.glm_perf.resident_harness --base-url http://127.0.0.1:8001 status
```

策略实现为 `tools/glm_perf/resident_candidates/prefill_graphs.py`；冻结完整候选为
`graph-qualified-v45-v4/candidate.txt`。本阶段重启保留永久模型和 v45 kernel bundle；后续已发布 v56，
resident 图策略须重新应用和验证，启动链的默认图启用尚未验证。

## 继续优化的方向

优先减少 INT32 Cube 读回、block32 scale/FP32 归并及重复装载开销，
并减少动态 attention 的 eager 断点；预数量化输入已经只占很小比例。
更大 scale 分组需要离线重新量化及质量门，不能在线直接改变数值。
本轮尚未修复 prefix cache 命中率为 0，没有验证混合并发 prefill 图、
EP/flashcomm 或 bs16 容量。沿用 TP4、MTP 1、max-seqs 4、max-len 311040；
文本模型不涉及多模态验证。
