# GLM 冷 prefill：31 行专家批次（2026-10-07）

本轮候选 v910 把每次专家批次从 15 行扩为 31 行，减少重复 Cube 调度、
读回和重建。只替换 resident prefill；16 token 及以下仍用 v905 decode。
对照是正在服务的 v909 prefill + v905 decode，四个 worker 和模型权重不重载。

## 实现与内存

- 新构建参数 `--prefill-rows-32` 默认关闭。需要 prepared fused MoE 和
  vector scale products；K128/K256、矩阵 Gather、LUT、FP16 SwiGLU 和其他读回
  实验组合在创建构建目录前拒绝。
- 激活 L0A 加载、Cube M、矩阵读回均支持两个 M16 块；最多 31 行，最后一行
  留给 A8 偏置。配对 A4 两个 block32 组使用 M64/K64 产品，FP32 求和顺序不变。
- GM route ABI 始终以 16 token 为边界，独立于 Cube 行容量；40 MiB/rank
  FP16 route scratch 保持一致，最终输出和稳定 route reduction 仍为 FP32。
- 重建完成后复用其 64 KiB UB：前 32 KiB 存 INT32 Cube 产品，后两个
  16 KiB 区存 FP32 low/high。accumulator 单独存放，跨 K256 重建保留。
  factors 复用 mask scratch；八个权重 scale 向量移到 gathered scratch 空闲尾部。
- 静态 UB 分配：gate/up 219,488 B，down 233,792 B，均保留 SDK 临时区。
  gate/up 的 L1 激活缓存上限 256 KiB，权重缓存上限 512 KiB；只在当前 kernel
  launch 内重用，每次 replay 重新准备。

## 验证

- 66 个真实 W2/W3/W4 × A4/A8 成对病例全部 bitwise 一致；token 为
  2/15/16/17/30/31/32/33/62/63/640。图 replay 修改激活、route、packed codes
  和 scale，覆盖重复专家、零权重和全部 peer routes。
- 独立参考数学和 replay 30 个病例，真实专家 12 个病例全部通过。
  真实 31 token gate 在 replay 时把两个 slots 都指向同一专家，覆盖 62 行。
- 隔离 HEAD checkout 的 scoped CPU suite：506 passed。排除既有三个独立实验
  文件，与上一轮范围相同。scoped pre-commit 通过。
- 未运行 dummy；硬件 gate 和 HTTP 测试均使用真实权重。

## 单专家设备 profile

下面是四个 fused stage 的各自 median 相加，不是全模型 TTFT，也不是 HBM 流量测量。
每组合 2 次 warmup、5 个样本，640 token，top-k=1，实际 checkpoint expert 0。

| 精度 | v909 (ms) | v910 (ms) | 时间减少 |
| --- | ---: | ---: | ---: |
| W2A4 | 44.040 | 35.630 | 19.10% |
| W3A4 | 44.234 | 35.587 | 19.55% |
| W4A4 | 43.983 | 35.507 | 19.27% |
| W2A8 | 113.667 | 96.735 | 14.90% |
| W3A8 | 113.812 | 96.952 | 14.81% |
| W4A8 | 113.706 | 96.710 | 14.95% |

## 在线证据与复现

宽行版本的完整 API 结果保留于 `measurements/v910-wide-rows-live.json`。
1280 token 两次 median TTFT：v909 18.832 s → v910 16.740 s，减少 11.11%；
2560 token 单次 39.730 s → 35.068 s，减少 11.74%。输出上限均为 8 token。
单次 short decode：9.755 → 9.380 tok/s；c4 14.010 → 14.005 总 tok/s。
该轮严格的单次速度保留条件未通过，控制器恢复 v909。重复测试在用户请求出现时
没有启动，随后按用户指示取消。历史自动评分为 19/20，但用户明确自行评判输出，
后续优化不运行输出质量 suite，也不要求相同文本作为保留条件。
最终组合及运行状态见 `validation.json` 和后续 quantization 报告。

硬件控制先确认 idle，采用 `/pause?mode=wait&clear_cache=true`，从不终止用户请求。
独立 gate/profile 完成后恢复原候选；HTTP 对照通过 resident switch 重捕获图。
冷请求用同一模型、seed=42、synthetic token ID 42、8 个输出 token，每次显式清空
prefix cache。冷实验中 cache hit=0 是实验条件，不表示功能失效。

运行环境：TP4、MTP1、max model len 311040、chunked prefill=640、max sequences=4。
原有 FULL decode graph 与 640 token segmented prefill graph 保留；本轮没有将所有
prefill 声称为 full graph。EP/flashcomm1 和更大并发不在本轮实验范围，模型为文本模型。

`protocols/*.py.txt` 为当时实际控制脚本；`frozen-source` 保留编译快照，
`frozen-builds/build-v910` 保留二进制和 SHA256 provenance。旧 v905/v909 对照包在
上一轮 `prefill-input-reuse-20261007` / `prefill-vector-readback-20261007` 中。
历史 helper `.py.txt` 导入前须恢复 `.py` 文件名并验证原 SHA256。
SDK 原始 log 以 `.log.gz` 保存，解压后的 SHA256 位于 `raw-log-hashes.json`。
最终 builder 增加了 K128/K256 组合拒绝，未改变已验证 `wide_cube_k=0` 的 kernel 源码。

紧凑重建命令：从 frozen source 恢复源文件，在服务 Python/CANN 环境内运行
`python -m tools.glm_perf.build_reconstruction`，使用新的 append-only build/version，
启用 `--output-columns 128 --tile-pipeline --all-bits --fused-moe`
`--prepared-weight-layout --pair-scale-groups --pair-prefill-scale-groups`
`--prefill-weight-cache --fp16-route-workspace --share-gate-up-input`
`--cache-gate-up-activations --vector-scale-products --prefill-rows-32`。

服务日志：
`/home/matteius/experiments/glm-reconstruction-20261006/serve-native-disk-head-nz-recovery-20261007-v2.log`。
API `http://192.168.53.187:8001/v1/models`；本轮不重启服务，不发布新的磁盘默认模型。
