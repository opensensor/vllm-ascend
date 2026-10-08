# GLM FP16 专家尺度存储

本候选将 block32 专家尺度持久化为 FP16；每次加载 tile 后提升到现有 FP32
缓存。尺度乘法、INT4 Cube 计算、FP32 累加顺序、SwiGLU 和路由归约保持原语义。
格式标记为 `fp16_storage_fp32_compute_v1`。

导出命令使用 `tools.glm_perf.prerounded_scale_checkpoint --fp16-storage`；
编译使用 `--fp16-weight-scales`。config、native manifest、safetensors shard
及 resident bank 标记必须一致。loader 在放置前检查 dtype，native 在提交前
拒绝不匹配的尺度缓存；与预舍入 FP32 编译选项互斥。原始或预舍入 FP32 均可
作为导出输入。codes 和 dense weights 保持硬链接，不做在线转换或解包。

K2048/K4096 使用对齐 DMA，一次 tile 加载只进行一次向量提升；小型或非对齐
测试行使用有界标量加载，避免读取最后一个 expert 之外的数据。FP32 缓存、
FP16 staging 与 gather indices 分离，兼容 gate/up 缓存。

真实模型已导出到
`/srv/ai/models/GLM-5.3-Flash-native-int4-fp16-scales-20261008`。
37,152 个尺度张量共 608,698,368 字节，恰为原 FP32 payload 一半；
包含 target 和 draft 时每 rank 节省 **145.125 MiB**，TP4 共 580.5 MiB。
硬链接源文件可能包含旧尺度，因此不能声称整个目录磁盘占用减少同等数量。

## 验证

- 最终兼容 CPU GLM suites **1,643 项通过**；最终聚焦回归 **102 项通过**。
  assembly、grouped gate/up 和 KPool ops 依赖本机缺失模块，仍排除这三个文件。
- 检查所有有限 FP16 位模式、有符号零、舍入边界、溢出拒绝、原始及预舍入导出、
  不变字节、实际 loader iterator、FP16 resident 分配以及错误 ABI 拒绝。
- v964/v965 分别通过 30 个 synthetic 和 12 个真实 expert 算术及 replay cases，
  覆盖 W2/W3/W4、A4/A8、真实 K4096/K2048。
- v962/v963 在 2/8/17/640 tokens 的配对及 changed-input graph checks 中逐位一致。
  v964/v965 kernel SHA 与其一致；C++ 最终修改只删除重复的编译期格式检查。
- 之前排队的 compact-W4、预舍入 FP32 与 route-column 实验均已执行。
  route-column v913→v952、v919→v953 配对也通过；四项特性、两个调度合计
  **192 paired cases** 逐位及 changed-input replay 通过；属于 operator 检查，
  不能宣称完整模型吞吐提升。

第一轮单 expert profile 的变化较小且不一致：W4/A4 gate/up 约 0.322→0.304 ms，
W3/A8 约 0.882→0.897 ms。输入为 synthetic，仅一个 expert，不能由此宣称
整模型吞吐提升。TP4/MTP1 服务已直接加载永久 FP16 checkpoint，并捕获 v964 decode/v965 prefill
混合调度。四 rank 回执均确认 43 个 resident banks、FP16 尺度 dtype 及每 rank
152,174,592 字节。服务请求已完成，640-token 图共 replay 11 次，未进入 native fallback。
synthetic token-id 请求中，c1 generation **8.47–8.76 tok/s**，640-token 冷 TTFT
**6.94 s**，6,400-token 冷 TTFT **80.93 s**。C4 包含 prefill 的总请求吞吐
**8.87 tok/s**；四请求独立 generation 为 2.70/4.39/4.54/4.54 tok/s。
这些是 fresh-load 绝对值，不能宣称配对加速。语言样本在 reasoning 阶段达到
48-token 上限，语言质量未评估。

v954 query converter 的 exhaustive hardware gate 失败，未接入服务。定位到
dav-m200 Compare 对不足 256 字节的 repeat 直接截断，短向量 mask 未初始化；
NE(x,x) 也不满足所需 NaN 分类。后续修复将 UB 比较长度补齐完整 repeat，
通过有限的 exponent/mantissa 数值分类 NaN；GM 只写拥有的 DMA padding。
修复后的 v966 与最终 v967 均通过 13 项 device gates，包含全部 FP16 位模式、
NaN、短尾边界、输入 guards、输出 padding 与 changed-input graph replay。
重建 Indexer forward 时保留永久 RoPE 转换；回归测试区分 query 和 legacy
converter。v966 服务测量发生在绑定修复之前，不属于最终候选。v967 已接入
12 个 target/draft indexers，保留 v964/v965 FP16 调度；归档包含 frozen helpers
和有校验的 native binaries。
最终组合服务 c1 **8.47–8.68 tok/s**，640-token 冷 TTFT **6.75 s**，
6,400-token 冷 TTFT **78.64 s**；C4 含 prefill 的总请求吞吐 **8.68 tok/s**，
各请求 generation 为 2.73/4.12/4.25/4.25 tok/s。prefill/decode 调度均执行，
四 rank 确认图有效、native 无失败且 query native 调用生效。这些不是配对整模型
加速或语言质量结论；96-token chat 样本包含可读 reasoning 并开始回答，但达到
输出上限，原始响应已归档，不声明质量通过。

## 永久格式复现

```bash
python -m tools.glm_perf.prerounded_scale_checkpoint \
  --source /path/to/permanent-native-checkpoint \
  --output /path/to/NEW-fp16-scale-checkpoint \
  --kernel-bundle /path/to/matching-fp16-scale-bundle \
  --fp16-storage
```

拒绝覆盖已有输出。失败导出保留 incomplete 标记，loader 拒绝加载。
`PrefillDecodeNative` 保留并校验两个调度共享的尺度 ABI。不要将 FP16 bank
提交给旧 FP32 尺度 kernel。语言质量及完整服务性能必须与算术/replay 验证区分。

## 当前服务及复现

端口 **8001**，模型名 `glm53-flash-selective-w3`。日志：
`/srv/ai/artifacts/glm-prefill-weight-reuse-20261007/fp16-scales-server-20261008.log`。
process JSON 保存完整启动命令。保留 TP4、MTP1、640-token chunk、4 个请求槽、
prefix caching 与 311,040 配置上下文；本轮最大测试 prompt 为 6,400 tokens，
不能将配置上限视为已验证容量。decode 2/8 与 resident 640-token prefill 图开启；prefill 回执为 **46 个 graph
segments、45 个 eager boundaries**，live attention/indexer 仍有 eager 边界。
最终 benchmark replay 11 次；这是分段图，并非完整 prefill 单一无间断图。
EP/flashcomm1 未修改或单独验证，image/video 禁用。使用真实 checkpoint 及
服务/replay 检查，未使用 dummy；保留用户的 4-slot 配置，未改为 skill 的
16-slot 容量基线。

`.py.txt` 保存启动、hot-swap 和 benchmark 脚本。启动后先运行 mixed-live
脚本加载合格 v965 manifest，再运行最终 v967-live 脚本加载转换器并重新捕图。
脚本使用新的 generation IDs，并检查四 rank 回执；native admission 要求对应
完整 gate JSON。具体 source/resource 路径在脚本内。切换前 drain 并清理 prefix，
成功后恢复服务；无在线权重 repacking，切换中 worker PID 及 storage digest 不变。

另一次使用 512-token 输出预算的服务 smoke 正常结束（`finish_reason=stop`），
返回非空最终答案；响应及四 rank 回执保存于
`fp16-scales-final-serving-smoke-20261008.json`。这验证请求完成，不代表模型质量
评估。最终回执确认服务恢复。
