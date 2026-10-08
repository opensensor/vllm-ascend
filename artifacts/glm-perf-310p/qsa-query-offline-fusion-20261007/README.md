# GLM QSA 元数据融合与下一轮离线候选

## 已有硬件证据

- v941 元数据内核通过 48 个 310P 用例，覆盖有符号整数、步长视图、
  尾部、输入保护区以及变化输入的 ACL graph replay。
- v942 接入目标模型和 draft 的已加载 attention 实例；四 rank decode
  trace 中原来的 36 个 AI-CPU FloorDiv 消失。
- v947 的 decode trace 中 AI-CPU Cast 从 14 个减少到 2 个。
  这是算子归因证据，不是端到端吞吐提升结论。
- 1280-token 冷预填充配对中，v942 相对 v938 基本持平。
  全尺寸 scalar query converter 曾使预填充变慢，随后限制为 decode。
  v945 的预填充配对也基本持平。没有据此宣称达到 10 tok/s。
- v947 将 capture 外描述符数量从 20,480 缩到 5,120，节省约
  1.52 MiB/rank；不支持的 mixed 几何保留原路径。
  mixed 请求硬件测试尚未运行。

## 服务状态与上下文

最后确认的配置是 max-model-len=311040、max-num-batched-tokens=640、
max-num-seqs=4、TP=4、MTP=1。每个请求的上下文上限包含输入和输出；
并发请求共享 KV 容量，不代表四个完整最大上下文能同时驻留。

v947 profiling 在有效 decode capture 后使用了不受支持的 prefill 标签。
RPC 异常使响应协议失步，不能声称服务已经恢复。
用户要求暂缓 NPU 操作后，停止了自己的控制器；之后只做 CPU 测试、
SDK 编译和已有文件归档，没有恢复服务、发请求或热替换。

## 新离线候选

| 构建 | 完整路径修改 | 状态 |
| --- | --- | --- |
| v952 | v913 decode：expert down 原生列顺序归约，完成 token 后只重排一次 | SDK 编译通过，未上卡 |
| v953 | v919 prefill：down 写原生列顺序 FP16 workspace，配套 reducer 最后重排 | SDK 编译通过，未上卡 |
| query v954 | FP16→BF16 向量转换，目标与 draft forward 可逆绑定 | SDK 编译通过，未上卡 |

列重排融合保留每个 route 的 FP16 舍入、FP32 权重乘法和稳定加法顺序。
CPU 位级检查覆盖有限 FP16 模式、正负零、抵消与被屏蔽的 route。
变化的是逐 route Gather 的位置，不是 route workspace 的字节数。
prefill producer 与 reducer 必须来自同一个构建，禁止只替换一端。

向量转换用 310P 支持的向量 Cast/Add/And/Or/GatherMask 代替全元素
scalar 循环；只有不足 16 个元素的尾部使用 scalar 读取。
CPU oracle 穷举了 63,490 个非 NaN FP16 位模式，NaN 保留原 converter
的带符号 canonical 策略。CPU oracle 不证明硬件向量指令行为。

v950/v951 复用已有 bridge，只用于离线/独立探针；resident manifest
要求 v954 的独立 namespace，避免重复注册算子。
诊断 RPC 的新 guard 将异常转为每个 rank 的错误响应；未部署。

## 后续完整验证

以下命令仅记录待执行步骤，用户暂缓 NPU 的要求仍然有效。
硬件可用并重新获准后，先完成独立算术门，再运行真实模型。

```bash
python -m tools.glm_perf.route_columns_probe \
  --baseline "$ROOT/build-v913" --candidate "$ROOT/build-v952" \
  --output decode-route-columns.json --allow-device-gate
python -m tools.glm_perf.route_columns_probe \
  --baseline "$ROOT/build-v919" --candidate "$ROOT/build-v953" \
  --output prefill-route-columns.json --allow-device-gate
python -m tools.glm_perf.query_bf16_vector_probe \
  --build-dir "$ROOT/build-query-vector-v954" \
  --output query-vector-gates.json --allow-device-gate
```

ROOT 是远端 `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007`。
route 探针比较完整 input quant → gate/up → SwiGLU/hidden quant → down
→ route reduction，覆盖 W2/W3/W4 × A4/A8 × 2/8/17/640 tokens，
变化输入 replay 后要求逐位一致，并交替采样 graph replay 时间。
仍需原有 real-weight binary gates、混合请求、全模型 graph capture，
以及 1280/7680-token 冷请求和 c1/c4 配对吞吐测试。
不得将小算术门当作真实模型性能或质量门。

## 文件与本地检查

压缩包包含 163 个原有证据文件及候选构建；manifest 记录逐文件 SHA256，
本地已逐个核验。两个原始 SDK trace 约各 1.9 GiB，保留在远端；
压缩包保存四 rank 的 CSV 与归因 JSON，没有复制原始 SDK 数据库。
构建的 provenance 标记硬件门未运行，不能用于宣称新候选加速。

本轮聚焦 CPU 测试 257 项通过。扩展 GLM 全目录首次收集受本机缺少
`torch_npu` 与 upstream `vllm.models.deepseek_v41` 阻挡；未安装或升级依赖。
更早的硬件门与本轮离线候选状态分开记录。

提交前兼容本机依赖的扩展 CPU 测试 1,508 项通过，排除 assembly、
grouped gate/up、KPool ops 三个依赖本机缺失模块的测试文件。
修复了新 CLI 参数的测试期待，并为历史 KDA patch 测试保存了原始
prepare header，避免后续代码变化破坏旧实验的源文件哈希门。
源代码和 Markdown 的 scoped hooks 通过；包含所有历史快照的全范围
检查仍报告旧探针脚本格式、技术词拼写与原始日志内容问题。
历史代码/日志快照保持原样，不能靠改写证据使检查通过。

US English 交付摘要见 [README.en.md](README.en.md)。
