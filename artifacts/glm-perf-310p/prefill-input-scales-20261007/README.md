# GLM prefill：original-token FP32 scales 直接传给路由 producer

用户明确表示 scale buffer 每 rank 节省 MiB 值得保留。本轮接续 v918 / v913，
仅改变 FP32 scale layout/传输，不改变权重、量化 arithmetic、rounding 或 decode。

## 实现与容量

- `--raw-input-scales` 默认关闭，要求 `--route-packed-input`。
  >16 token A4 的 original-token pack 直接复制 quantizer 已保留的八个 scalar FP32 scales，
  不再向 GM 写八 lane broadcast。route producer 对每个 routed token 直接复制 scalar row，
  不再先 copy broadcast 再 gather；该分支去掉 scale-input 和 index UB buffers。
- 其余 pipeline 使用已有 routed scale bank，gate/up/down 的量化与 FP32 累加不变。
  没有新 kernel launch、CPU sync 或额外 route metadata 回传。
- opt-in private pack descriptor 加入 CPU-visible token count，用来精确判定 bulk，
  无需根据 hidden group count 猜测 tokens；A8 与 <=16 token 保持原八 lane buffer。
  默认 descriptor 仍为旧两字段，decode 始终独立使用 v913。
- 实际640/hidden4096：原 `[640,128,8]` **2.5 MiB** → `[640,128]` **0.3125 MiB**，
  每 rank 再少 **2.1875 MiB**。这里只是保留 buffer 容量；不宣称 caching allocator 的
  reserved bytes 同步归还系统，也不改变已启动的 KV block/context 配额。

## 验证

66 paired real W2/W3/W4 × A4/A8 exact replay + 30 independent math/replay +
12 real reference 全通过。包含 partial tails、changed inputs/routes/codes/scales、
重复 route、零权重、peer suffix 与 all-peer 输出；真实权重来自既有 canonical checkpoint。

隔离 CPU suite **573 passed**；新增 UT 覆盖 private token descriptor、dtype/shape、
A8/small bypass、shared scratch 重用、scalar/broadcast row 位模式一致、无效配置与 provenance。
scoped pre-commit、markdown 检查通过；没有运行文本质量评分，用户继续判读生成质量。

## 性能与保留策略

640-token 单真实 expert top-k1、2 warmup/5 samples，route stage 约 **0.30 → 0.24–0.25 ms**。
全部五阶段各自 median 相加：W2A4 27.266 → 27.218 ms；W3A4 27.419 → 27.481 ms；
W4A4 27.089 → 27.019 ms。整体很接近，不宣称完整专家 pipeline 明显加速。

同 seed42、1280 个 token ID42、输出上限1，每次清空 prefix cache；两组交错 matched
median：**v918 14.402 s → v919 14.465 s**，增加 **0.43%**。只用四条 matched 样本计算
median；控制器切回 v918 的额外 qualification 请求，以及最终 v919 qualification 请求，
不计入这个比较。此次不宣称端到端 speedup，也不重复无关的长 context 或 decode benchmark。

纯时间选择器最初选 v918。根据用户已明确的 memory 优先方向，随后保留 v919；
retention controller 设置1% observed slowdown 上界，实际0.43%满足该界限。
这属于明确的容量/小幅时延取舍，不把 slow-down 隐藏为加速。
最终 **v919 prefill / v913 decode**，四 rank 新图 replay>=2、native fallback=0、
native_failed=false、graphs_dirty=false、unpaused；PID 与 weight storage digest 不变。
每 rank original-input scale bank 确认 `[640,128]` /327680 bytes；回执在 `validation.json`。

## 复现与边界

v919 构建参数是 v918 加 `--raw-input-scales`；build/provenance/helper/package/bin/bridge
均冻结并验证 SHA256，`.py.txt` 恢复名称并验证 hash 才可导入。原始 SDK 日志 `.log.gz`
解压 hash 在 raw-log-hashes.json；额外 memory retention 协议和请求单独归档。

既有 TP4/MTP1、max len311040、batch tokens640、max sequences4、永久磁盘权重和模型
启动配置保持原样。FULL decode2/8、单请求640 prefill 46 segments/45 eager breaks 不变；
不宣称消除了全部 eager boundaries。没有修改上游 model runner、引入 env 或重新变换模型。

API：`http://192.168.53.187:8001/v1/models`。
日志：
`/home/matteius/experiments/glm-reconstruction-20261006/serve-native-disk-head-nz-recovery-20261007-v2.log`。
