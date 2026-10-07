# GLM prefill：压缩 down FP32 activation scales

接续 v916 prefill / v913 decode，保持 quantizer、FP32 scale precision、累加与权重不变。
用户自行判读生成质量；本轮仅做真实权重数学/replay 与性能验证。

## 两个独立 schedule

- `--route-compact-down-scales` 默认关闭，要求 packed down input。
  v917 将每个 N128 tile 的四个独立 FP32 scales 写成 tile-major `[slot,tile,32,8]`。
  八 lane 中前四个保存 scales，后四个是 DMA padding；down 一次读取整块，再从本地
  product readback scratch gather active rows，替代每行重复的 GM copy。
- `--raw-hidden-scales` 默认关闭，要求 compact layout + four-row quantization。
  v918 直接使用 quantizer 在广播前已算出的 scalar scales；四行 × 四 scales 是一个
  64-byte 对齐 copy，写 `[slot,tile,32,4]`，无需 producer gather/clear/per-row copy。
  partial tail 的 padding row 也有定义好的 quantizer 值，但 consumer 只读取有效行。
- 每个输出 tile 独占自己的 scale slice，不引入 producer launch、额外 UB 或 CPU sync。
  consumer 使用下一次 Cube 操作前尚未存放新 products 的 scratch 暂存 compact bank；
  scale 按 tile/row/group 的独立 offset gather 到既有 SX 矩阵，数值位模式不变。
- A8 / <=16 token 保持旧 broadcast ABI；decode 始终使用独立 v913。
  CPU scratch 容量依赖 shapes 和原 packed-down expert count key，不读取设备 route ends。

实际 640/top-k8/72 local experts/intermediate2048：原 scale bank **10 MiB**，
v917 `[238,16,32,8]` **3.71875 MiB**；v918 `[238,16,32,4]` **1.859375 MiB**。
这是预留容量，不是实测总 GM/HBM traffic；consumer 仍读取整块含 inactive padding 的 bank。

## 验证

v917：66 paired real W2/W3/W4 × A4/A8 exact replay + 30 independent math/replay +
12 real reference 全通过；CPU **552 passed**。
v918：同样的 **66 + 30 + 12** 硬件 gate 全通过；CPU **564 passed**。
UT 覆盖 DMA alignment、partial tail、scalar FP32 位模式、capacity、A8/decode bypass、
无效 build/provenance；隔离 HEAD 的 scoped pre-commit 通过。
没有进行文本质量评分或变换/重新发布磁盘权重。

## v917 在线比较

同 seed42、1280 个 token ID42、输出上限1，每次清空 prefix cache；两组交错样本：
**v916 14.740 s → v917 14.630 s**，观察到 median TTFT 减少 **0.75%**。
这次时间改善很小；另有确定的 **6.28125 MiB/rank** scale scratch 容量减少。
四 rank 至少2次新 prefill replay、fallback=0、workers/weight storage 不变、unpaused。

v917 的 640-token 单真实 expert、top-k1、2 warmup/5 samples 全阶段 median 相加：
W2A4 27.749 → 27.591 ms；W3A4 27.903 → 27.777 ms；W4A4 27.450 → 27.396 ms。
down 少约0.25 ms，但 producer packing 抵消一部分，所以继续尝试直接 scalar copy。
A8 未启用新布局，不从小幅测量变化推导 A8 加速。

## v918 直接 scalar copy

同设置的两组 matched median：**v917 14.599 s → v918 14.443 s**，再减少 **1.07%**。
没有使用跨实验/跨时间的绝对时延计算叠加百分比；两个比较都有各自匹配的基准。
当前短 prompt 最终驻留 v918 / v913，四 rank 至少2次新图 replay、fallback=0，
PID 与 weight storage digest 保持不变，unpaused。

7680-token 长 prompt 的单次 matched pair：**96.795 s → 96.852 s**，增加约 **0.06%**，
实际接近平。没有宣称本轮 scalar copy 改善长 prefill；这个杠杆主要带来短 prompt 的小幅
改善与确定的 buffer 容量减少。最终保留 matched short median 更好的 v918 / v913；
四 rank 新图 replay>=12、fallback=0、native_failed=false、graphs_dirty=false、unpaused，
workers 与 weight storage digest 不变，完整回执在 `validation.json`。

单专家640-token profile（各阶段 median 相加，包含全部五阶段）：

| 精度 | v917 ms | v918 ms | 时间减少 |
| --- | ---: | ---: | ---: |
| W2A4 | 27.662 | 27.358 | 1.10% |
| W3A4 | 27.730 | 27.502 | 0.82% |
| W4A4 | 27.488 | 27.095 | 1.43% |

gate/up 少约0.31 ms；down 时间基本相同，未从 scale bank 字节减半推导 matmul 时间减半。
相对 v916 的 scale scratch 减少 **8.140625 MiB/rank**；相对 v917 再减少1.859375 MiB。
这不改变独立 v913 decode，也不宣称新的 decode tok/s 改善。

## 复现与服务边界

v917 参数是 v916 加 `--route-compact-down-scales`；v918 再加 `--raw-hidden-scales`。
frozen builds 包含 binary/bridge/provenance/helpers，历史 `.py.txt` 恢复文件名并校验
SHA256 才可导入。SDK raw log 用 `.log.gz` 保存，解压 hash 在 raw-log-hashes.json。
v918 frozen evidence 与最终在线回执在 `raw-copy/`；v917 证据保留在本目录。

TP4/MTP1、max len311040、batch tokens640、max sequences4、永久模型与 PIDs 保持既有配置。
FULL decode2/8 和单请求640 prefill 46 segments/45 eager breaks 不变；未在本轮重新验证
mixed/full prefill、EP/flashcomm1、新 context 或并发参数。
API：`http://192.168.53.187:8001/v1/models`。
日志：
`/home/matteius/experiments/glm-reconstruction-20261006/serve-native-disk-head-nz-recovery-20261007-v2.log`。
