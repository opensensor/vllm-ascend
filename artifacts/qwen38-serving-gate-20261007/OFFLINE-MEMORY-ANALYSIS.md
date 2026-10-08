# Qwen 310P 内存搬运离线分析与待测 profile

当前阶段没有NPU权限；用户要求保持server关闭，所有分析仅使用保存的
源码、日志与metadata。下一次实验配置保存于[queued-npu-profile.json](queued-npu-profile.json)，
状态为awaiting explicit NPU access grant，无自动启动或定时任务。

## 已有证据

恢复实例的[只读snapshot](recovery-mamba-status.json)记录四rank、每rank
三个state groups，共412次spill和0次restore。每group checkpoint为
9,744,384 bytes，累计spill payload约3.739GiB；详细计算见
[offline-transfer-accounting.json](offline-transfer-accounting.json)。
这是累计payload估计，不是瞬时带宽或耗时测量。warning每个tier首次spill
才打印，不能用warning行数代替transfer counters。

KV usage约10%时Mamba primary/archive已满，说明attention token容量无法
代表prefix checkpoint容量。最新profile未通过持续热稳定性验收。温升是
持续设备负载的信号，但不能仅凭温度把原因归于CPU transfers；NPU内存访问、
计算、占用率和散热状态均需在相同条件下对照。

## 源码中的搬运与同步

| 路径 | 已确认行为 | 离线处理与后续验证 |
| --- | --- | --- |
| Mamba `_snapshot` | 逐tensor `.to("cpu", copy=True)`；未指定non_blocking，spill会阻塞host | 有界scheduler先撤销旧prefix hashes，worker再回收state，避免为不可保留历史创建host快照 |
| Mamba host restore | host snapshot逐tensor复制回state slots | 此snapshot的restore counters为0；不能声称反复restore解释了当前低吞吐 |
| Mamba `_synchronize_device_state` | 全NPU drain保护pending graph writes和slot复用 | 新retirement批量处理所有groups，最多一次drain；已完成layout-change drain时复用它。没有删除正确性必需的等待 |
| device archive | NPU→NPU复制或swap；维持checkpoint完整性 | 有界保留降低历史state数量，但archive地址不重建；实际访存成本待NPUtrace |
| PLE gather与H2N | PLE表在host，gather结果传给NPU；既有frontend复用host mirrors与gather pool | 不因长host调用时长就归为额外串行延迟。先核对overlap与payload，不直接关闭必要输入搬运 |
| QSA K/V gathers | device cache读取与重排属于NPU访存 | 与host spill分别计量；检查bytes和dense/sparse dispatch，不能把所有memory任务都算成CPU transfer |
| hyperconnection dtype casts | 既有路径保持FP32 mixing/recurrent精度 | 重复cast可作为独立candidate；未经exact tensor parity和changing-input replay，不降precision或并入本profile |

本轮forward fix只处理scheduler-owned checkpoint保留与批量retirement。
默认scheduler、KV allocation、OPP、MTP与图尺寸不变。它不会消除模型本身
必需的权重/cache访存，也不构成thermal修复已完成的证据。

## 等待授权的NPU对照

保持TP4/EP4、MTP2、三个slots、graphs `[3,9]`、1024-token scheduler batch、
fraction0.70、4GiB reserve和一张图片。对照只改变default scheduler与
`PrefixMambaBoundedScheduler`；保留已有96°C watchdog，不自动调高cutoff。

下一次明确获准访问后，使用独立完整runtime和证据目录，执行以下顺序：

1. 校验完整runtime、OPP symbols、FP32 states、graph capture与image encoder；
   确认bounded profile日志为每group27个cached checkpoints。
2. 单请求cold/repeat、三请求相同长历史、增量tool回合及image-prefill。
   使用相同prompt和输出长度，记录prefix hits与被evicted后的正确重算。
3. 对比每rank/group的spill、restore、retirement、device archive hit增量；
   保存TTFT、decode gaps、输出hash/语义、queue时间与固定间隔thermal样本。
4. 用named profiler分开记录Mamba snapshot/drain、archive copies、PLE、
   QSA gather、native W4 projection与dtype casts；区分host wall duration、
   device任务和overlap，不将未测量时间归给某一个counter。
5. 只有正确性与持续thermal/throughput结果均保存后才考虑promote；
   目前无部署、无新NPUbenchmark，也没有spill-free性能保证。

host tests已覆盖真实BlockPool的canonical/partial hash eviction、活跃与CoW
状态保护、IPC、slot reuse、同步失败回滚及三请求400步的state隔离。
其零spill结果是CPU状态模拟验收，不替代真实权重和NPUgraph资格。
