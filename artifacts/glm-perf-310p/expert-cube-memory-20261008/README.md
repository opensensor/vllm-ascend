# GLM 专家 Cube 与内存实验

保留已经验证的 KDA score-row batching 和磁盘 FP16 scale checkpoint。本批次
增加两个独立的专家 kernel 编译选项，并处理 resident 热切换时旧图缓存的释放。
没有改变量化器、激活量化或专家累加顺序。完整 US English 说明见
[英文报告](README.en.md)。

## 原生实现

- `--active-cube-rows`：A4 paired prefill 按实际行数选择 M16/M32/M48/M64。
  激活打包不变，Cube 输出读回和 cast/gather 使用对应物理步长；每核 UB
  增加 1,024 字节索引表。1–8、17–24 行减少计算/读回行数，其余行数保持原几何。
- `--direct-w4-l1`：磁盘已准备的 W4 直接 GM→L1，删除 UB 暂存及 UB→L1
  拷贝，沿用现有 projection cache 容量，不增加 GM/UB/L1 分配。
  W2/W3 reconstruction 与 scale 运算不变。
- v979 为行数实验，v980 同时启用两个选项，v981 仅 direct W4。两选项默认
  关闭，builder/provenance/compiler define/真实权重门禁均记录。
  服务只替换 prefill 对象，decode 保留 v964；编译原始源码和 binary 都冻结保存。

## NPU 验证

v979 有 30 个独立算术、12 个真实权重和 108 个 paired 边界用例。
v980 在四个真实 worker/rank 上分别通过 30/12/60 个用例；v981 在空闲设备
通过同样完整门禁，再在每个 serving rank 执行标准 12 个 admission 用例。
真实权重为 layer 10/11/33 的 expert 0，覆盖 W3/W4/W2、A4/A8、Cube 行数边界、
31 行专家 batch 及 640 token。每个 paired 用例三次图重放改变输入、路由权重、
all-hot/all-peer 路由，与 v965 字节完全一致。仅 peer 路由输出为零。
这些是算术和图行为验证，不是全模型语言质量认证。

## 完整服务测量

| Variant | Cold 640 first / repeat (s) | Cold 1,280 first / repeat (s) | Cold 6,400 (s) |
| --- | --- | --- | --- |
| Rows only (v979) | 5.71 / 5.41 | 11.79 / 11.46 | 66.08 |
| Rows + direct W4 (v980) | 5.75 / 5.43 | 11.80 / 11.47 | 65.31 |
| Direct W4 (v981) | 5.71 / 5.37 | 11.63 / 11.45 | 65.30 |

请求使用相同 token ID，冷测量清除 prefix cache，记录 HTTP 请求至首个可见文本
的完整时间。各组之间有重启和首次分配影响，不能把微小差异称为确定加速。
之前 KDA 的 640/1280/6400 记录为 5.40/11.68/65.36 秒（重复短请求为
5.36/11.43 秒）。v980 的 6400 为 65.31 秒，保留累计改进；本批专家选项
尚未证明额外吞吐收益。单个热专家 W4 A4 gate/up 为父版本 15.07 ms、
direct-only 15.08 ms、组合版本 15.23 ms。减少拷贝不能代替完整关键路径测量。

最终服务选择 v981 direct-only，decode 保留 v964。C1 请求至完整 completion
生成速率为 **9.64 / 9.45 / 9.89 tok/s**，C4 含 prefill 的总吞吐为
**10.34 tok/s**。真实文本请求 HTTP 200、有非空光合作用解释。
四 rank 均为 clean graph、切换前后 weight-storage digest 相同、专家 fallback
为零；health/models HTTP 200、服务未暂停。当前日志为
`expert-cube-final-server-20261008.log`。

## 热切换与恢复

两个 serving recapture 在设备内存接近满额时停滞，完整算术门禁已经通过。
第一次发生在 worker 内完整验证后，第二次发生在标准小规模 admission 和
再次切换后。失败与恢复日志保留，未算作通过的 serving 测量。

resident harness 在丢弃旧图并 `gc.collect()` 后，新增
`torch.npu.empty_cache()`，在 replacement capture 前释放不再使用的缓存块。
真实权重及 KV tensor 保持存活；仅切换模式不做清理或重捕获。回归测试验证
清理顺序及 tensor 地址/内容保持。清理后 prefill capture 通过，但 decode capture 仍出现明确的 242 MiB
分配失败。清理缓存不能称为解决全部 recapture 问题。最终恢复先 admission 全部
resource、选定最终 prefill，再只做一次 replacement capture，避免中间的父版本
capture；根因仍未完整查清，失败记录均保留。

只停止身份核对过且已暂停的 GLM 进程树。保留全部八项融合、KDA OPP、
永久磁盘布局、TP4、EP/FlashComm1、target/draft A4、MTP1、prefix caching、
640 chunk、四槽、311040 配置 context 和 decode capture sizes 2/8。
禁用多模态；本批最大实测 6400 token，未证明配置最大容量。

## 本地检查和交付

1,521 个 focused CPU 测试通过，包括 17 个 Cube/builder 测试及新增的
allocator 清理顺序/存活 tensor 地址回归测试。本批文件 scoped hooks 通过。

运行了规定的 `bash format.sh ci`；repo 已有全局 lint 问题记录在压缩日志中，
其不相关格式修改已经撤销。仅本批文件的 hooks 通过。默认 pytest 遇到已有
upstream conftest 依赖缺失；CPU 门禁用本目录 confcutdir，NPU 门禁用真实 kernel
及 checkpoint，不把 mock 或 startup 当作推理通过。

冻结 binary 在 `bundles/`，编译源码及当前 candidate/launch 在 `sources/`，
算术、admission、切换、HTTP 时序在 `measurements/`，执行协议在 `protocols/`，
压缩原始输出在 `logs/`。`SHA256SUMS.json` 绑定全部交付文件。
参见[双语运行说明](RUNBOOK.md)。
