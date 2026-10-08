# GLM：三组后续离线优化

本目录记录离线阶段；后续 NPU 门禁与服务测速见
[设备交付报告](../expert-softmax-kda-device-20261008/README.zh.md)。

三组候选均已实现，默认关闭。GLM 独立 CPU 测试共 1,585 项通过。
本轮未编译 CANN 内核、未使用 NPU、未加载真实权重进行资格认证，也没有测速。
服务进程未安装本轮候选。不能把主机模拟测试当作设备正确性或模型质量证明。

## 专家投影与路由写入

构建工具新增独立开关 `--product-pipe-events` 和 `--bulk-route-store`。
前者用 M/V、V/M 事件替代 Product 中两次全流水线屏障，保护 CO1 读回和复用。
其他权重、激活生命周期屏障保留，没有引入双缓冲。

后者仅用于 native-column FP16 路由工作区的 down 投影，将逐行写入改为一次
带跨行步长的 DMA。W4 特化 down 同样受控；gate/up 不启用该宏，decode 分支
保持原实现，FP16 舍入点和后续 FP32 加权累加顺序不变。

主机测试编译实际 Product/Combine 函数，验证事件调用顺序、部分行的完整写入
位置及 decode 累加结果。主机 Cube 和流水线原语使用模拟实现，设备门禁仍待运行。

## QSA softmax 按头批处理

在 v991 共享 K/V、输出批处理及累加批处理基础上，新增
`--softmax-head-batch`，对应 `GLM_QSA_SOFTMAX_HEAD_BATCH`。
最大值/索引按双元素结果布局读取；求和按单元素布局读取。在线 softmax 状态
沿用原 FP32 标量表达式，概率仍在原边界转 FP16。每核新增 UB 256 字节。

事件 7 必须未被父版本占用。测试覆盖 1/2/3/4/12/16 个头、五种 tile 长度及
连续三轮状态更新，比较完整输出缓冲区。夜间设备测试增加六个部分头数案例，
完整门禁共 36 例并包含输入变化后的图重放。本轮未运行设备门禁。

## KDA score 矩阵批处理

`stage_kda_score_matrix.py` 仅修改带列缓存的 FP16 K=128、BT=64 完整 chunk。
行操作数用零 repeat 步长广播，将列减法、两次乘积及两组归约改为 repeat 指令。
保留两个独立 64-lane FP32 归约树、原 `+0 + p0 + p1` 顺序及 FP16 中间舍入。
其他形状与尾块保持原路径。

复用原 scratch：FP16 乘积 16–32 KiB、FP32 乘积 32–64 KiB，列缓存从
80 KiB 开始；新增 UB 为零。测试执行实际完整分支，比较两张 64x64 score
矩阵，检查不可变列缓存、禁用时预处理一致性和无效组合拒绝。设备数学原语
在主机测试中为模拟实现。

冻结父头文件来自已部署的完整 row-batch 包，SHA256：
`131bfc255e0e66bf528f51b759cea529af3c2823c442e3dcf1f6f767d1c75a2c`。
候选保留已有 KDA 融合；新宏必须同时启用 row-batch、vector-reduce、gated-key
reuse 和 column-cache 四个父版本特性。

## 后续验证

独立 CPU 命令为
`pytest --confcutdir=tests/ut/glm_perf -q tests/ut/glm_perf`。
设备恢复后按 [RUNBOOK.md](RUNBOOK.md) 分别编译、资格认证、计时，再检查组合。
未通过真实权重和四 rank 图重放门禁前不得加载服务。所有源文件、哈希和状态见
[SUMMARY.json](SUMMARY.json)；本轮没有宣称任何速度提升。
