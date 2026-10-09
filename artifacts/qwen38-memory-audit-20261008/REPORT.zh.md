# Qwen 310P 内存传输与同步审计

[US English report](REPORT.en.md)提供同一范围、结果和限制。
本轮仅离线审计：server保持停止，没有执行NPU workload、profiler capture、
pause/resume或model reconfiguration。94°C hold / 85°C resume仍仅stage，
96°C emergency cutoff保留，实机验证待后续授权。

## 结论

没有Mamba spill warning不等于没有传输或barrier。失败的serving profile仍有
MTP expert routing、PLE rows、GDN chunk plan、Mamba accepted-token与sampled
tokens的host/device边界，以及QSA/GDN/MoE/HC中大量device-local intermediates
和TP collective。最强的内存流量候选是QSA prefill K/V重复materialization；
最值得减少的host stalls是MTP routing readback、GDN plan readback和重复的
accepted-token边界。target FULL graph replay之前也仍有host stream drain。

这些是已核对dispatch的源码发现，并结合旧实机trace；不能由此确定10月8日
thermal shutdown的热量来源、能耗占比或实测当前profile吞吐。

## 范围与证据

扫描四个冻结source roots：fork `2e5f07071`的`vllm_ascend/csrc/tools`；
通过只读SSH取得的`qwen38-prefix-bounded-runtime-20261008`；OpenSensor vLLM
`3ab5dda29`对应的实际remote sources；已部署native HC bridge/kernel sources。
不包含其他agent的未提交实验。remote vLLM比clean commit多generated
`_version.py`，且`model_executor/kernels/mhc/torch.py`不同，已取得实际版本。
Qwen使用自己的HC实现，后者不在当前模型调用路径。

共扫描**6281 files、56161 candidate sites**，parse errors为0；手动dispatch
审查覆盖**36个serving及相邻区域**。这些counts包括inactive paths，不是实测
runtime operation counts。

deployed plugin与committed fork有26个Python文件不同，其中包括model.py，
故不能把fork已有优化当作全部已上线。
见[source hashes](source-manifest.json)、[coverage](coverage.json)、
[runtime differences](runtime-differences.json)和[upstream provenance](upstream-provenance.json)。
原始runtime source archive保持byte-for-byte。

`inventory.jsonl.gz`列出每个candidate的root/path/line/function/category/excerpt
与SHA256。它包括inactive models和diagnostic tools。CPU `.item()`不等于NPU
同步，同dtype `.to()`可能不复制，view可能不分配，device event wait也不等于
host blocking。native部分是lexical inventory，不能代替完整动态控制流测量。

## 五项优先发现

### 1. QSA prefill重复搬运selected K/V到每query scratch

`ops/qsa_batched_attention_310.py:132–350`以64-query tile把paged NZ K/V写入
selected scratch，再QK、FP32 scores/softmax、PV。两个gather streams及events
保护上一consumer和下一writer。这是NPU-local traffic，不是host offload。

TP4每rank为六个query heads、一个KV head；2048-budget加tail与NZ alignment
得到2064 tokens。K+V为每query `2×1×256×2064×2=2113536 bytes`。
64-query scratch为**129 MiB**，完整2560-token sparse chunk每QSA layer写
**5.039 GiB logical payload**，12个target QSA layers合计每rank **60.469 GiB**，
共有**960 K/V gather calls**。这些不是实测DDR bytes；cache reuse、早期context、
partial chunks会影响physical reads，且不含scores/indexer等其它work。
详见[logical estimates](logical-byte-estimates.json)。

10月5日四rank旧trace中value-gather占summed task time **11.02–11.28%**，
每rank4168次。可以支持优先研究，但不能当作speedup或heat占比。
候选是fused selected-page attention或query tile内group reuse并保留准确causal
mask。已有group-major使用`torch.unique`与动态shape，可能换来隐式host协调，
不是已qualified替换。直接删stream waits会破坏buffer安全。

### 2. MTP每draft forward仍读expert counts到host

`mtp.py:43–58,206–282,311–315`使用`nonzero`选local routes，sort/bincount后
`counts.tolist()`，由CPU dispatch小W8A16 expert matmuls。graph对此显式eager
callback，并copy到stable output。当前overrides未设置`mtp_expert_execution`，
因此实际为default **w8a16_routed**，不是grouped分支。

128个INT64 counts为每rank每draft forward **1 KiB**；MTP2重复两次，另有
dynamic route-size协调。bytes很小，但dependency boundary与小launch开销可观。
target W4则是device-routed decode/device-grouped prefill，host-routed
`.cpu().tolist()` fallback没有被此backend选择。

候选：qualification已有w8a8_grouped或fixed-shape device-routed W8A16。
W8A8改变activation quantization，需要acceptance、numerics、text/tool/image
及持续负载gate；只改config不等于所有host dispatch消失。

### 3. GDN chunk plan caching仍每metadata group做D2H

`gdn_310.py:151–175`通过`cu_seqlens.to(int64).cpu()`构建host plan。
memoization避免每layer重复，但36个GDN layers分为三个Mamba cache groups，
prefill/mixed的每group metadata和不同query-boundary tensor仍可能各发生一次。
不是36次每step，也不是完全消除。
可从scheduler已有CPU boundaries构建同一plan，准确匹配spec/non-spec partitions
与buffer lifetime；可先离线验证，不改变FP32 recurrent state。

### 4. Mamba align fallback另外读accepted counts到CPU

`patch_mamba_utils.py:177–265,393–405`选择310P tensor-copy fallback，用默认
blocking copy读取accepted counts后决定align copies。另有sampled-token CPU
bookkeeping，下一prepare又sync accepted-count event并H2D。这是不同边界。
对conv/SSM state执行`dst.copy_(src.clone())`的device copies不是prefix spill，
clone还承担overlap保护。

候选：在request row ownership和reset-to-one语义都一致时复用已完成host snapshot，
或用fixed-shape device operator处理align/copy。不能在未证明storage disjoint时
直接去clone，也不能混淆原始accepted count与postprocess重置值。

### 5. target FULL replay仍有host stream synchronize

`compilation/breakable_aclgraph.py:74–91`在ENPU关闭且非draft例外时同步main
stream。310P runner已有update/main stream waits，但最终host drain仍存在。
pinned vLLM把MTP视为EAGLE-style，draft有例外；不能宣称MTP2每轮同样drain三次。
删除前必须证明previous replay、mutable task params、update submission和current
replay的完整顺序。单个device event不能证明host-side graph update安全。

## 其它区域审查

完整anchors见`reviewed-areas.json`，英文报告给出逐项phase与处理边界。

| 区域 | 发现与边界 |
| --- | --- |
| Prefix retirement/layout | 请求slot变化device-wide drain；bounded retirement在groups间共用一次。invalidate/CoW/admission仍可额外drain，不能删writer保护。 |
| Device archive/spill | 每group checkpoint为9744384 bytes。D2D archive/swap与D2H/H2D仍有分支；bounded不意味着所有D2D都消失。 |
| Compact Mamba tables | 每prepare经host→NPU temporary→stable table；可改bounded pinned staging和单copy，保留tails/remapping正确性。 |
| 输入metadata | tokens/positions/slots多次小H2D，大多已有persistent buffers。CPU mirrors上的tolist/numpy不算NPU readback。PLE history是普通CPU allocation。 |
| Embedding staging | 大FP16 CPU buffer故意不用pinning，原因是已复现AVX2 zero-fill crash；不能全面启用pinning。 |
| Sampled output/logprobs | sync mode读取NPU token IDs供scheduler/history/streaming。logprobs按需增加readback；Qwen MTP/PLE async scheduling当前被拒绝，不能只开flag。 |
| QSA positions/select | 小host position H2D；compressed keys gather/FP32 score/sort；已有visible-page bounds与可兼容forward cache。eager callbacks不一定产生whole-decode fallback warning。 |
| QSA ND fallback/cache writes | ND转换会复制visible pages，但2048-budget长sparse prefill用直接NZ gather。native cache writes为device-side，reference loops不是live证据。 |
| MoE count/dispatch | 25600 routes×128 experts为3276800 comparisons/layer/chunk。quant packing已按token共享；peer sentinel仍占route geometry。 |
| MoE scratch/finalizer | gate/up62.5 MiB、down125 MiB每full chunk/rank。cann_v2已避免完整FP32 route-output，combined result仍FP32 TP reduce，舍入差异已知。 |
| Shared expert streams | 当前tp_sharded，不走overlap/deferred分支，不能把其events计为live stalls。 |
| GDN WY/layout | 仍有contiguous/FP32 transforms/inverse scratch、state transpose/gather。grouped Gram已减key重复；FP32 recurrence和生产head geometry必须保留。 |
| HC mix/native residual | mix/norm仍FP16↔FP32 activation materialization；weights/affines已cache，linear不反复copy全权重。native bridge当前stream launch，无显式host memcpy/sync；kernel PIPE barriers在AI Core内部。 |
| Native kernels | W4/GDN/QSA有GM/L1/L0/UB DMA和producer/consumer flags；resident weights仍被kernel读取。external OPP binary不能假定等同当前fork source。 |
| TP/HCCL/head | target48 attention+48MoE=96decoder reduction sites/forward，另有embedding/head/draft traffic。custom W4没有CPU expert all-to-all。 |
| Startup quant/load | W8 head/PLE projection准备一次并保留；draft共享embedding/head。load-time CPU/device copies不能单独解释steady-state反复升温。 |
| Images | fresh encoder/pixels staging及vision scratch仅在scheduled encoder输入时发生，不是每decode token。cached embeddings复用，图片保持启用。 |
| Zeroing/CoW | 新attention pages写零/shared-prefix block copies是初始化和隔离，不是offload；按block bytes计量。 |
| Optional features | 当前未启用CPU offload、KV transfer、PP/DCP/PCP、dynamic EPLB、FlashComm1/cache parallelism；这些inventory sites不归因于本incident。 |
| Admin/profiling | reset/recapture/native validation的drains与普通inference分开。thermal hold使用keep，不清cache/offload weights。 |

## 离线复现的latent fallback defect

`gdn_310.py:80`在二维state indices、非uniform、非capture时提前访问尚未赋值
的`seq_lens`。extracted-function CPU probe复现UnboundLocalError；同fallback还把
`flat_cpu.is_pinned`当attribute而未调用method，pinning条件因此不生效。
uniform MTP bypass此分支，不能归因于本shutdown。依赖variable-shape fallback
前应修复并做regression。见[offline proof](offline-latent-gdn-proof.json)。

## 历史实机证据与限制

[历史receipts](historical-profiler-receipts.json)离线重算10月5日四rankexports：
cast/layout/copy为summed task time **12.37–12.72%**，collectives **9.09–10.92%**，
native W4 projections **19.88–21.96%**。rank1有12次Event::synchronize，host
self duration合计**3642.51 ms**；8次nonzero为**433.94 ms**；128个H2D和8个D2H
records。没有copy byte sizes；names无法唯一映射source，nested/cross-stream可
overlap。这些trace早于bounded scheduler及当前native HC，不能当当前性能结果。

旧DDR exports在device0/1按MB/s标示有不合理数值；PCIe transaction-class rates
也不等于application memcpy bytes，未拿来宣称实测bandwidth或heat attribution。
330 prompt tok/s下，2560 tokens仅prefill就约7.76秒，符合mixed时decode延迟，
但不是memcpy stall测量。少windows可减contention，不会自动消除单请求持续
升温；小chunk也不保证总energy/throughput改善。

## 后续顺序与验证

先使用host metadata消除GDN plan readback、合并table staging并修latent defect；
再qualification device MTP routing与accepted-count复用；随后测量QSA/MoE/state
D2D payload和graph/retirement wait，使用host-owned sizes与phase labels，不新增
tensor.item或timing sync。再研究QSA group reuse与GDN/HC fusion，最后在ordering
contract证明后优化graph drains。

[profiling-plan.json](profiling-plan.json)将cold/warm prefill、C1/C2/C3 decode、
mixed与fresh/cached images分开，短all-rank traces记录copy direction/bytes、event
intervals、MTE/HCCL和CPU sampling，并与thermal controller共用时间基准。
plan disabled，不启动server；实机需未来授权。CANN/PyTorch/driver内部、binary
kernel、隐式dynamic-shape sync与physical heat仍是测量缺口。

**11项offline regression checks通过**：8项scanner及3项profiler receipt，
包括self-duration accounting与无效duration rejection。历史receipts和logical formulas已
离线运行。scope checks及required全repo format.sh ci结果随report保存。
全repo因既有Ruff、拼写、Clang、Markdown和forbidden-import问题失败；无关自动
格式修改限制在隔离check worktree，没有带入交付。
原始source/trace证据不做拼写修正。本轮只交付analysis tooling/tests/reports，
没有声称新的NPU性能或thermal qualification。复现命令见英文报告。
