# Qwen持续thermal shutdown与延后启动的请求hold

[US English report](REPORT.en.md)提供一致结果与限制。
之前短functional checks通过，但持续interactive使用未通过thermal资格。
10月8日**23:37:55 UTC**，watchdog读取96°C后向API PID 3820668发SIGTERM。
随后四个NPU均idle且降温；engine/workers停止后，API进程暂留等待HTTP
连接关闭。用户明确延后启动新server；本轮没有启动server或thermal controller。

## 观测证据

| 首次观测最高device温度 | UTC时间 |
| --- | --- |
| 80°C | 23:21:06 |
| 84°C | 23:25:05 |
| 88°C | 23:28:23 |
| 90°C | 23:30:20 |
| 94°C | 23:34:13 |
| 96°C | 23:37:43 |

watchdog独立采样，23:37:55动作；记录约5.8秒一份。完整log没有Mamba
spill warning、CPU→NPU checkpoint restore warning或decode graph fallback
warning。shutdown后没有最终worker counter snapshot，不能把无warning
当作实测transfer counter。

第二条长prefill加入decode后，generation反复降到0.1–0.6 tok/s，attention
KV usage上升。例如一条请求计算19486新prompt tokens耗时58.971秒，另一条
仍在运行。这与先前controlled test中串行model steps阻塞decode的模式一致。
仅temperature/request log不能区分host/NPU copies、device-local memory traffic
与compute对热量的贡献；有界checkpoint修复未证明持续thermal问题解决。

## 已实现fallback，实机gate仍待运行

controller读取`npu-smi info`暴露的四个310P chip温度；任意读数**>=94°C**时
发送`POST /pause?mode=keep&clear_cache=false`。固定版本vLLM设为`PAUSED_ALL`，
冻结active requests并让新请求排队，不abort、不清prefix/KV/Mamba cache、
不offload weights。已提交的work仍需到达engine pause边界才停止调度。

全部有效读数**<=85°C**后才恢复自己请求的pause。86–93°C保持hold。
缺失、partial、无效temperature要求hold且不允许resume。已有manual pause
不会被controller自动恢复。通过pause state处理丢失的HTTP acknowledgment；
resume失败不提前清thermal latch。loopback控制绕过HTTP proxy。

API PID start time防止PID复用；进程消失或复用时controller退出且不resume。
controller从不启动或重启server。原有**96°C emergency watchdog保留**。
pause API没有ownership token或pause mode，controller持有hold时需协调manual
pause/resume与resident reconfiguration；停止controller会保留pause。
降温期间client timeout仍可能到期。实际pause latency、cooling duration、
stream continuity、图片及NPU state保存仍需获准后的硬件验证。

## 离线验证与延后launcher

**24项offline tests通过**：94/85边界、全部device降温、hysteresis、无效
sensor、manual pause、pause/resume failure、丢ack、不完整HTTP响应重试、PID复用及实际loopback
HTTP的keep/no-clear query。fake engine测试不是实际cooling或NPU state验收。
Ruff、shell syntax与scoped formatting检查记录在本目录。
本轮code/documentation全部适用scoped hooks通过。按要求在隔离worktree
运行`bash format.sh ci`，因既有repository-wide Ruff、拼写、Clang、Markdown
及forbidden-import问题失败；未携带无关修改。原始log保持原样，不做拼写检查。

`start-fast-image-paced.sh`保留上一轮2560-token、三个slots、MTP2、`[3,9]`、
有界checkpoint与图片，在未来获准启动时挂载controller及原emergency watchdog。
control log为watchdog同目录的`thermal-control-8001.jsonl`；native HC仍使用
既有qualified resident transaction。module和延后launcher已stage到隔离runtime，
**二者均未执行**。配置见`deferred-profile.json`。

源码为`tools/qwen4exp/thermal_controller.py`与
`tests/ut/qwen38_1m/test_thermal_controller.py`。
原始证据为`incident.json`、`thermal.jsonl`、`watchdog.log`与`server.log`。
[温度与吞吐图](thermal.png)。
