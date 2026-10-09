# NPU validation access status

Historical offline-stage report. Hardware validation followed; see [hardware source record](HARDWARE_SOURCE.json).

The user has authorized NPU validation. This execution session cannot open an
SSH socket to `192.168.53.187`: `socket: Operation not permitted`. Bypassing the
unreadable system SSH configuration with `-F /dev/null` reaches the same network
restriction. The session has restricted networking and no approval escalation.
No local Ascend devices or standard Ascend runtime paths were found.

No connection to the remote host was established. No server was stopped,
started or modified. No native binary was compiled and no NPU test ran.

A [13-file source archive](packed-state-source-20261008.tar.gz) is ready for
staging in a new isolated copy of the existing GLM source. Every archived file
was checked against the implementation's recorded SHA256. This archive is a
source overlay, not a built or hardware-qualified runtime bundle.

[Machine-readable status](NPU_ACCESS_STATUS.json) records both failed attempts,
the archive digest and pending validation steps. Resume with SSH-capable
execution; the user's authorization remains in place. First inspect current
server/device ownership, then rebuild and gate the v2 division kernel, validate
the actual writers, qualify the per-rank memory envelope, and complete
real-weight prefill/decode/prefix/MTP requests.

## 中文摘要

用户已经授权使用 NPU。本执行会话的网络沙箱阻止 SSH 建立 socket，错误为
`Operation not permitted`；使用 `-F /dev/null` 排除系统 SSH 配置问题后仍被阻止。
本地未发现 Ascend 设备或标准运行时路径。没有连接远程主机，没有更改服务器，
没有编译原生二进制，也没有运行 NPU 测试。

已准备并核对 13 个文件的源码归档及 SHA256。它是隔离源码副本的覆盖包，
不是已编译或硬件验证的部署包。需要具备 SSH 网络能力的执行会话才能继续；
无需用户重新授权。后续先核实设备/服务占用，再进行内核构建和门控、真实写入器
验证、每卡内存上界确认，最后完成真实权重的预填充、解码、前缀及 MTP 请求。
