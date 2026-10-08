# GLM score-row-batch runbook / 运行说明

Current remote root:
`/srv/ai/artifacts/glm-prefill-weight-reuse-20261007`.
Experiment: `prefill-kda-score-row-batch-full-v2`.
The recorded process receipt is `score-row-batch-server-process-20261008.json`;
the server log is `score-row-batch-server-20261008.log`.

1. Source `/srv/ai/bin/ascend-env.sh`. Use the existing
   `/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python` interpreter.
2. Stage only against parent header SHA
   `a1dc064bef56b6bae800ed4ce679e146f213d48561df641db0d3700dbe857429`.
   Build v2 with the frozen protocol and all six recorded KDA compiler defines.
   The production source header remains unchanged. Verify source/package binary
   SHA against `measurements/build.json` before any admission.
3. Qualify on free devices against the frozen full-operator reference. Do not
   repeat the failed second-context attempt alongside the fully reserved server.
   Check running/waiting request counts first; stop only the identity-checked GLM
   process tree. The frozen free-device protocol relaunches the parent on failure.
4. Replace only the KDA vendor slot. Preserve all other OPP and host-library
   slots, checkpoint, launch overrides and flags from the saved launch source.
   Native OPP binary changes require a fresh server process.
5. After real-weight startup, run the frozen restore protocol to admit the eight
   native resources, apply the combined source, recapture and resume. Verify all
   four receipts: matching source digest, clean graphs and no native failures.
6. Run the frozen benchmark with no competing user requests. Cold measurements
   deliberately clear scheduler prefix state. Measure full first-text TTFT and
   decode through request completion, not only first-to-last visible fragment.
   C4 total throughput includes prefill; synthetic token-ID prompts are not a
   representative language-quality evaluation.
7. Optional CANN profiling uses the existing private profiler footer. Restore
   the qualified source in `finally`, retaining worker/weight-storage identities.
   Profiler task sums are not critical-path latency or unprofiled throughput.

本次沿用用户既有 port 8001、真实模型和四槽配置。设备 busy 时延期测试，不取消
用户请求。停止/重启仅针对记录身份的 GLM 树。资格认证失败恢复父版本，不能把
编译成功或 startup 成功当作推理通过。完整 NPU 日志、冻结协议和包保留于本目录。
