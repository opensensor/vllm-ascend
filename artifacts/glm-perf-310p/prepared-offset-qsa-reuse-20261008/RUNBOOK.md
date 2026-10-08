# GLM offset / QSA runbook / 运行说明

Remote root: `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007`.
Server log: `prepared-offset-server-20261008.log`.
Identity receipt: `prepared-offset-server-process-20261008.json`.

1. Source `/srv/ai/bin/ascend-env.sh`; use
   `/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python`. Retain the saved custom
   OPP/libopapi stack and permanent FP16-scale checkpoint. The saved launch source
   captures the actual port/context/cache/MTP options; do not substitute defaults.
2. Verify bundle/helper/provenance and full gate report hashes. Unpack frozen
   bundles to their own immutable directories. Never overwrite registered
   binaries or change a loaded manifest. v984/v985 add only prepared offset
   tables to decode v964 / prefill v981 schedules. Original v965 still owns
   prepared-bank admission and layout bookkeeping.
3. Defer while scheduler running/waiting counts are nonzero. Pause/wait for idle
   admission; do not cancel user requests. Full expert gates require free-device
   memory; resident admission intentionally uses bounded fixtures. NativeSession
   failure requires fresh identity-checked workers before another native load.
4. QSA build uses `tools.glm_perf.build_qsa_shared` with a unique positive version,
   qualified source root, CANN root and the existing ACLRTC compiler executable.
   Parent/shared binaries use the same direct tiling wrapper. v991 enables
   `--vector-output --vector-accumulate` in addition to shared K/V reuse. The frozen
   `qsa_shared_probe.qualify` compares both binaries and the installed operator.
   Run `test_qsa_shared_cache_310.py --qsa-build-dir <bundle>` directly for the
   complete nightly hardware regression (same custom OPP/libopapi environment).
5. Admit expert v984/v985 and QSA v991 with the saved hash-bound manifests. Require
   all four acknowledgments and clean native sessions. QSA uses reversible
   per-instance operator getters, not a class staticmethod patch. Geometry
   descriptors are prepared before capture from CPU tensor metadata.
6. Use the frozen combined source, retaining every prior fusion and the qualified
   KDA vendor. Select the final candidate before capture, resume and verify all
   rank source digests, worker IDs, weight-storage digests and graph state.
7. Clear prefix state deliberately for each cold timing. Compare QSA parent/shared
   through the same wrapper; distinguish startup/allocation warmth and MTP
   acceptance effects. C1 generation and C4 full-request throughput are separate.
8. Check a real-text HTTP response, `/health` and unpaused state at handoff.
   Configured context is 311040 with four slots, not an exhaustively tested
   capacity claim. The retained current candidate and receipt identify the
   tested state independently of later server changes.

先验 SHA 和真实权重门禁，再做四 rank admission / capture；设备忙则延期。
失败 native session 不能继续加载，必须核对身份后重建。冷测试清 prefix 状态，
合成 token 测速不能替代质量评测，最终恢复真实 HTTP 可用且未暂停的服务。
