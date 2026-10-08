# GLM expert Cube runbook / 运行说明

Remote root: `/srv/ai/artifacts/glm-prefill-weight-reuse-20261007`.
Final server log: `expert-cube-final-server-20261008.log`.
Process identity receipt: `expert-cube-final-server-process-20261008.json`.

1. Source `/srv/ai/bin/ascend-env.sh`; use
   `/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python`.
2. Verify bundle/provenance/helper SHA and full gate reports before admission.
   Use unique native versions; never overwrite a registered binary or manifest.
   v979 selects populated Cube rows, v980 selects both options, v981 direct W4.
3. Check scheduler running/waiting counts and postpone for user requests. Full
   arithmetic/real-weight boundaries should run on free devices, avoiding large
   test graph allocations alongside the fully reserved model.
4. Use the standard hash-bound manifest for bounded worker admission. All four
   acknowledgments must pass. After a native admission failure restart only the
   identity-checked paused GLM tree; do not resume a failed native session.
5. The runtime `resident_worker.py` includes cache release after discarded graphs
   are collected. Preserve all prior native fusions, the KDA vendor slot, permanent
   disk scale marker, checkpoint and launch flags from the saved launch source.
   Cleanup does not guarantee all repeated captures fit: final recovery admits
   every resource and selects the final kernel before a single combined capture.
6. Swap only the selected prefill native object in the frozen combined factory;
   retain v964 decode and the v965 prepared bank/weight-layout resource. Recapture
   target and draft graphs, then resume. Verify source digest, clean graphs,
   zero native failure, worker IDs and unchanged weight-storage digests.
7. Test real HTTP inference, not startup alone. For cold prompts clear prefix state
   deliberately; for throughput report C1 generation and C4 full-request wall time
   separately. Synthetic token-ID responses are not a quality evaluation.
8. Confirm `/is_paused` is false and `/health` succeeds at handoff. Preserve failure
   receipts as well as the final successful requests. The saved source and raw
   evidence identify the tested state independently of future remote changes.

先校验 SHA、真实权重门禁及四 rank admission，再热切换并重捕获。设备忙则延期，
不取消用户请求。测试残留和旧图缓存可能消耗 recapture 空间；真实权重/KV 不应
被移动或改写。启动成功不能代替真实 HTTP 推理通过。每次交付保存英文及中文说明。
