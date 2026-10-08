# Offline GLM candidate runbook / 离线候选说明

This batch does not change the server. Hardware work is deferred until authorized
for the active device workload. Preserve the existing FP16-scale disk checkpoint,
custom OPP/libopapi stack, TP4, MTP, graph captures, cache configuration and port.

1. Verify `SHA256SUMS` and `SUMMARY.json`. Allocate unused immutable build
   versions. Unpack `frozen-sources.tar.gz` into a new staging directory; verify
   each member hash in the summary. Never overwrite loaded modules, manifests or vendor binaries.
2. Expert: clone the exact v984/v985 build options from their provenance. Build
   event-only, bulk-store-only and combined candidates independently. Add
   `--product-pipe-events` and/or `--bulk-route-store`. The latter requires
   `--fused-moe --fp16-route-workspace --native-route-columns`; these are already
   present in the retained native prefill schedule. Decode still needs its own
   event-only comparison. Require actual SDK builds for all W3/W4 stage binaries.
3. Run `test_prefill_memory_reuse_310.py` with matching parent/candidate bundle
   arguments, then complete real-expert projection/route checks and changed graph
   replays. Test route tails, zero/peer weights, both activation limbs and all
   retained shape specializations. Check CO1 readback safety over repeated calls.
4. QSA: build the v991 comparator with `--vector-output --vector-accumulate` and
   build the candidate with the additional `--softmax-head-batch`. Compare their
   `shared.bin` through the same tiling wrapper. The builder's `parent.bin` is the
   older original QSA reference, not the v991 schedule. Use the full 36-case
   `qsa_shared_probe.qualify` and nightly QSA test with the installed production
   operator, independent dense reference, partial heads and changed graph inputs.
   Require exact parent equality before accepting any speed claim. Specifically
   inspect batched Exp, paired max outputs, sum strides and event 7 on 310P.
5. KDA: stage against the frozen deployed parent, preserving the full row-batch
   OPP build protocol in the earlier `prefill-score-batching-20261008` artifact.
   Add `GLM_KDA_SCORE_MATRIX_BATCH` to its existing compiler defines. The staging
   CLI requires `--source`, `--destination`, `--expected-sha256` and `--report`.
   Require the full KDA operator gate for outputs and recurrent state, full and
   partial chunks, lower-bound gates and changed replays. Keep the two reduction
   trees separate; do not substitute a different dot-product instruction.
6. Profile and benchmark each candidate separately before combining them.
   Compare cold 640/1280/6400-token prefill with cleared prefix state, long-context
   attention, C1 and C4 generation, keeping wrapper/startup warmth consistent.
   Record MTP acceptance and graph state. Task sums are not critical-path time.
7. Admit only fully qualified candidates while no user requests are running or
   queued. Never cancel a user request. Kernel OPP replacement requires the
   existing identity-checked restart procedure. Recapture after binding changes
   and verify all four rank receipts, unchanged weight-storage hashes, no eager
   expert fallback and no native failures. Recover the qualified parent on gate
   failure, then check real HTTP output, health and unpaused state at handoff.

先分别编译和门禁，再测单项及组合。CPU 模拟没有证明 NPU 执行、数值质量或速度；
设备 busy 时延期。当前服务保留已认证版本，本轮只交付离线候选及完整源哈希。
