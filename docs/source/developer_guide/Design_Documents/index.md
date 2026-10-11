# Design Documents

This section provides an overview of the features implemented in vLLM Ascend. Developers can refer to this guide to understand how vLLM Ascend works.

- [KVPP: KV Cache Layer Parallelism](kvpp.md) — Physical cache placement, full-layer broadcast, and test design.
- [Persistent Incremental Csrc Build Cache](persistent_csrc_build_cache.md) — Fine-grained native build reuse, correctness identity, concurrency, and CI integration.
- [GLM 5.3 Flash Performance and Context Plan for Four Ascend 310P Chips](glm53_flash_310p_performance_prd.md) — Measured baseline, acceptance targets, optimization sequence, and context qualification.
- [GLM CUDA/Ascend Sidecard Research](glm_sidecard_research.md) — Prior design, hardware inventory, resident expert placement, and integration gates.
