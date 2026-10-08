# GLM-5.3-Flash MTP-1 feasibility audit

Date: 2026-10-01. Source audit only; no NPU devices or full model were used.

Update 2026-10-04: an opt-in packed MTP adapter and rejection-state changes
are now staged with 60 passing CPU checks. See the
[candidate report](../mtp-packed-candidate-20261004/README.md).
The findings below describe the original audit; hardware qualification remains
pending and MTP remains disabled in the validated serving profile.

## Decision

**No-go for enabling MTP-1 in the current four-chip serving build.** The
source checkpoint declares one next-token prediction layer, and related local
manifests contain layer 45 tensors, but the current W2 target intentionally
skips those tensors. Its registered W2 MTP class is a stub that raises on
construction. The compact live KDA cache also rejects speculative decoding.
Removing that guard without a state/rollback design would risk incorrect
continuations after rejected drafts, preemption, or request-slot reuse.

This is a source readiness decision, not a finding that MTP-1 cannot work on
310P. Revisit only after eager quality and graph replay are qualified. Keep
MTP disabled by default.

## Evidence

- The exact served checkpoint on Threadripper,
  `/srv/ai/models/GLM-5.3-Flash-W4through32-noclip-310p`, has a safetensors
  index SHA-256 of
  `df84e6a7eccd6f0896455f0b35e78b6f5ac3cfae00d88e0b350115bbf109fefb`.
  Read-only manifest inspection found 75,575 total tensors, 1,753
  `model.language_model.layers.45.*` tensors, and no
  `layers.45.shared_head.head.weight`. Its config declares one
  `text_config.num_nextn_predict_layers`. The local FP8 and nearby W4
  manifests agree on the layer/head pattern; the exact checkpoint lives only
  on the NPU host in this workspace.
- `vllm_ascend/models/glm5next/mtp.py` detects whether an own MTP head was
  loaded and allows a target-head-sharing path. Absence of an own head alone
  is therefore not a reason to reject MTP; this sharing must be verified for
  the actual W2 target.
- `vllm_ascend/models/glm5next_w2/model.py` skips all layer-45 weights in the
  target loader and defines `Glm5NextW2MTP` as a registration-only class that
  raises. The W2 class registry points at that stub. The generic
  `patch_speculative_config.py` recognizes `glm5_next_mtp`, but that does not
  make the quantized draft runnable.
- `vllm_ascend/models/glm5next/cache_config.py` rejects speculative decoding
  with compact live Mamba/KDA specs. `vllm_ascend/_310p/model_runner_310p.py`
  allows its compact-state mapping under speculation only for the Qwen4Exp
  policy and rejects speculative decoding in GLM host-MLA mode. Host mode
  also requires eager execution. The current GLM graph path has not passed
  capture/replay parity.

## Required design before a candidate

1. Confirm layer-45 expert, attention, normalization, and shared-head tensors
   in the **exact** served quantized checkpoint. Define an MTP loader that
   uses the same mixed W2/W4 packed expert mapping as the target and explicitly
   binds the target LM head when the draft owns none. Do not infer head
   ownership from an allocated parameter.
2. Specify the recurrent-state slot count and promotion/rollback rules for
   every proposed token. A rejected draft must leave only the accepted KDA
   state and convolution state live. Cover request movement, preemption,
   block-ID reuse, padded decode rows, and multi-request batches.
3. Keep resident MLA and host MLA as separate configurations. MTP-1 is first
   tested only with resident MLA, eager baseline quality, and stable decode
   graph replay. Host-MLA MTP remains unsupported until its state and transfer
   behavior is designed separately.
4. Extend T13 ownership before implementation: likely
   `vllm_ascend/models/glm5next_w2/model.py`,
   `vllm_ascend/models/__init__.py`, and/or
   `vllm_ascend/patch/platform/patch_speculative_config.py`, in addition to
   the cache and runner files already listed. This needs an architectural
   review because the model runner and cache guards protect correctness.
5. With NPU access, compare deterministic MTP-off/on outputs, accepted draft
   tokens, verifier time, one/four-stream generated tokens/s, HBM, final
   answers, and state after rejection/preemption. Acceptance rate by itself
   is insufficient. If speed or quality worsens, leave MTP off.

## Validation

- Local manifest inspection passed for FP8, W4through34,
  W4through32-overlay, and W2 safetensors indexes. Read-only SSH inspection
  of the exact served checkpoint's index and config passed; no NPU process
  was started or interrupted.
- Source search for `speculative decoding`, `has_own_lm_head`, and
  `GLM_HOST_KV` completed.
- `python3 -m pytest -q --noconftest
  tests/ut/models/test_glm5next_mtp_rotation.py
  tests/ut/models/test_glm5next_cache_config.py` could not collect with the
  installed vLLM; it lacks `FusedMoEFactory` and
  `register_all_kvcache_specs` expected by this checkout.
- Re-running with `PYTHONPATH=/run/media/matteius/20TB-drive/vllm:$PWD`
  and normal unit-test stubs reached `tests/ut/conftest.py` but could not
  load it because `fla_npu` is not installed in the local CPU environment.
  These tests must be rerun in the matching development environment.
