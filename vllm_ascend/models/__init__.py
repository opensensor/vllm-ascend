from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "Gemma4ForConditionalGeneration",
        "vllm_ascend.models.gemma4_mm:AscendGemma4ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "KimiLinearForCausalLM",
        "vllm_ascend.models.kimi_k3:AscendKimiLinearForCausalLM",
    )
    # Keep the release-branch text architecture as a compatibility alias for
    # checkpoints whose config predates vLLM's KimiLinear rename.
    ModelRegistry.register_model(
        "KimiK3ForCausalLM",
        "vllm_ascend.models.kimi_k3:AscendKimiLinearForCausalLM",
    )
    ModelRegistry.register_model(
        "KimiK3ForConditionalGeneration",
        "vllm_ascend.models.kimi_k3:AscendKimiK3ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "KimiK3MTPModel",
        "vllm_ascend.models.kimi_k3_mtp:AscendKimiK3MTP",
    )
    ModelRegistry.register_model(
        "K3DSparkModel",
        "vllm_ascend.models.kimi_k3_dspark:AscendK3DSparkForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4.model:AscendDeepseekV4ForCausalLM"
    )
    ModelRegistry.register_model(
        "DeepseekV4ForConditionalGeneration",
        "vllm_ascend.models.deepseek_v4.vl_model:AscendDeepseekV4ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration",
    )
    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4.mtp:DeepSeekV4MTP")
    ModelRegistry.register_model(
        "DSparkDraftModel",
        "vllm_ascend.models.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
    ModelRegistry.register_model("Qwen3DSparkModel", "vllm_ascend.models.qwen3_dspark:AscendQwen3DSparkForCausalLM")
    ModelRegistry.register_model(
        "Qwen3OmniDSparkModel",
        "vllm_ascend.models.qwen3_dspark:AscendQwen3DSparkForCausalLM",
    )
    ModelRegistry.register_model(
        "DFlash2DraftModel",
        "vllm_ascend.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM",
    )
    ModelRegistry.register_model("DeepSeekMTPModel", "vllm_ascend.models.deepseek_mtp:AscendDeepSeekMTP")
    ModelRegistry.register_model("DeepseekV32MTPModel", "vllm_ascend.models.deepseek_mtp:AscendDeepSeekMTP")
    ModelRegistry.register_model("GlmMoeDsaForCausalLM", "vllm_ascend.models.deepseek_mtp:AscendGlmMoeDsaForCausalLM")
    ModelRegistry.register_model(
        "Eagle3LlamaForCausalLM", "vllm_ascend.models.llama_eagle3:AscendEagle3LlamaForCausalLM"
    )
    ModelRegistry.register_model(
        "Glm5NextForCausalLM",
        "vllm_ascend.models.glm5next.model:Glm5NextForCausalLM",
    )
    ModelRegistry.register_model(
        "Glm5NextForConditionalGeneration",
        "vllm_ascend.models.glm5next.model:Glm5NextForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Glm5NextMTPModel",
        "vllm_ascend.models.glm5next.mtp:Glm5NextMTP",
    )
    # GLM-5.3-Flash (glm5_next, 288 experts at 2-bit / W2) on Ascend 310P (plan
    # G3). An ADAPT of the shipped glm5next: the CausalLM subclasses the shipped
    # Glm5NextForCausalLM; the ConditionalGeneration alias rejects multimodal at
    # the first gate (text-only; GLM's model.visual.* tower is excluded);
    # Glm5NextW2MTPModel points at a registration-only MTP-1 stub wired in G7.
    # All rows are additive -- the shipped Glm5Next* registrations above are
    # untouched.
    ModelRegistry.register_model(
        "Glm5NextW2ForCausalLM",
        "vllm_ascend.models.glm5next_w2.model:AscendGlm5NextW2ForCausalLM",
    )
    ModelRegistry.register_model(
        "Glm5NextW2ForConditionalGeneration",
        "vllm_ascend.models.glm5next_w2.model:AscendGlm5NextW2ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Glm5NextW2MTPModel",
        "vllm_ascend.models.glm5next_w2.model:Glm5NextW2MTP",
    )
    ModelRegistry.register_model(
        "LlamaForCausalLMEagle3", "vllm_ascend.models.llama_eagle3:AscendEagle3LlamaForCausalLM"
    )
    # Qwen4Exp (Qwen3.8-Flash-Next) on Ascend 310P. MTP has a separate FP16
    # draft head; the 310P v2 runner still gates speculative decoding.
    ModelRegistry.register_model(
        "Qwen4ExpForCausalLM",
        "vllm_ascend.models.qwen4_exp.model:AscendQwen4ExpForCausalLM",
    )
    ModelRegistry.register_model(
        "Qwen4ExpForConditionalGeneration",
        "vllm_ascend.models.qwen4_exp.model:AscendQwen4ExpForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Qwen4ExpMTP",
        "vllm_ascend.models.qwen4_exp.mtp:AscendQwen4ExpMTP",
    )
    # DeepSeek V4.1 (552B, 2-bit / W2 experts) on Ascend 310P (plan E2.1). An
    # ADAPT of the shipped deepseek_v4: the CausalLM subclasses
    # AscendDeepseekV4ForCausalLM; the ConditionalGeneration alias rejects
    # multimodal at the first gate (text-only); DeepSeekV41MTPModel points at a
    # registration-only MTP-3 stub wired in E4.2. All rows are additive -- the
    # shipped DeepseekV4* registrations above are untouched.
    ModelRegistry.register_model(
        "DeepseekV41ForCausalLM",
        "vllm_ascend.models.deepseek_v41.model:AscendDeepseekV41ForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepseekV41ForConditionalGeneration",
        "vllm_ascend.models.deepseek_v41.model:AscendDeepseekV41ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "DeepSeekV41MTPModel",
        "vllm_ascend.models.deepseek_v41.mtp:DeepSeekV41MTP",
    )
