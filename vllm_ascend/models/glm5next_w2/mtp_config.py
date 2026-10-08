# SPDX-License-Identifier: Apache-2.0
"""Host-only configuration contracts for the experimental packed GLM drafter."""

from copy import copy

PACKED_GLM_TARGET_ARCHITECTURES = (
    "Glm5NextW2ForCausalLM",
    "Glm5NextW2ForConditionalGeneration",
)
PACKED_GLM_MTP_ARCHITECTURE = "Glm5NextW2MTPModel"
MAX_GLM_MTP_DRAFT_TOKENS = 7  # 310P recurrent kernel accepts at most eight verifier tokens.


def normalize_glm_mtp_hf_config(config, *, packed=False):
    """Flatten the text tower before upstream's MTP architecture conversion.

    Copy both wrapper and text overrides so normalization does not mutate the
    target config shared with the serving model.
    """
    text = getattr(config, "text_config", config)
    normalized = copy(text)
    for name, value in vars(config).items():
        if name.startswith("ascend_glm_"):
            setattr(normalized, name, value)
    if getattr(normalized, "quantization_config", None) is None:
        quant = getattr(config, "quantization_config", None)
        if quant is not None:
            normalized.quantization_config = quant
    count = getattr(normalized, "num_nextn_predict_layers", None)
    if not isinstance(count, int) or count < 1:
        raise ValueError("GLM MTP requires a positive num_nextn_predict_layers")
    normalized.model_type = "glm5_next_mtp"
    normalized.n_predict = count
    normalized.architectures = [PACKED_GLM_MTP_ARCHITECTURE if packed else "Glm5NextMTPModel"]
    return normalized


def validate_packed_glm_mtp(vllm_config):
    """Keep the first candidate on the historical, block-aligned state path."""
    spec = vllm_config.speculative_config
    if spec is None or spec.method != "mtp":
        raise ValueError("Packed GLM draft requires speculative method 'mtp'")
    count = spec.num_speculative_tokens
    if not isinstance(count, int) or not 1 <= count <= MAX_GLM_MTP_DRAFT_TOKENS:
        raise ValueError("Packed GLM MTP requires 1..7 draft tokens")
    if vllm_config.cache_config.mamba_cache_mode != "align":
        raise ValueError("Packed GLM MTP requires mamba_cache_mode='align' for accepted-state copies")
    if getattr(vllm_config.scheduler_config, "async_scheduling", False):
        raise ValueError("Packed GLM MTP initially requires synchronous scheduling")
    if getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) != 1:
        raise ValueError("Packed GLM MTP initially requires pipeline_parallel_size=1")
    if not vllm_config.model_config.enforce_eager:
        if count != 1 or vllm_config.scheduler_config.max_num_seqs > 4:
            raise ValueError("Packed GLM full graphs initially require MTP1 and at most four sequences")
        hf_config = vllm_config.model_config.hf_config
        if not getattr(hf_config, "ascend_glm_mtp_full_graph", False):
            raise ValueError("Packed GLM graph replay requires ascend_glm_mtp_full_graph and the native state kernel")
