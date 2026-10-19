import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

from vllm_ascend.models.llama_eagle3 import load_quarot_target_layer
from vllm_ascend.utils import (
    get_rotation_matrix,
    get_rotation_path,
    is_310p,
)

TARGET_EMBED_WEIGHT_NAMES = (
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
TARGET_LM_HEAD_WEIGHT_NAMES = (
    "language_model.lm_head.weight",
    "lm_head.weight",
)


# Process the first linear weight with rotation matrix, if the target model uses rotary quantization
def process_weight(linear_weight: torch.Tensor, rotation_weight: torch.Tensor):
    assert linear_weight.shape[1] % rotation_weight.shape[0] == 0, (
        f"Linear weight shape[1] must be a multiple of rotation weight shape[0],"
        f" but get {linear_weight.shape[1]=} and {rotation_weight.shape[0]=}"
    )
    rotation_weight = rotation_weight.to(device=linear_weight.device, dtype=torch.float32)
    hidden_size = rotation_weight.shape[0]
    ori_dtype = linear_weight.dtype
    processed_weight = torch.empty(linear_weight.shape, dtype=torch.float32, device=linear_weight.device)
    for start_pos in range(0, linear_weight.shape[1], hidden_size):
        linear_weight_chunked = linear_weight[:, start_pos : start_pos + hidden_size].to(torch.float32)
        processed_weight[:, start_pos : start_pos + hidden_size].copy_(
            torch.matmul(linear_weight_chunked, rotation_weight)
        )
    return processed_weight.to(ori_dtype)


@torch.no_grad()
def align_draft_weights(model, projection, vllm_config):
    """Align draft inputs with the rotated target without modifying shared weights."""
    rotation_path = get_rotation_path(vllm_config)
    if rotation_path is None:
        return
    rotation = get_rotation_matrix(rotation_path).cpu()
    weight = projection.weight
    weight.copy_(process_weight(weight.cpu(), rotation).to(weight.device))
    target_config = vllm_config.model_config.hf_text_config
    for owner, name, layer_cls, weight_names, own_flag in (
        (model.model, "embed_tokens", VocabParallelEmbedding, TARGET_EMBED_WEIGHT_NAMES, "has_own_embed_tokens"),
        (model, "lm_head", ParallelLMHead, TARGET_LM_HEAD_WEIGHT_NAMES, "has_own_lm_head"),
    ):
        if getattr(model, own_flag, False):
            continue
        with torch.device(weight.device):
            layer = layer_cls(target_config.vocab_size, target_config.hidden_size, params_dtype=weight.dtype)
        load_quarot_target_layer(layer, vllm_config.model_config.model, weight_names, rotation, f"draft {name}.weight")
        layer.quant_method.process_weights_after_loading(layer)
        setattr(owner, name, layer)
        setattr(model, own_flag, True)


class AscendQwen3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = self.config
        self.enable_confidence_head = bool(getattr(config, "enable_confidence_head", False))
        if is_310p():
            # Worker-only import: the 310P operator depends on torch_npu.
            from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310

            for module in self.model.modules():
                if isinstance(module, AscendRotaryEmbedding310):
                    module.use_model_owned_rope = True

    def prepare_target_vocabulary(self, vllm_config: VllmConfig) -> None:
        """Populate missing TP1 vocabulary modules before ordinary sharing.

        Called under the drafter's TP group. The default sharing path remains
        unchanged for equal TP sizes and other target families.
        """
        from vllm_ascend._310p.qwen4exp_draft_vocab import load_full_vocab_weight
        from vllm_ascend._310p.qwen4exp_mtp import is_qwen4exp_speculative_target

        spec = vllm_config.speculative_config
        if not is_qwen4exp_speculative_target(vllm_config.model_config, spec):
            return
        if vllm_config.parallel_config.tensor_parallel_size == spec.draft_tensor_parallel_size:
            return
        if spec.draft_tensor_parallel_size != 1 or vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("QFN DSpark vocabulary replication requires draft TP1 and target PP1")
        if get_rotation_path(vllm_config) is not None:
            raise ValueError("QFN DSpark TP1 vocabulary replication requires an unrotated target")
        config = vllm_config.model_config.hf_text_config
        if self.config.vocab_size != config.vocab_size or self.config.hidden_size != config.hidden_size:
            raise ValueError("QFN DSpark requires matching target and draft vocabulary and hidden size")
        if getattr(self, "draft_id_to_target_id", None) is not None:
            raise ValueError("QFN DSpark TP1 vocabulary replication does not support vocabulary remapping")
        pending = []
        for owner, name, layer_cls, names, flag in (
            (
                self.model,
                "embed_tokens",
                VocabParallelEmbedding,
                ("model.language_model.embed_tokens.weight", *TARGET_EMBED_WEIGHT_NAMES),
                "has_own_embed_tokens",
            ),
            (self, "lm_head", ParallelLMHead, TARGET_LM_HEAD_WEIGHT_NAMES, "has_own_lm_head"),
        ):
            if getattr(self, flag, True):
                continue
            with torch.device(self.model.fc.weight.device):
                layer = layer_cls(config.vocab_size, config.hidden_size, params_dtype=torch.float16)
            receipt = load_full_vocab_weight(
                layer, vllm_config.model_config.model, names, config.vocab_size, config.hidden_size
            )
            layer.quant_method.process_weights_after_loading(layer)
            pending.append((owner, name, layer, flag, receipt))
        # Publish only after both missing weights loaded successfully.
        self.target_vocab_load_receipts = {name: receipt for _, name, _, _, receipt in pending}
        for owner, name, layer, flag, _ in pending:
            setattr(owner, name, layer)
            setattr(self, flag, True)

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        if not self.enable_confidence_head:
            raise RuntimeError("The DSpark confidence head is disabled.")
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def configure_target_aux_hidden_capture(self, target_model: torch.nn.Module) -> None:
        """Select draft auxiliary inputs, without changing target Eager/Graph mode."""
        set_capture_mode = getattr(target_model, "set_dspark_aux_capture_materialized", None)
        if set_capture_mode is None:
            get_language_model = getattr(target_model, "get_language_model", None)
            if callable(get_language_model):
                set_capture_mode = getattr(get_language_model(), "set_dspark_aux_capture_materialized", None)
        if set_capture_mode is not None:
            set_capture_mode(True)

    def post_process(self, vllm_config: VllmConfig) -> None:
        align_draft_weights(self, self.model.fc, vllm_config)
