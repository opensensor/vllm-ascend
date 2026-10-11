# SPDX-License-Identifier: Apache-2.0
"""Explicit expert ownership; independent of CUDA, CANN, and vLLM imports."""

from dataclasses import dataclass

SCHEMA_VERSION = 1
CUDA_OWNER = "cuda:0"
ASCEND_OWNERS = ("npu:0", "npu:1", "npu:2", "npu:3")
GLM_LAYERS = 45
GLM_FIRST_MOE_LAYER = 3
GLM_EXPERTS = 288
GLM_TOP_K = 8
GLM_HIDDEN = 4096
GLM_INTERMEDIATE = 2048
GLM_ROUTED_SCALE = 2.5
DEFAULT_CUDA_EXPERTS = 72


@dataclass(frozen=True)
class ExpertPlacement:
    """One owner for every expert in every routed layer; no implicit TP layout."""

    owners: tuple[str, ...]
    layers: tuple[tuple[str, ...], ...]
    first_moe_layer: int = GLM_FIRST_MOE_LAYER

    def __post_init__(self):
        if not isinstance(self.owners, tuple) or not isinstance(self.layers, tuple):
            raise ValueError("placement containers must be immutable tuples")
        if any(not isinstance(layer, tuple) for layer in self.layers):
            raise ValueError("layer ownership must be an immutable tuple")
        if not self.owners or len(set(self.owners)) != len(self.owners):
            raise ValueError("owners must be unique and nonempty")
        if any(not isinstance(owner, str) or not owner for owner in self.owners):
            raise ValueError("owner names must be nonempty strings")
        if type(self.first_moe_layer) is not int or self.first_moe_layer < 0:
            raise ValueError("invalid first MoE layer")
        if not self.layers or not self.layers[0]:
            raise ValueError("placement must contain routed layers and experts")
        experts = len(self.layers[0])
        if any(len(layer) != experts or any(owner not in self.owners for owner in layer) for layer in self.layers):
            raise ValueError("incomplete expert coverage or unknown owner")

    def layer(self, layer_id: int) -> tuple[str, ...]:
        if type(layer_id) is not int or not self.first_moe_layer <= layer_id < self.first_moe_layer + len(self.layers):
            raise ValueError("layer is outside routed placement")
        return self.layers[layer_id - self.first_moe_layer]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "owners": list(self.owners),
            "first_moe_layer": self.first_moe_layer,
            "layers": [list(layer) for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, value: dict):
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported placement schema")
        return cls(tuple(value["owners"]), tuple(tuple(layer) for layer in value["layers"]), value["first_moe_layer"])


def glm_placement(cuda_experts: int = DEFAULT_CUDA_EXPERTS) -> ExpertPlacement:
    """Contiguous seed placement; expert popularity and cost are not inferred."""
    if type(cuda_experts) is not int or not 0 <= cuda_experts <= GLM_EXPERTS:
        raise ValueError("CUDA expert count must be between 0 and 288")
    remaining = GLM_EXPERTS - cuda_experts
    base, extra = divmod(remaining, len(ASCEND_OWNERS))
    layer = (CUDA_OWNER,) * cuda_experts
    for index, owner in enumerate(ASCEND_OWNERS):
        layer += (owner,) * (base + (index < extra))
    return ExpertPlacement((CUDA_OWNER, *ASCEND_OWNERS), (layer,) * (GLM_LAYERS - GLM_FIRST_MOE_LAYER))


def validate_glm_config(config: dict) -> None:
    """Reject a different GLM geometry rather than silently applying this seed."""
    text = config.get("text_config", config)
    expected = {
        "model_type": "glm5_next_text",
        "num_hidden_layers": GLM_LAYERS,
        "first_k_dense_replace": GLM_FIRST_MOE_LAYER,
        "n_routed_experts": GLM_EXPERTS,
        "num_experts_per_tok": GLM_TOP_K,
        "hidden_size": GLM_HIDDEN,
        "moe_intermediate_size": GLM_INTERMEDIATE,
        "routed_scaling_factor": GLM_ROUTED_SCALE,
    }
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f"unsupported GLM config: {key} must be {value!r}, got {text.get(key)!r}")


def storage_lower_bound(placement: ExpertPlacement, bits: int = 4) -> dict[str, int]:
    """Logical packed codes only. Scales, padding, dense weights, and caches are extra."""
    if bits not in (2, 3, 4):
        raise ValueError("expert bits must be 2, 3, or 4")
    if (
        placement.first_moe_layer != GLM_FIRST_MOE_LAYER
        or len(placement.layers) != GLM_LAYERS - GLM_FIRST_MOE_LAYER
        or len(placement.layers[0]) != GLM_EXPERTS
    ):
        raise ValueError("storage estimate requires the main-model GLM placement")
    parameters = 3 * GLM_HIDDEN * GLM_INTERMEDIATE
    return {
        owner: sum(layer.count(owner) for layer in placement.layers) * parameters * bits // 8
        for owner in placement.owners
    }
