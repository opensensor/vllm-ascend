# SPDX-License-Identifier: Apache-2.0
"""Opt-in safetensors reader for TP-contiguous GLM W2 expert banks.

The model's own ``load_weights`` still performs all remapping and validation.
This reader only avoids materializing nonlocal decoder expert codes/scales.
Other tensors, including vision and MTP, follow the ordinary loader path.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from collections.abc import Generator

import regex as re
import torch
from safetensors import safe_open
from torch import nn
from tqdm.auto import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import _BAR_FORMAT, _natural_sort_key, enable_tqdm

from vllm_ascend.models.glm5next_w2.moe import _ep_rank_size, ep_expert_range
from vllm_ascend.models.glm5next_w2.mtp_config import PACKED_GLM_MTP_ARCHITECTURE

LOAD_FORMAT = "glm_w2_filtered"
_EXPERT_TENSOR_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?:gate_proj|up_proj|down_proj)_(?P<kind>codes|scale)$"
)
_KIND_DTYPE_BYTES = {"codes": ("U8", 1), "scale": ("F32", 4)}


def local_decoder_expert_range(model_config: ModelConfig) -> tuple[int, int, int, int]:
    """Validate GLM W2 geometry and return local expert and layer bounds."""
    is_draft = model_config.architectures == [PACKED_GLM_MTP_ARCHITECTURE]
    if not is_draft and model_config.architectures != ["Glm5NextW2ForCausalLM"]:
        raise ValueError(f"{LOAD_FORMAT} requires Glm5NextW2ForCausalLM or packed GLM MTP architecture")
    text = model_config.hf_text_config
    if not is_draft and (
        getattr(model_config.hf_config, "model_type", None) != "glm5_next"
        or (getattr(text, "model_type", None) != "glm5_next_text")
    ):
        raise ValueError(f"{LOAD_FORMAT} requires a glm5_next text checkpoint")
    experts = getattr(text, "n_routed_experts", None)
    layers = getattr(text, "num_hidden_layers", None)
    dense_layers = getattr(text, "first_k_dense_replace", None)
    if not all(isinstance(value, int) for value in (experts, layers, dense_layers)):
        raise ValueError("GLM text config lacks integer expert/layer geometry")
    if experts <= 0 or layers <= 0 or not 0 <= dense_layers < layers:
        raise ValueError("invalid GLM expert/layer geometry")
    ep_rank, ep_size = _ep_rank_size()
    if ep_size <= 0 or not 0 <= ep_rank < ep_size:
        raise ValueError("invalid TP/EP rank geometry")
    lo, hi = ep_expert_range(ep_rank, ep_size, experts)
    if is_draft:
        count = getattr(text, "num_nextn_predict_layers", None)
        if not isinstance(count, int) or count < 1:
            raise ValueError("GLM draft config lacks next-token prediction layers")
        return lo, hi, layers, layers + count
    return lo, hi, dense_layers, layers


def should_skip_glm_expert(
    name: str,
    *,
    local_range: tuple[int, int],
    dense_layers: int,
    decoder_layers: int,
    num_experts: int,
) -> tuple[bool, str | None]:
    """Classify only exact decoder W2 code/scale names before get_tensor."""
    match = _EXPERT_TENSOR_RE.fullmatch(name)
    if match is None:
        return False, None
    layer = int(match["layer"])
    expert = int(match["expert"])
    if not 0 <= expert < num_experts:
        raise ValueError(f"GLM checkpoint has out-of-range expert in {name}")
    if layer >= decoder_layers:
        # The unwired MTP head is intentionally left to model.load_weights.
        return False, None
    if layer < dense_layers:
        raise ValueError(f"GLM checkpoint has routed expert in dense layer: {name}")
    lo, hi = local_range
    return not lo <= expert < hi, match["kind"]


def skipped_tensor_bytes(tensor_slice: object, kind: str) -> int:
    expected_dtype, bytes_per_element = _KIND_DTYPE_BYTES[kind]
    dtype = tensor_slice.get_dtype()  # type: ignore[attr-defined]
    if dtype != expected_dtype:
        raise ValueError(f"GLM {kind} tensor has dtype {dtype}, expected {expected_dtype}")
    return math.prod(tensor_slice.get_shape()) * bytes_per_element  # type: ignore[attr-defined]


@register_model_loader(LOAD_FORMAT)
class GlmW2FilteredSafetensorsLoader(DefaultModelLoader):
    """Default file discovery plus pre-read filtering of peer W2 experts."""

    def __init__(self, load_config: LoadConfig):
        if load_config.load_format != LOAD_FORMAT:
            raise ValueError(f"expected load format {LOAD_FORMAT}")
        if load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError(f"{LOAD_FORMAT} supports only lazy safetensors loading")
        if load_config.model_loader_extra_config.get("enable_multithread_load"):
            raise ValueError(f"{LOAD_FORMAT} does not support multithread loading")
        # Reuse DefaultModelLoader's exact safetensors file/index selection.
        super().__init__(dataclasses.replace(load_config, load_format="safetensors"))
        self._local_range: tuple[int, int] | None = None
        self._dense_layers = 0
        self._decoder_layers = 0
        self._num_experts = 0
        self._model_path: str | None = None
        self._revision: str | None = None
        self._draft_layer_prefixes: tuple[str, ...] = ()
        self.skipped_tensors = 0
        self.skipped_bytes = 0
        self.matched_expert_tensors = 0
        self.skipped_superseded_tensors = 0
        self.skipped_superseded_bytes = 0

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        lo, hi, dense_layers, layers = local_decoder_expert_range(model_config)
        self._local_range = (lo, hi)
        self._dense_layers = dense_layers
        self._decoder_layers = layers
        self._num_experts = model_config.hf_text_config.n_routed_experts
        self._model_path = model_config.model
        self._revision = model_config.revision
        self._draft_layer_prefixes = (
            tuple(f"model.language_model.layers.{index}." for index in range(dense_layers, layers))
            if model_config.architectures == [PACKED_GLM_MTP_ARCHITECTURE]
            else ()
        )
        self.skipped_tensors = 0
        self.skipped_bytes = 0
        self.matched_expert_tensors = 0
        self.skipped_superseded_tensors = 0
        self.skipped_superseded_bytes = 0
        super().load_weights(model, model_config)
        if not self.matched_expert_tensors:
            raise ValueError(f"{LOAD_FORMAT} found no GLM W2 decoder expert tensors in checkpoint")
        logger.info(
            "GLM W2 filtered load: experts [%d,%d), skipped %d peer tensors / %.2f GiB "
            "and %d superseded local tensors / %.2f GiB",
            lo,
            hi,
            self.skipped_tensors,
            self.skipped_bytes / 1024**3,
            self.skipped_superseded_tensors,
            self.skipped_superseded_bytes / 1024**3,
        )

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        if (
            self._local_range is None
            or source.model_or_path != self._model_path
            or source.revision != self._revision
            or source.subfolder is not None
            or source.prefix
        ):
            raise ValueError(f"{LOAD_FORMAT} received an unexpected model/checkpoint source")
        folder, files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if not use_safetensors:
            raise ValueError(f"{LOAD_FORMAT} selected non-safetensors weights")
        index_path = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
        indexed_shards = None
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as index_file:
                indexed_shards = json.load(index_file)["weight_map"]
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        for shard in tqdm(
            sorted(files, key=_natural_sort_key),
            desc="Loading safetensors checkpoint shards",
            disable=not enable_tqdm(self.load_config.use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        ):
            with safe_open(shard, framework="pt") as handle:
                for name in handle.keys():  # noqa: SIM118 - safe_open is not a dict
                    # The draft shares target embeddings/head. Read only its
                    # own layer tensors instead of materializing the backbone.
                    if self._draft_layer_prefixes and not name.startswith(self._draft_layer_prefixes):
                        continue
                    skip, kind = should_skip_glm_expert(
                        name,
                        local_range=self._local_range,
                        dense_layers=self._dense_layers,
                        decoder_layers=self._decoder_layers,
                        num_experts=self._num_experts,
                    )
                    if kind is not None:
                        self.matched_expert_tensors += 1
                    if skip:
                        assert kind is not None
                        self.skipped_bytes += skipped_tensor_bytes(handle.get_slice(name), kind)
                        self.skipped_tensors += 1
                        continue
                    if kind is not None and indexed_shards is not None:
                        indexed_shard = indexed_shards.get(name)
                        if indexed_shard is not None and indexed_shard != os.path.basename(shard):
                            self.skipped_superseded_bytes += skipped_tensor_bytes(handle.get_slice(name), kind)
                            self.skipped_superseded_tensors += 1
                            continue
                    yield source.prefix + name, handle.get_tensor(name)
