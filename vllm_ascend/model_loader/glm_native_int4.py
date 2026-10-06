# SPDX-License-Identifier: Apache-2.0
"""Direct safetensors loading of a committed GLM native INT4 checkpoint."""

import dataclasses
import json
import time
from pathlib import Path

import regex as re
from safetensors import safe_open
from vllm.config.load import LoadConfig
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from tools.glm_perf.fused_moe_profile import frozen_helper
from tools.glm_perf.indexer_bundle import install as install_indexer_bundle
from tools.glm_perf.native_checkpoint import INDEX, LAYOUT, MANIFEST, NativeInt4MoEMethod, selected_weights

LOAD_FORMAT = "glm_native_int4"
LAYER_PATH = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@register_model_loader(LOAD_FORMAT)
class GlmNativeInt4Loader(DefaultModelLoader):
    """Copy native bytes to existing grouped banks; no packing or rollback copy."""

    def __init__(self, load_config: LoadConfig):
        if load_config.load_format != LOAD_FORMAT:
            raise ValueError(f"expected {LOAD_FORMAT}")
        if load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError("native GLM checkpoint requires lazy safetensors loading")
        if load_config.model_loader_extra_config.get("enable_multithread_load"):
            raise ValueError("native GLM loader owns its authoritative shard selection")
        super().__init__(dataclasses.replace(load_config, load_format="safetensors"))
        self.folder = None
        self.native_manifest = None
        self.local_range = None
        self.layer_keys = ()
        self.draft = False
        self.native_code_tensors = 0

    def load_weights(self, model, model_config):
        self.folder = Path(model_config.model).resolve()
        manifest = json.loads((self.folder / MANIFEST).read_text())
        if (
            manifest.get("schema_version") != 1
            or manifest.get("complete") is not True
            or manifest.get("layout") != LAYOUT
        ):
            raise ValueError("native GLM checkpoint is incomplete or has an unsupported layout")
        if getattr(model_config.hf_config, "ascend_glm_expert_layout", None) != LAYOUT:
            raise ValueError("native GLM checkpoint config lacks the matching layout marker")
        named_owners = [
            (name, module) for name, module in model.named_modules() if getattr(module, "w2_experts", None) is not None
        ]
        owners = [module for _, module in named_owners]
        if not owners or any(not hasattr(module, "routed_experts_forward") for module in owners):
            raise ValueError("native GLM loader requires composed routed MoE modules")
        banks = [module.w2_experts for module in owners]
        ranges = {(bank.local_expert_offset, bank.local_expert_offset + bank.num_local_experts) for bank in banks}
        if len(ranges) != 1 or any(bank.offload_to_cpu for bank in banks):
            raise ValueError("native GLM loader requires one resident expert partition")
        self.local_range = ranges.pop()
        if self.local_range[1] - self.local_range[0] != manifest["num_experts"] // manifest["world_size"]:
            raise ValueError("native checkpoint and runtime expert partition differ")
        matches = [LAYER_PATH.search(name) for name, _ in named_owners]
        if any(match is None for match in matches):
            raise ValueError("native GLM banks lack layer identities")
        self.layer_keys = tuple("layers." + match[1] for match in matches)
        self.draft = model.__class__.__name__ == "Glm5NextW2MTP"
        self.native_manifest = manifest
        # place_resident_tensor otherwise converts canonical bytes to NZ. The
        # checkpoint already contains Cube bytes and must pass through unchanged.
        for bank in banks:
            bank.nz_packed_codes = False
        logger.info("GLM native INT4 reading layers %s, local experts %s", self.layer_keys, self.local_range)
        super().load_weights(model, model_config)
        expected = len(banks) * (self.local_range[1] - self.local_range[0]) * 3
        if self.native_code_tensors != expected:
            raise ValueError("native GLM loader did not read every local code tensor")
        helper, options = frozen_helper(self.folder / manifest["kernel_bundle"])
        native = helper.NativeFusedMoE(
            self.folder / manifest["kernel_bundle"],
            namespace=options["namespace"],
            activation_bits=manifest["activation_bits"],
            prepared_weight_layout=True,
        )
        for module, bank in zip(owners, banks):
            bank.native_weight_layout = LAYOUT
            module._method = NativeInt4MoEMethod(native)
        indexers = 0
        if "indexer_kernel_bundle" in manifest:
            indexers = install_indexer_bundle(model, self.folder / manifest["indexer_kernel_bundle"])
        model._native_int4_load_report = {
            "layout": LAYOUT,
            "activation_bits": manifest["activation_bits"],
            "banks": 2 * len(banks),
            "native_code_tensors": self.native_code_tensors,
            "transformed_code_tensors": 0,
            "layout_backup_bytes": 0,
            "aicore_bf16_indexers": indexers,
        }
        logger.info("GLM native INT4 loaded directly: %s", model._native_int4_load_report)

    def _get_weights_iterator(self, source):
        if (
            self.folder is None
            or Path(source.model_or_path).resolve() != self.folder
            or source.prefix
            or source.subfolder
        ):
            raise ValueError("native GLM loader requires its local checkpoint source")
        index = json.loads((self.folder / INDEX).read_text())["weight_map"]
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        selected = selected_weights(index, self.local_range, self.layer_keys, self.draft)
        # Follow the index for every tensor, including MTP and dense weights.
        # Hard-linked source shards also contain superseded canonical codes.
        by_shard = {}
        for name in selected:
            by_shard.setdefault(index[name], []).append(name)
        native_files = {record["file"] for record in self.native_manifest["native_shards"]}
        for shard, names in sorted(by_shard.items()):
            with safe_open(str(self.folder / shard), framework="pt", device="cpu") as handle:
                if shard in native_files and (handle.metadata() or {}).get("layout") != LAYOUT:
                    raise ValueError("native expert shard lacks the matching layout metadata")
                for name in sorted(names):
                    if name.endswith("_proj_codes"):
                        if shard not in native_files:
                            raise ValueError("canonical expert bytes appear in a native checkpoint index")
                        self.native_code_tensors += 1
                    tensor = handle.get_tensor(name)
                    if name.endswith("_proj_codes"):
                        # Ascend's long-term page pinning can repeatedly split
                        # huge file-backed pages when copying a safetensors view.
                        # Stage only this tensor in ordinary CPU storage. This
                        # is a byte copy, never a decode or a layout transform.
                        tensor = tensor.clone()
                    yield name, tensor
