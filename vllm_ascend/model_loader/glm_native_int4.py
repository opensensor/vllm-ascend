# SPDX-License-Identifier: Apache-2.0
"""Direct safetensors loading of a committed GLM native INT4 checkpoint."""

import dataclasses
import json
import time
from pathlib import Path

import regex as re
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from vllm.config.load import LoadConfig
from vllm.distributed import get_pp_group
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from tools.glm_perf.deployment_profiles import validate_profile
from tools.glm_perf.fused_moe_profile import frozen_helper
from tools.glm_perf.fused_weight_layout import FP16_SCALE_LAYOUT
from tools.glm_perf.indexer_bundle import install as install_indexer_bundle
from tools.glm_perf.indexer_heads import HEAD_OPTIONS
from tools.glm_perf.indexer_heads import install as install_indexer_heads
from tools.glm_perf.indexer_heads import plan_install as plan_indexer_heads
from tools.glm_perf.indexer_heads import read_options as read_head_options
from tools.glm_perf.indexer_owner import OWNER_OPTIONS, plan_install, read_options
from tools.glm_perf.indexer_owner import install as install_indexer_owner
from tools.glm_perf.native_checkpoint import (
    INDEX,
    LAYOUT,
    MANIFEST,
    NativeInt4MoEMethod,
    SplitNativeMoE,
    file_digest,
    selected_weights,
    validate_scale_contract,
)
from tools.glm_perf.router_owner import ROUTER_OPTIONS
from tools.glm_perf.router_owner import install as install_router_owner
from tools.glm_perf.router_owner import plan_install as plan_router_owner
from tools.glm_perf.router_owner import read_options as read_router_options

LOAD_FORMAT = "glm_native_int4"
LAYER_PATH = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def resolve_native_folder(model_path, revision=None, download_dir=None):
    """Fetch the entire versioned native bundle, not only safetensors weights."""
    path = Path(model_path)
    if path.is_dir():
        return path.resolve()
    if path.is_absolute():
        raise FileNotFoundError(path)
    return Path(
        snapshot_download(
            repo_id=model_path,
            revision=revision,
            cache_dir=download_dir,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.txt",
                "*.jinja",
                "*.jinja2",
                "README.md",
                "SOURCE_README.md",
                "LICENSE*",
                "NOTICE*",
                "native-kernels/*",
                "native-kernels/**/*",
                "native-decode-kernels/*",
                "native-decode-kernels/**/*",
                "indexer-kernels/*",
                "indexer-kernels/**/*",
            ],
        )
    ).resolve()


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
        self.indexer_owner_options = read_options(load_config.model_loader_extra_config)
        self.router_owner_options = read_router_options(load_config.model_loader_extra_config)
        self.indexer_head_options = read_head_options(load_config.model_loader_extra_config)
        base_options = {
            key: value
            for key, value in load_config.model_loader_extra_config.items()
            if key not in (*OWNER_OPTIONS, *ROUTER_OPTIONS, *HEAD_OPTIONS)
        }
        super().__init__(
            dataclasses.replace(load_config, load_format="safetensors", model_loader_extra_config=base_options)
        )
        self.folder = None
        self.native_manifest = None
        self.local_range = None
        self.layer_keys = ()
        self.draft = False
        self.pipeline_partition = None
        self.native_code_tensors = 0

    def load_weights(self, model, model_config):
        if self.indexer_owner_options is not None:
            plan_install(model, **self.indexer_owner_options)
        if self.router_owner_options is not None:
            plan_router_owner(model, **self.router_owner_options)
        if self.indexer_head_options is not None:
            plan_indexer_heads(model, **self.indexer_head_options)
        self.folder = resolve_native_folder(model_config.model, model_config.revision, self.load_config.download_dir)
        self.resolved_model_source = model_config.model
        manifest = json.loads((self.folder / MANIFEST).read_text())
        if (
            manifest.get("schema_version") != 1
            or manifest.get("complete") is not True
            or manifest.get("layout") != LAYOUT
        ):
            raise ValueError("native GLM checkpoint is incomplete or has an unsupported layout")
        if getattr(model_config.hf_config, "ascend_glm_expert_layout", None) != LAYOUT:
            raise ValueError("native GLM checkpoint config lacks the matching layout marker")
        profile_name = getattr(model_config.hf_config, "ascend_glm_deployment_profile", None)
        if profile_name != manifest.get("deployment_profile"):
            raise ValueError("checkpoint deployment profile markers disagree")
        if profile_name is not None:
            parallel = model.vllm_config.parallel_config
            spec = model.vllm_config.speculative_config
            validate_profile(
                profile_name,
                tensor_parallel_size=parallel.tensor_parallel_size,
                pipeline_parallel_size=parallel.pipeline_parallel_size,
                mtp_tokens=spec.num_speculative_tokens if spec else 0,
                layer_bits=manifest["layer_bits"],
            )
        provenance_path = self.folder / manifest["kernel_bundle"] / "provenance.json"
        if (
            "kernel_provenance_sha256" in manifest
            and file_digest(provenance_path) != manifest["kernel_provenance_sha256"]
        ):
            raise ValueError("prefill kernel provenance differs from manifest")
        options = json.loads(provenance_path.read_text())["_build"]
        scale_marker = validate_scale_contract(
            manifest, getattr(model_config.hf_config, "ascend_glm_expert_scale_layout", None), options
        )
        decode_root = None
        if "decode_kernel_bundle" in manifest:
            decode_root = self.folder / manifest["decode_kernel_bundle"]
            decode_provenance = decode_root / "provenance.json"
            if file_digest(decode_provenance) != manifest["decode_kernel_provenance_sha256"]:
                raise ValueError("decode kernel provenance differs from manifest")
            decode_options = json.loads(decode_provenance.read_text())["_build"]
            validate_scale_contract(manifest, scale_marker, decode_options)
            if any(decode_options.get(key) == options.get(key) for key in ("namespace", "helper_package")):
                raise ValueError("decode bundle requires a distinct operator namespace and helper package")
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
        if model.vllm_config.parallel_config.pipeline_parallel_size > 1:
            backbone = model.model
            pp = get_pp_group()
            self.pipeline_partition = (backbone.start_layer, backbone.end_layer, pp.is_first_rank, pp.is_last_rank)
        self.native_manifest = manifest
        # place_resident_tensor otherwise converts canonical bytes to NZ. The
        # checkpoint already contains Cube bytes and must pass through unchanged.
        for bank in banks:
            bank.nz_packed_codes = False
            bank.native_weight_scale_layout = scale_marker
        logger.info("GLM native INT4 reading layers %s, local experts %s", self.layer_keys, self.local_range)
        super().load_weights(model, model_config)
        expected = len(banks) * (self.local_range[1] - self.local_range[0]) * 3
        if self.native_code_tensors != expected:
            raise ValueError("native GLM loader did not read every local code tensor")
        helper, options = frozen_helper(self.folder / manifest["kernel_bundle"])
        route_options = {"fp16_route_workspace": True} if options.get("fp16_route_workspace") else {}
        native = helper.NativeFusedMoE(
            self.folder / manifest["kernel_bundle"],
            namespace=options["namespace"],
            activation_bits=manifest["activation_bits"],
            prepared_weight_layout=True,
            shared_prefill_max_tokens=model.vllm_config.scheduler_config.max_num_batched_tokens,
            **route_options,
        )
        if decode_root is not None:
            decode_helper, decode_options = frozen_helper(decode_root)
            decode = decode_helper.NativeFusedMoE(
                decode_root,
                namespace=decode_options["namespace"],
                activation_bits=manifest["activation_bits"],
                prepared_weight_layout=True,
                **({"fp16_route_workspace": True} if decode_options.get("fp16_route_workspace") else {}),
            )
            native = SplitNativeMoE(native, decode)
        for module, bank in zip(owners, banks):
            bank.native_weight_layout = LAYOUT
            module._method = NativeInt4MoEMethod(native)
        indexers = 0
        if "indexer_kernel_bundle" in manifest:
            indexer_root = self.folder / manifest["indexer_kernel_bundle"]
            if (
                "indexer_manifest_sha256" in manifest
                and file_digest(indexer_root / "manifest.json") != manifest["indexer_manifest_sha256"]
            ):
                raise ValueError("indexer manifest differs from checkpoint")
            indexers = install_indexer_bundle(model, indexer_root)
        owner_report = None
        if self.indexer_owner_options is not None:
            owner_report = install_indexer_owner(model, **self.indexer_owner_options)
        router_report = None
        if self.router_owner_options is not None:
            router_report = install_router_owner(model, **self.router_owner_options)
        heads_report = None
        if self.indexer_head_options is not None:
            heads_report = install_indexer_heads(model, **self.indexer_head_options)
        model._native_int4_load_report = {
            "indexer_heads": heads_report,
            "router_owner": router_report,
            "layout": LAYOUT,
            "activation_bits": manifest["activation_bits"],
            "banks": 2 * len(banks),
            "native_code_tensors": self.native_code_tensors,
            "transformed_code_tensors": 0,
            "layout_backup_bytes": 0,
            "weight_scale_layout": scale_marker,
            "aicore_bf16_indexers": indexers,
            "indexer_owner": owner_report,
        }
        logger.info("GLM native INT4 loaded directly: %s", model._native_int4_load_report)

    def _get_weights_iterator(self, source):
        if (
            self.folder is None
            or (
                source.model_or_path != getattr(self, "resolved_model_source", str(self.folder))
                and Path(source.model_or_path).resolve() != self.folder
            )
            or source.prefix
            or source.subfolder
        ):
            raise ValueError("native GLM loader requires its local checkpoint source")
        index = json.loads((self.folder / INDEX).read_text())["weight_map"]
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        selected = selected_weights(
            index, self.local_range, self.layer_keys, self.draft, getattr(self, "pipeline_partition", None)
        )
        # Follow the index for every tensor, including MTP and dense weights.
        # Hard-linked source shards also contain superseded canonical codes.
        by_shard = {}
        for name in selected:
            by_shard.setdefault(index[name], []).append(name)
        native_files = {record["file"] for record in self.native_manifest["native_shards"]}
        scale_records = {row["file"]: row for row in self.native_manifest.get("scale_shards", [])}
        storage_records = {row["file"]: row for row in self.native_manifest.get("storage_promotion_shards", [])}
        marker = self.native_manifest.get("weight_scale_layout")
        for shard, names in sorted(by_shard.items()):
            if shard in scale_records:
                actual_digest = file_digest(self.folder / shard)
                if actual_digest != scale_records[shard]["sha256"]:
                    raise ValueError(
                        f"permanent weight scale shard differs from its checksum: {shard}, actual={actual_digest}"
                    )
            if shard in storage_records and file_digest(self.folder / shard) != storage_records[shard]["sha256"]:
                raise ValueError("permanent W4 storage shard differs from its checksum")
            with safe_open(str(self.folder / shard), framework="pt", device="cpu") as handle:
                if shard in native_files and (handle.metadata() or {}).get("layout") != LAYOUT:
                    raise ValueError("native expert shard lacks the matching layout metadata")
                for name in sorted(names):
                    if name.endswith("_proj_codes"):
                        if shard not in native_files:
                            raise ValueError("canonical expert bytes appear in a native checkpoint index")
                        self.native_code_tensors += 1
                    if marker is not None and name.endswith("_proj_scale") and ".mlp.experts." in name:
                        if shard not in scale_records or (handle.metadata() or {}).get("weight_scale_layout") != marker:
                            raise ValueError("unqualified weight scales in prerounded checkpoint index")
                    tensor = handle.get_tensor(name)
                    if name.endswith("_proj_scale") and ".mlp.experts." in name:
                        expected_dtype = torch.float16 if marker == FP16_SCALE_LAYOUT else torch.float32
                        if tensor.dtype != expected_dtype:
                            raise ValueError("expert scale tensor dtype differs from permanent storage marker")
                    # Ascend's long-term page pinning must not target mmap-backed
                    # checkpoint storage. Dense parameters also reach CPU-to-NPU
                    # copies through the upstream loader; stage them as well as
                    # expert codes/scales. This preserves every byte and dtype.
                    tensor = tensor.clone()
                    yield name, tensor
