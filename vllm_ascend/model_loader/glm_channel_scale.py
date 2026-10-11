# SPDX-License-Identifier: Apache-2.0
"""Explicit cold-load candidate: original-FP8 channel W4 banks plus base GLM."""

import dataclasses
import json
import time
from pathlib import Path

from safetensors import safe_open
from vllm.distributed import get_pp_group
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from tools.glm_perf.channel_scale_checkpoint import file_sha256
from tools.glm_perf.channel_scale_stage import prepare_stage
from tools.glm_perf.indexer_bundle import install as install_indexer_bundle
from tools.glm_perf.indexer_owner import OWNER_OPTIONS, plan_install, read_options
from tools.glm_perf.indexer_owner import install as install_indexer_owner
from tools.glm_perf.native_checkpoint import INDEX, MANIFEST, selected_weights

LOAD_FORMAT = "glm_channel_scale_candidate"
REQUIRED_OPTIONS = ("channel_overlay", "channel_bundle", "channel_host_budget_bytes", "channel_device_budget_bytes")
OPTIONAL_OPTIONS = ("channel_decode_bits", "channel_prefill_bits", *OWNER_OPTIONS)


@register_model_loader(LOAD_FORMAT)
class GlmChannelScaleLoader(DefaultModelLoader):
    """All local experts use the candidate pipeline; no legacy bank is loaded.

    This is deliberately a named candidate format. Device reservation excludes
    dense parameters, KV, graph/HCCL resources and pipeline packet storage: the
    caller must reserve those separately in an offline memory plan.
    """

    def __init__(self, load_config):
        if load_config.load_format != LOAD_FORMAT:
            raise ValueError("requires explicit channel-scale candidate load format")
        extra = dict(load_config.model_loader_extra_config)
        if any(key not in extra for key in REQUIRED_OPTIONS) or set(extra) - set(REQUIRED_OPTIONS + OPTIONAL_OPTIONS):
            raise ValueError("channel-scale loader requires explicit overlay, bundle and CPU/device reservations")
        self.indexer_owner_options = read_options(extra)
        self.channel_options = extra
        self.folder = None
        super().__init__(dataclasses.replace(load_config, load_format="safetensors", model_loader_extra_config={}))

    def load_weights(self, model, model_config):
        self.folder = Path(model_config.model).resolve(strict=True)
        if not self.folder.is_dir() or not (self.folder / INDEX).is_file():
            raise ValueError("channel-scale candidate requires a local indexed base GLM checkpoint")
        parallel = model.vllm_config.parallel_config
        if model.vllm_config.speculative_config is not None:
            raise ValueError("new numerical contract requires target-only qualification before MTP")
        scheduler = model.vllm_config.scheduler_config
        if not hasattr(model, "_glm_text_config"):
            raise ValueError("candidate requires the existing composed GLM text adapter")
        if self.indexer_owner_options is not None:
            plan_install(model, **self.indexer_owner_options)
        indexer_root = None
        base_manifest_path = self.folder / MANIFEST
        if base_manifest_path.exists():
            base_manifest = json.loads(base_manifest_path.read_text())
            if "indexer_kernel_bundle" in base_manifest:
                relative = Path(base_manifest["indexer_kernel_bundle"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe permanent indexer bundle path")
                indexer_root = self.folder / relative
                if file_sha256(indexer_root / "manifest.json") != base_manifest.get("indexer_manifest_sha256"):
                    raise ValueError("base GLM indexer bundle differs from its checksum")
        top_k = model._glm_text_config.num_experts_per_token
        options = self.channel_options
        report = prepare_stage(
            model,
            options["channel_overlay"],
            options["channel_bundle"],
            max_tokens=scheduler.max_num_batched_tokens,
            top_k=top_k,
            host_budget_bytes=options["channel_host_budget_bytes"],
            device_budget_bytes=options["channel_device_budget_bytes"],
            decode_bits=options.get("channel_decode_bits", 8),
            prefill_bits=options.get("channel_prefill_bits", 8),
        )
        self.layer_keys = tuple(f"layers.{n}" for n in report["layers"])
        self.pipeline_partition = None
        if parallel.pipeline_parallel_size > 1:
            pp = get_pp_group()
            self.pipeline_partition = (
                model.model.start_layer,
                model.model.end_layer,
                pp.is_first_rank,
                pp.is_last_rank,
            )
        super().load_weights(model, model_config)
        if indexer_root is not None:
            report["aicore_bf16_indexers"] = install_indexer_bundle(model, indexer_root)
        if self.indexer_owner_options is not None:
            report["indexer_owner"] = install_indexer_owner(model, **self.indexer_owner_options)
        logger.info("GLM channel-scale OFFLINE CANDIDATE loaded: %s", report)

    def _get_weights_iterator(self, source):
        if (
            self.folder is None
            or Path(source.model_or_path).resolve() != self.folder
            or source.prefix
            or source.subfolder
        ):
            raise ValueError("channel-scale candidate requires its local base checkpoint source")
        index = json.loads((self.folder / INDEX).read_text())["weight_map"]
        self.counter_before_loading_weights = time.perf_counter()
        names = selected_weights(index, (0, 288), self.layer_keys, False, self.pipeline_partition)
        shards = {}
        for name in names:
            # Ignore all derived packed experts and original FP8 experts before
            # materializing tensors. The candidate overlay is authoritative.
            if ".mlp.experts." in name:
                continue
            filename = index[name]
            if Path(filename).name != filename:
                raise ValueError("unsafe base checkpoint shard path")
            shards.setdefault(filename, []).append(name)
        for filename, names in sorted(shards.items()):
            with safe_open(str(self.folder / filename), framework="pt", device="cpu") as handle:
                for name in sorted(names):
                    yield name, handle.get_tensor(name).clone()
