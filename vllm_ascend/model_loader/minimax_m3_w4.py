# SPDX-License-Identifier: Apache-2.0
"""Load offline M3 W4 exports without constructing dense routed experts."""

import dataclasses
from pathlib import Path

from vllm.distributed import get_pp_group
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from tools.minimax_m3.checkpoint import dense_weights, load_expert_banks, read_manifest


@register_model_loader("minimax_m3_w4")
class MiniMaxM3W4Loader(DefaultModelLoader):
    def __init__(self, load_config):
        options = load_config.model_loader_extra_config or {}
        if set(options) - {"expert_layout", "max_grouped_routes"}:
            raise ValueError("Unknown M3 W4 loader options")
        self.expert_layout = options.get("expert_layout", "canonical")
        self.max_grouped_routes = options.get("max_grouped_routes", 128)
        if self.expert_layout not in ("canonical", "nz256"):
            raise ValueError("M3 expert_layout must be canonical or nz256")
        if type(self.max_grouped_routes) is not int or self.max_grouped_routes not in (128, 512):
            raise ValueError("M3 max_grouped_routes must be 128 or 512")
        super().__init__(dataclasses.replace(load_config, load_format="safetensors", model_loader_extra_config={}))

    def download_model(self, model_config):
        # Intentionally local-only; never silently download the BF16 checkpoint.
        read_manifest(Path(model_config.model))

    def load_weights(self, model, model_config):
        folder = Path(model_config.model)
        manifest = read_manifest(folder)
        backbones = [module for module in model.modules() if type(module).__name__ == "MiniMaxM3Model"]
        if len(backbones) != 1:
            raise ValueError("M3 W4 loader requires exactly one MiniMaxM3Model backbone")
        backbone = backbones[0]
        load_expert_banks(
            model, folder, manifest, expert_layout=self.expert_layout, max_grouped_routes=self.max_grouped_routes
        )
        loaded = model.load_weights(
            dense_weights(
                folder,
                manifest,
                start_layer=backbone.start_layer,
                end_layer=backbone.end_layer,
                first_stage=get_pp_group().is_first_rank,
                last_stage=get_pp_group().is_last_rank,
            )
        )
        if loaded is None:
            raise ValueError("M3 model did not report loaded parameter names")
        # There are no online-quantization parameters in this format. Do not
        # use DefaultModelLoader's exemption for modules with postprocessing:
        # it can also exempt ordinary dense linears from completeness checks.
        missing = {name for name, _ in model.named_parameters()} - loaded
        if missing:
            raise ValueError(f"M3 parameters not initialized from checkpoint: {sorted(missing)}")
