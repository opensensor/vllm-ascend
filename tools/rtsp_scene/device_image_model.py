"""Opt-in registered model/processor inheritance for worker image preparation.

The general plugin registers an additional architecture only. Stock Qwen models
retain their original processor and worker behavior.
"""

from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig

from tools.rtsp_scene.raw_image_transport import with_raw_images


class SceneProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_processor(self, **kwargs):
        return with_raw_images(super().get_hf_processor(**kwargs))


class SceneMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        if "scene_rgb" in hf_inputs:
            geometry = hf_inputs["scene_geometry"]
            fields.update(
                scene_rgb=MultiModalFieldConfig.flat_from_sizes("image", geometry.prod(-1) * 3),
                scene_geometry=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
                image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            )
        return fields


@MULTIMODAL_REGISTRY.register_processor(
    SceneMultiModalProcessor, info=SceneProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class SceneDeviceQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def embed_multimodal(self, **kwargs):
        rgb = kwargs.pop("scene_rgb", None)
        geometry = kwargs.pop("scene_geometry", None)
        if rgb is not None or geometry is not None:
            if rgb is None or geometry is None or "pixel_values" in kwargs or "image_embeds" in kwargs:
                raise ValueError("Incomplete or ambiguous scene byte inputs")
            if not hasattr(self, "scene_image_preparer"):
                raise RuntimeError("Scene byte inputs require the gated scene worker")
            kwargs["pixel_values"] = self.scene_image_preparer(rgb, geometry, kwargs["image_grid_thw"])
        return super().embed_multimodal(**kwargs)
