"""Register the isolated scene architecture through vLLM's plugin interface."""


def register():
    # General plugins run before model resolution. Defer model/processor imports
    # until that resolution to avoid importing model code during platform setup.
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "SceneDeviceQwen3VLForConditionalGeneration",
        "tools.rtsp_scene.device_image_model:SceneDeviceQwen3VLForConditionalGeneration",
    )
