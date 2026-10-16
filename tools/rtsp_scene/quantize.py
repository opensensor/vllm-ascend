"""Create a separate, uncalibrated W8A8_DYNAMIC text checkpoint for evaluation."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

TEXT_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def selected_weight(name: str, quantize_head: bool = False) -> bool:
    parts = name.split(".")
    return (quantize_head and name == "lm_head.weight") or (
        name.startswith("model.language_model.layers.")
        and len(parts) == 7
        and parts[3].isdigit()
        and parts[-2] in TEXT_PROJECTIONS
        and parts[-1] == "weight"
    )


def symmetric_int8(weight):
    """Per-output-channel weights; preserve zero rows without zero scales."""
    # Torch is optional for host configuration checks; imported by this worker.
    import torch

    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("Quantization requires a finite weight matrix")
    scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    quantized = (weight.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale, torch.zeros_like(scale)


def runtime_description(description: dict[str, str]) -> dict[str, str]:
    """Include construction-time vLLM aliases without renaming checkpoint tensors."""
    mapped = dict(description)
    prefixes = {
        "model.visual.": "visual.",
        "model.language_model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    }
    for name, value in description.items():
        for old, new in prefixes.items():
            if name.startswith(old):
                mapped[new + name.removeprefix(old)] = value
                break
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantize-head", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        parser.error("Use a new output directory; source weights are never modified")
    # Optional checkpoint dependencies are confined to the conversion worker.
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(4)
    os.umask(0o077)
    args.output.mkdir(parents=True)
    index_path = args.source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    source_config = args.source / "config.json"
    config = json.loads(source_config.read_text())
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("This experiment supports dense Qwen3-VL checkpoints only")
    config["quantization_config"] = {"quant_method": "ascend", "model_quant_type": "W8A8_DYNAMIC"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for path in args.source.iterdir():
        if (
            path.is_file()
            and path.name != "config.json"
            and not path.name.endswith(".safetensors")
            and not path.name.startswith("model.safetensors")
        ):
            shutil.copyfile(path, args.output / path.name)
    description, weight_map, total_size = {}, {}, 0
    converted = []
    started = time.monotonic()
    for shard_name in sorted(set(index.values())):
        tensors = {}
        with safe_open(args.source / shard_name, framework="pt", device="cpu") as shard:
            for name in list(shard.keys()):
                value = shard.get_tensor(name)
                if selected_weight(name, args.quantize_head):
                    weight, scale, offset = symmetric_int8(value)
                    tensors[name] = weight
                    prefix = name.removesuffix(".weight")
                    tensors[prefix + ".weight_scale"] = scale
                    tensors[prefix + ".weight_offset"] = offset
                    description[name] = "W8A8_DYNAMIC"
                    converted.append(name)
                else:
                    tensors[name] = value
                    description[name] = "FLOAT"
        for name, value in tensors.items():
            weight_map[name] = shard_name
            total_size += value.numel() * value.element_size()
        save_file(tensors, args.output / shard_name, metadata={"format": "pt"})
        print(json.dumps({"shard": shard_name, "converted_weights": len(converted)}), flush=True)
        del tensors
    expected = sum(selected_weight(name, args.quantize_head) for name in index)
    if not converted or len(converted) != expected:
        raise ValueError("Incomplete text projection conversion")
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n"
    )
    (args.output / "quant_model_description.json").write_text(
        json.dumps(runtime_description(description), indent=2) + "\n"
    )
    # The pinned runtime treats any HF quantization_config as the complete
    # description and skips the sidecar. Include parameters rather than only
    # format metadata; this also works with newer ModelSlim config readers.
    config["quantization_config"] = {"quant_method": "ascend", **runtime_description(description)}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "source": str(args.source.resolve()),
        "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "method": "uncalibrated symmetric per-channel INT8 weights, runtime per-token INT8 activations",
        "converted_weights": converted,
        "quantize_head": args.quantize_head,
        "stored_tensor_bytes": total_size,
        "elapsed_seconds": time.monotonic() - started,
        "evaluation_required": True,
    }
    (args.output / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
