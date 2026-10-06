# SPDX-License-Identifier: Apache-2.0
"""Persist verified native expert bytes; publish only a complete checkpoint."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .fused_weight_layout import pack_cube, tensor_digest
from .glm_int4 import unpack_canonical_codes

LAYOUT = "cube_n128_k256_v1"
MANIFEST = "native-layout.json"
INDEX = "model.safetensors.index.json"
CODE_NAME = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj_codes$")
EXPERT_NAME = re.compile(
    r"^model\.language_model\.(layers\.\d+)\.mlp\.experts\.(\d+)\."
    r"(?:gate|up|down)_proj_(?:codes|scale)$"
)
SHA_CHUNK_BYTES = 8 * 1024**2


def selected_weights(index, local_range, layer_keys, draft=False):
    """Authoritative local tensors, excluding peer and superseded source codes."""
    first, stop = local_range
    prefixes = tuple("model.language_model." + key + "." for key in layer_keys)
    for name in index:
        match = EXPERT_NAME.fullmatch(name)
        if match is not None:
            if match[1] not in layer_keys or not first <= int(match[2]) < stop:
                continue
        elif draft and not name.startswith(prefixes):
            continue
        yield name


class NativeInt4MoEMethod:
    """Per-MoE composition with an explicit hot-swap seam and no legacy fallback."""

    def __init__(self, native):
        self.native = native

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        return self._apply_device_grouped(
            None, layer.w2_experts, x, topk_weights, topk_ids, getattr(layer, "w2_shared_expert", None)
        )

    def _apply_device_grouped(self, grouped_op, experts, x, topk_weights, topk_ids, shared_expert):
        if getattr(experts, "native_weight_layout", None) != LAYOUT:
            raise ValueError("native INT4 method requires the permanent Cube layout")
        result = self.native(
            x.to(torch.float16).contiguous(),
            experts.gate_up_packed_bank,
            experts.gate_up_scale_bank,
            experts.down_packed_bank,
            experts.down_scale_bank,
            topk_weights.float().contiguous(),
            topk_ids.contiguous(),
            experts.local_expert_offset,
        )
        if shared_expert is not None:
            result += shared_expert.forward(x).float()
        return result


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(SHA_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_geometry(source):
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    index = json.loads((source / INDEX).read_text())
    names = [name for name in index["weight_map"] if CODE_NAME.fullmatch(name)]
    layers = sorted({int(CODE_NAME.fullmatch(name)[1]) for name in names})
    experts = text["n_routed_experts"]
    expected = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj_codes"
        for layer in layers
        for expert in range(experts)
        for projection in ("gate", "up", "down")
    }
    if not layers or set(names) != expected:
        raise ValueError("source checkpoint has incomplete routed code coverage")
    return config, index, layers, experts


def initialize(source, output, bundle, world_size=4, activation_bits=4):
    """Share unchanged shards by hard link, keeping an independent index/config."""
    source, output, bundle = (Path(path).resolve() for path in (source, output, bundle))
    if output.exists():
        raise FileExistsError(output)
    config, index, layers, experts = source_geometry(source)
    if world_size < 1 or experts % world_size or activation_bits not in (4, 8):
        raise ValueError("native export requires even expert partition and A4/A8")
    if source.stat().st_dev != output.parent.stat().st_dev:
        raise ValueError("use the source filesystem for hard-linked unchanged shards")
    provenance = json.loads((bundle / "provenance.json").read_text())
    options = provenance["_build"]
    if not options.get("fused_moe") or not options.get("prepared_weight_layout"):
        raise ValueError("kernel bundle does not consume the permanent native layout")
    output.mkdir()
    for shard in sorted(set(index["weight_map"].values())):
        os.link((source / shard).resolve(), output / shard)
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name not in (INDEX, "config.json"):
            if path.name not in (MANIFEST, "progress.json", "manifest.json"):
                shutil.copy2(path, output / path.name)
    target_bundle = output / "native-kernels"
    target_bundle.mkdir()
    package = options["helper_package"]
    (target_bundle / package).mkdir()
    for name, expected in provenance["_helpers"].items():
        path = bundle / package / name
        if file_digest(path) != expected:
            raise ValueError("kernel helper differs from provenance")
        shutil.copy2(path, target_bundle / package / name)
    for name in (
        f"glm_reconstruction_bridge_v{options['version']}.so",
        "glm_fused_gate_up.bin",
        "glm_fused_down.bin",
        "glm_fused_pack.bin",
        "provenance.json",
    ):
        shutil.copy2(bundle / name, target_bundle / name)
    if "glm_fused_reduce.bin" in provenance:
        shutil.copy2(bundle / "glm_fused_reduce.bin", target_bundle / "glm_fused_reduce.bin")
    config["ascend_glm_expert_layout"] = LAYOUT
    config["ascend_glm_native_activation_bits"] = activation_bits
    config["architectures"] = ["Glm5NextW2ForCausalLM"]
    write_json(output / "config.json", config)
    write_json(output / INDEX, index)
    manifest = {
        "schema_version": 1,
        "layout": LAYOUT,
        "complete": False,
        "source": str(source),
        "source_index_sha256": file_digest(source / INDEX),
        "world_size": world_size,
        "num_experts": experts,
        "layers": layers,
        "activation_bits": activation_bits,
        "kernel_bundle": "native-kernels",
        "kernel_provenance_sha256": file_digest(target_bundle / "provenance.json"),
    }
    write_json(output / MANIFEST, manifest)
    return manifest


def export_rank(layout, source, output, rank, world_size):
    """Snapshot existing prepared banks; never invoke a layout transformation.

    Every full bank must match its prepared SHA. One expert per bank is also
    checked against an independent canonical-checkpoint packing reference to
    detect a layer-order or projection mix-up before publishing the shard.
    """
    source, output = Path(source), Path(output)
    manifest = json.loads((output / MANIFEST).read_text())
    config, index, layers, experts = source_geometry(source)
    if manifest["complete"] or manifest["layout"] != LAYOUT or manifest["world_size"] != world_size:
        raise ValueError("export destination or rank geometry does not match")
    if not 0 <= rank < world_size or experts % world_size:
        raise ValueError("invalid export rank")
    if not layout.receipt["prepared"] or len(layout.backups) != 2 * len(layers):
        raise ValueError("export requires all native banks already prepared")
    text = config.get("text_config", config)
    hidden, inter = text["hidden_size"], text["moe_intermediate_size"]
    count = experts // world_size
    first = rank * count
    report = {"rank": rank, "complete": False, "layout": LAYOUT, "records": []}
    report_path = output / f"native-export-rank{rank}.json"
    layout.synchronize()
    for position, layer in enumerate(layers):
        tensors, hashes = {}, []
        for bank_index, projections, n, k in (
            (2 * position, ("gate", "up"), 2 * inter, hidden),
            (2 * position + 1, ("down",), hidden, inter),
        ):
            device = layout.backups[bank_index][0]
            cpu = device.detach().cpu()
            if cpu.shape[:2] != (count, n):
                raise ValueError("resident bank and checkpoint layer geometry differ")
            digest = tensor_digest(cpu)
            if digest != layout.receipt["bank_byte_hashes"][bank_index]["prepared"]:
                raise ValueError("resident native bank differs from prepared receipt")
            hashes.append(digest)
            for projection_index, projection in enumerate(projections):
                rows = n // len(projections)
                sample = f"model.language_model.layers.{layer}.mlp.experts.{first}.{projection}_proj_codes"
                with safe_open(str(source / index["weight_map"][sample]), framework="pt", device="cpu") as handle:
                    original = handle.get_tensor(sample)
                bits = original.shape[-1] * 8 // k
                expected = pack_cube(unpack_canonical_codes(original, k)[None], bits)[0]
                actual = cpu[0, projection_index * rows : (projection_index + 1) * rows]
                if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                    raise ValueError("layer/projection order differs from canonical checkpoint")
                for local in range(count):
                    name = f"model.language_model.layers.{layer}.mlp.experts.{first + local}.{projection}_proj_codes"
                    tensors[name] = cpu[local, projection_index * rows : (projection_index + 1) * rows].view(
                        torch.uint8
                    )
        filename = f"native-rank{rank}-layer{layer:03d}.safetensors"
        path = output / filename
        temporary = path.with_suffix(".tmp")
        save_file(tensors, str(temporary), metadata={"layout": LAYOUT, "rank": str(rank), "layer": str(layer)})
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        report["records"].append(
            {
                "layer": layer,
                "file": filename,
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
                "prepared_bank_sha256": hashes,
                "tensor_count": len(tensors),
            }
        )
        write_json(report_path, report)
        del tensors, cpu, actual, expected, original
    report["complete"] = True
    write_json(report_path, report)
    return {
        "passed": True,
        "rank": rank,
        "exported_layers": len(report["records"]),
        "layout": LAYOUT,
        "repacked_banks": 0,
        "report": str(report_path),
    }


def finalize(output):
    """Commit the authoritative index only after every rank's export passes."""
    output = Path(output)
    manifest = json.loads((output / MANIFEST).read_text())
    if manifest["complete"]:
        return manifest
    index = json.loads((output / INDEX).read_text())
    expected_names = {name for name in index["weight_map"] if CODE_NAME.fullmatch(name)}
    replaced, files = {}, []
    for rank in range(manifest["world_size"]):
        report = json.loads((output / f"native-export-rank{rank}.json").read_text())
        if report["complete"] is not True or report["layout"] != LAYOUT or report["rank"] != rank:
            raise ValueError("native export rank has not completed")
        if [record["layer"] for record in report["records"]] != manifest["layers"]:
            raise ValueError("native export layer coverage differs")
        for record in report["records"]:
            path = output / record["file"]
            if path.stat().st_size != record["bytes"]:
                raise ValueError("native export shard was truncated")
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if handle.metadata().get("layout") != LAYOUT:
                    raise ValueError("native export shard lacks layout metadata")
                for name in handle.keys():  # noqa: SIM118 - safetensors handle is not a dict
                    if name in replaced or name not in expected_names:
                        raise ValueError("duplicate or unexpected native expert tensor")
                    replaced[name] = record["file"]
            files.append(record)
    if set(replaced) != expected_names:
        raise ValueError("native export lacks routed expert code coverage")
    index["weight_map"].update(replaced)
    index["metadata"]["expert_layout"] = LAYOUT
    write_json(output / INDEX, index)
    manifest.update(complete=True, native_shards=files, native_tensor_count=len(replaced))
    write_json(output / MANIFEST, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "finalize"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--kernel-bundle", type=Path)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    if args.action == "initialize":
        if args.source is None or args.kernel_bundle is None:
            parser.error("initialize requires --source and --kernel-bundle")
        result = initialize(args.source, args.output, args.kernel_bundle, args.world_size, args.activation_bits)
    else:
        result = finalize(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
