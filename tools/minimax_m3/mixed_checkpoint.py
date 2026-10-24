# SPDX-License-Identifier: Apache-2.0
"""Pinned, bounded M3.1 checkpoint inspection and CPU dequantization.

This is a preparation tool, not a runnable checkpoint exporter. Range reads
fetch only requested tensors; header accounting does not verify full payloads.
"""

import argparse
import hashlib
import json
import math
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import regex as re
import requests
import torch

from tools.minimax_m3.export_w4 import EXPERT_NAME, atomic_json
from tools.minimax_m3.serving_budget import make_budget

REPOSITORY = "MiniMaxAI/MiniMax-M3.1-Flash"
REVISION = "54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e"
HEADER_LIMIT = 16 << 20
RANGE_CHUNK = 8 << 20
DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
}
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_nvfp4(packed, scales, global_scale):
    """Compressed-tensors layout: low nibble first, FP8 scale / global scale."""
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] % 8:
        raise ValueError("NVFP4 requires a U8 matrix with groups of 16 values")
    if scales.dtype != torch.float8_e4m3fn or scales.shape != (packed.shape[0], packed.shape[1] // 8):
        raise ValueError("NVFP4 scale dtype/shape mismatch")
    if global_scale.numel() != 1 or not torch.isfinite(global_scale).all() or not (global_scale > 0).all():
        raise ValueError("NVFP4 global scale must be finite and positive")
    scale = scales.float()
    if not torch.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("NVFP4 block scales must be finite and nonnegative")
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    values = lut[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    result = values * (scale / global_scale.float()).repeat_interleave(16, -1)
    if not torch.isfinite(result).all():
        raise ValueError("NVFP4 reconstruction overflow")
    return result


def dequant_mxfp8(weight, scales):
    """Raw U8 E8M0 exponents, not numerical uint8 multipliers."""
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2 or weight.shape[1] % 32:
        raise ValueError("MXFP8 requires an E4M3 matrix with groups of 32 values")
    if scales.dtype != torch.uint8 or scales.shape != (weight.shape[0], weight.shape[1] // 32):
        raise ValueError("MXFP8 scale dtype/shape mismatch")
    if (scales == 255).any() or not torch.isfinite(weight.float()).all():
        raise ValueError("Nonfinite MXFP8 weight or E8M0 NaN scale")
    result = weight.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    if not torch.isfinite(result).all():
        raise ValueError("MXFP8 reconstruction overflow")
    return result


def fetch_range(url, start, stop):
    """Require exact HTTP partial content; never accidentally download a shard."""
    with requests.get(
        url + f"?range={start}-{stop}", headers={"Range": f"bytes={start}-{stop}"}, timeout=90, stream=True
    ) as response:
        response.raise_for_status()
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status_code != 206 or not match or tuple(map(int, match.groups()[:2])) != (start, stop):
            raise ValueError("Server did not honor exact byte range")
        data = response.raw.read(stop - start + 2)
        if len(data) != stop - start + 1:
            raise ValueError("Truncated or oversized range response")
        return data, int(match[3])


class MixedCheckpoint:
    def __init__(self, model, cache, revision=REVISION):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("An immutable Hub revision is required")
        self.model, self.cache, self.revision = Path(model), Path(cache) / revision, revision
        self.cache.mkdir(parents=True, exist_ok=True)
        self.config = json.loads((self.model / "config.json").read_text())
        self.weight_map = json.loads((self.model / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def url(self, shard):
        if Path(shard).name != shard:
            raise ValueError("Expected shard basename")
        return f"https://huggingface.co/{REPOSITORY}/resolve/{self.revision}/{shard}"

    def header(self, shard):
        if shard in self.headers:
            return self.headers[shard]
        path = self.cache / (shard + ".json")
        if path.exists():
            receipt = json.loads(path.read_text())
        else:
            prefix, size = fetch_range(self.url(shard), 0, 7)
            length = struct.unpack("<Q", prefix)[0]
            if not 0 < length <= HEADER_LIMIT:
                raise ValueError("Invalid header length")
            raw, total = fetch_range(self.url(shard), 8, 7 + length)
            if size != total:
                raise ValueError("Shard size changed")
            receipt = {"header": json.loads(raw), "payload_start": length + 8, "file_size": size}
            atomic_json(path, receipt)
        cursor = 0
        tensors = {k: v for k, v in receipt["header"].items() if k != "__metadata__"}
        for name, info in sorted(tensors.items(), key=lambda kv: kv[1]["data_offsets"]):
            dtype = DTYPES[info["dtype"]]
            start, stop = info["data_offsets"]
            if any(type(d) is not int or d < 0 for d in info["shape"]):
                raise ValueError("Invalid tensor dimensions")
            count = math.prod(info["shape"]) * torch.empty((), dtype=dtype).element_size()
            if start != cursor or stop - start != count or self.weight_map.get(name) != shard:
                raise ValueError("Index/header/offset mismatch")
            cursor = stop
        if cursor + receipt["payload_start"] != receipt["file_size"]:
            raise ValueError("Shard payload size mismatch")
        receipt["header"] = tensors
        self.headers[shard] = receipt
        return receipt

    def tensor(self, name):
        shard = self.weight_map[name]
        receipt = self.header(shard)
        info = receipt["header"][name]
        start, stop = info["data_offsets"]
        path = self.cache / (hashlib.sha256(name.encode()).hexdigest() + ".bin")
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                for offset in range(start, stop, RANGE_CHUNK):
                    data, total = fetch_range(
                        self.url(shard),
                        receipt["payload_start"] + offset,
                        receipt["payload_start"] + min(stop, offset + RANGE_CHUNK) - 1,
                    )
                    if total != receipt["file_size"]:
                        raise ValueError("Shard size changed")
                    handle.write(data)
            temporary.replace(path)
        data = bytearray(path.read_bytes())
        if len(data) != stop - start:
            raise ValueError("Cached tensor length mismatch")
        return torch.frombuffer(data, dtype=DTYPES[info["dtype"]]).reshape(info["shape"]).clone()

    def weight(self, name):
        base = name.removesuffix(".weight")
        if base + ".weight_packed" in self.weight_map:
            return dequant_nvfp4(
                *(self.tensor(base + suffix) for suffix in (".weight_packed", ".weight_scale", ".weight_global_scale"))
            )
        weight = self.tensor(name)
        if base + ".weight_scale_inv" in self.weight_map:
            return dequant_mxfp8(weight, self.tensor(base + ".weight_scale_inv"))
        if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unpaired quantized weight: {name}")
        return weight

    def audit(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.header, sorted(set(self.weight_map.values()))))
        infos = {name: info for receipt in self.headers.values() for name, info in receipt["header"].items()}
        if infos.keys() != self.weight_map.keys():
            raise ValueError("Incomplete checkpoint index coverage")
        records, consumed = {}, set()
        for name, info in infos.items():
            if name.endswith(".weight_packed"):
                base = name.removesuffix(".weight_packed")
                output = base + ".weight"
                if not EXPERT_NAME.fullmatch(output) or info["dtype"] != "U8" or len(info["shape"]) != 2:
                    raise ValueError("Unexpected packed expert")
                n, half_k = info["shape"]
                k = half_k * 2
                text = self.config["text_config"]
                expected_shape = [text["intermediate_size"], text["hidden_size"]]
                if EXPERT_NAME.fullmatch(output)[3] == "w2":
                    expected_shape.reverse()
                if [n, k] != expected_shape:
                    raise ValueError("Expert geometry differs from config")
                if n % 32 or k % 32:
                    raise ValueError("Expert is not block32 tiled")
                expected = {
                    ".weight_scale": ("F8_E4M3", [n, k // 16]),
                    ".weight_global_scale": ("F32", []),
                    ".input_global_scale": ("F32", []),
                }
                for suffix, (dtype, shape) in expected.items():
                    entry = infos[base + suffix]
                    if entry["dtype"] != dtype or entry["shape"] != shape:
                        raise ValueError("NVFP4 companion geometry mismatch")
                    consumed.add(base + suffix)
                tensors = {
                    "codes": {"dtype": "U8", "shape": [n, half_k]},
                    "scales": {"dtype": "F16", "shape": [n // 32, k // 32]},
                }
                records[output] = {"expert": True, "tensors": tensors}
                consumed.add(name)
            elif info["dtype"] == "F8_E4M3" and name.endswith(".weight"):
                n, k = info["shape"]
                scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
                scale = infos[scale_name]
                if k % 32 or scale["dtype"] != "U8" or scale["shape"] != [n, k // 32]:
                    raise ValueError("MXFP8 companion geometry mismatch")
                records[name] = {"expert": False, "tensors": {name: {"dtype": "F16", "shape": [n, k]}}}
                consumed.update((name, scale_name))
        for name in infos.keys() - consumed:
            info = infos[name]
            if info["dtype"] not in ("BF16", "F16", "F32") or not name.endswith(
                (".weight", ".bias", ".e_score_correction_bias")
            ):
                raise ValueError(f"Unaccounted tensor: {name}")
            records[name] = {
                "expert": False,
                "tensors": {name: {"dtype": "F32" if info["dtype"] == "F32" else "F16", "shape": info["shape"]}},
            }
        text = self.config["text_config"]
        expected = {
            f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.weight"
            for layer, enabled in enumerate(text["moe_layer_freq"])
            if enabled
            for expert in range(text["num_local_experts"])
            for proj in ("w1", "w2", "w3")
        }
        if {n for n, r in records.items() if r["expert"]} != expected:
            raise ValueError("Incomplete expert coverage")
        budget = make_budget(
            {"weights": records, "tensor_count": len(records)}, self.config, device_gib=45.6, utilization=0.95
        )
        return {
            "repository": REPOSITORY,
            "revision": self.revision,
            "validation": "remote_headers_only_not_converted_or_hardware_validated",
            "source_payload_bytes": sum(r["file_size"] - r["payload_start"] for r in self.headers.values()),
            "config_sha256": hashlib.sha256((self.model / "config.json").read_bytes()).hexdigest(),
            "index_sha256": hashlib.sha256((self.model / "model.safetensors.index.json").read_bytes()).hexdigest(),
            "source_tensors": len(infos),
            "budget": budget,
            "dspark": "Separate subdirectory excluded; no speculative decoding in this budget",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = MixedCheckpoint(args.model, args.cache).audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
