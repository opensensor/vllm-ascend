"""Three actual matrices; reconstruction proxy, never a language-quality claim."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.deepseek_w2.w2_format import unpack_codes
from tools.glm_w2.convert_full import dequant_fp8_e4m3_f32block

root = Path("/run/media/matteius/20TB-drive/models")
source = root / "GLM-5.3-Flash-FP8"
base = root / "GLM-5.3-Flash-selective-W3-310p"
w3 = root / "GLM-5.3-Flash-three-card-W3-canonical-candidate"
source_index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
base_index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
rows = []
for projection in ("gate", "up", "down"):
    stem = f"model.language_model.layers.33.mlp.experts.0.{projection}_proj"
    weight = stem + ".weight"
    with safe_open(str(source / source_index[weight]), framework="pt", device="cpu") as f:
        raw = f.get_tensor(weight)
        raw_scale = f.get_tensor(weight + "_scale_inv")
        expected = dequant_fp8_e4m3_f32block(raw, raw_scale)
    row = {"projection": projection, "parameters": expected.numel()}
    for bits, directory, filenames in [(2, base, base_index), (3, w3, None)]:
        code_name, scale_name = stem + "_codes", stem + "_scale"
        code_file = base / filenames[code_name] if filenames else w3 / "w3-overlay-layer-33.safetensors"
        scale_file = base / filenames[scale_name] if filenames else code_file
        with safe_open(str(code_file), framework="pt", device="cpu") as f:
            codes = unpack_codes(f.get_tensor(code_name), expected.shape[-1], n_bits=bits).float()
        with safe_open(str(scale_file), framework="pt", device="cpu") as f:
            scales = f.get_tensor(scale_name).half().float()
        reconstructed = codes * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
        error = (reconstructed - expected).square().sum()
        row[f"w{bits}"] = {
            "relative_squared_error": float(error / expected.square().sum()),
            "cosine": float(
                torch.nn.functional.cosine_similarity(reconstructed.reshape(1, -1), expected.reshape(1, -1))
            ),
        }
    rows.append(row)
output = {
    "validation": "CPU weight reconstruction proxy only; not inference, activation or language quality",
    "selection": "layer33, expert0, gate/up/down; narrow sample, not representative of every bank",
    "scale_contract": "FP16-round/promote matching native storage",
    "matrices": rows,
}
Path("artifacts/glm-perf-310p/card-profiles-20261009/weight-proxy.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
