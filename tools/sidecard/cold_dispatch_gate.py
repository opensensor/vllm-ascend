# SPDX-License-Identifier: Apache-2.0
"""Arbitrary ownership and two-chip expert merge against the CPU quantizer."""

import argparse
import importlib
import json
import sys
from pathlib import Path

from .cold_dispatch import create_dispatch
from .cold_placement import cuda_keys, load
from .layer_weights import CudaExperts, private_native_banks
from .native_expert_probe import contained, digest_file
from .resident import NativeResidentPlan

GATE_LAYERS = (18, 19)


def scenarios(hot):
    cold = [expert for expert in range(288) if expert not in hot]
    low, high = [expert for expert in cold if expert < 144], [expert for expert in cold if expert >= 144]
    return {
        "hot_only": hot[:8],
        "hot_tail": hot[-8:],
        "cold_rank0": low[:8],
        "cold_rank1": high[:8],
        "cold_tail0": low[-8:],
        "cold_tail1": high[-8:],
        "cold_both": low[:4] + high[:4],
        "mixed_both": [hot[0], hot[-1]] + low[:3] + high[:3],
    }


def run(config_path, output):
    """Run real arbitrary expert-bank mappings while both cold workers coexist.

    Expected values use individually unpacked authenticated checkpoint experts,
    with the frozen CPU activation quantizer and asymmetric SwiGLU clipping.
    Repeat requests change amplitudes and row counts to expose stale buffers.
    """
    import torch

    spec = json.loads(Path(config_path).read_bytes())
    if tuple(spec.get("cold_gate_layers", ())) != GATE_LAYERS:
        raise ValueError("bounded cold gate requires only layers 18,19")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    hot = load(spec)
    cuda = dispatcher = None
    report = {
        "passed": False,
        "real_checkpoint_weights": True,
        "physical_npu_devices": [2, 3],
        "hccl": False,
        "cases": [],
    }
    try:
        dispatcher = create_dispatch(config_path, None, torch, wait=False)
        cuda = CudaExperts(
            Path(spec["checkpoint"]),
            spec["checkpoint_sha256"],
            spec["cuda_backend"],
            torch,
            keys=cuda_keys(hot, gate_layers=GATE_LAYERS),
        )
        dispatcher.cuda = cuda
        dispatcher.ready()
        root = Path(spec["checkpoint"])
        manifest = json.loads((root / "native-layout.json").read_bytes())
        bundle = contained(root, manifest["kernel_bundle"])
        provenance = json.loads((bundle / "provenance.json").read_bytes())
        package = provenance["_build"]["helper_package"]
        for name, checksum in provenance["_helpers"].items():
            if digest_file(bundle / package / name) != checksum:
                raise ValueError("cold CPU reference helper changed")
        sys.path.insert(0, str(bundle))
        projection = importlib.import_module(package + ".fused_moe_probe").projection_reference
        unpack = importlib.import_module(package + ".fused_weight_layout").unpack_cube
        with torch.inference_mode():
            for layer in GATE_LAYERS:
                for name, ids in scenarios(hot[str(layer)]).items():
                    plan = NativeResidentPlan(root, tuple((layer, expert) for expert in ids), spec["checkpoint_sha256"])
                    plan.authenticate()
                    bank = next(private_native_banks(plan, torch))[1]
                    g, gs, d, ds = bank["tensors"]
                    signed_g, signed_d = unpack(g, 4096), unpack(d, 2048)
                    for rows in (1, 4, 17, 32):
                        torch.manual_seed(72711 + rows + layer)
                        hidden = (torch.randn(rows, 4096) * 4).half()
                        selected = torch.tensor(ids, dtype=torch.int64).expand(rows, -1).contiguous()
                        weights = (torch.arange(1, 9).float() / 36 * 2.5).expand(rows, -1).contiguous()
                        expected = torch.zeros(rows, 4096)
                        clipped = 0
                        for local in range(8):
                            gu = projection(hidden, signed_g[local], gs[local].float(), 4)
                            gate, up = gu.float().chunk(2, -1)
                            clipped += int(((gate > 10) | (up.abs() > 10)).sum())
                            activation = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).half()
                            expected += (
                                projection(activation, signed_d[local], ds[local].float(), 4).float()
                                * weights[:, local, None]
                            )
                        actual = dispatcher(layer, hidden.cuda(), selected.cuda(), weights.cuda()).cpu()
                        repeated = dispatcher(layer, hidden.cuda(), selected.cuda(), weights.cuda()).cpu()
                        torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
                        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
                        report["cases"].append(
                            {
                                "layer": layer,
                                "case": name,
                                "rows": rows,
                                "passed": True,
                                "repeat_bitwise_equal": True,
                                "clipping_activated_elements": clipped,
                                "max_abs_error": float((actual - expected).abs().max()),
                                "global_expert_ids": ids,
                            }
                        )
        report["passed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
        if dispatcher is not None:
            dispatcher.close()
        if cuda is not None:
            cuda.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
