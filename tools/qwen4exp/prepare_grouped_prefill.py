# SPDX-License-Identifier: Apache-2.0
"""Seal prefill resources and bounded validation for the resident protocol."""

import argparse
import hashlib
import json
from pathlib import Path

from tools.glm_perf.resident_native import NativeManifest


def prepare(build, resource_source, paired_limbs=False):
    build = Path(build)
    provenance = json.loads((build / "provenance.json").read_text())
    torch_classes = provenance["namespace"]
    scope = {}
    # A dedicated namespace keeps the imported runtime module unchanged. The
    # exact resource implementation is a hashed manifest asset.
    exec(compile(Path(resource_source).read_text(), str(resource_source), "exec"), scope)
    torch = scope["torch"]
    kernel = getattr(torch.classes, torch_classes).Kernel
    launch = getattr(torch.ops, torch_classes).launch
    entries = (
        (
            "projection",
            "NativeGroupedPrefill",
            "native_grouped_prefill",
            "qwen_grouped_paired_prefill_v1" if paired_limbs else "qwen_grouped_local_prefill_v1",
        ),
        ("gather", "NativeLocalRouteGather", "native_route_gather", "qwen_local_route_gather_v1"),
    )
    return {
        name: scope[cls](kernel(provenance["binaries"][binary]["path"], symbol), launch)
        for name, cls, binary, symbol in entries
    }


def validate(resources):
    """Exact arithmetic/layout gate on each rank before binding resources.

    Full real-weight and serving measurements are separate qualification gates.
    This checks empty groups, local tails and poison in inactive route indices.
    """
    import torch

    from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device

    class Bank:
        pass

    cases = 0
    device = torch.device("npu", torch.npu.current_device())
    generator = torch.Generator().manual_seed(3847)
    with torch.inference_mode():
        for outputs, inputs in ((1280, 2560), (2560, 640)):
            bank = Bank()
            bank.weight = torch.randint(-128, 128, (3, outputs, inputs // 2), dtype=torch.int8, generator=generator).to(
                device
            )
            shape = (3, outputs, inputs // 128)
            bank.weight_scale = (torch.rand(shape, generator=generator) * 0.001).half().to(device)
            bank.weight_offset = torch.zeros(shape, dtype=torch.float16, device=device)
            bank.weight_sum = torch.zeros_like(bank.weight_offset)
            x = torch.randn(320, inputs, generator=generator).half().to(device)
            packed = pack_activation_device(x)
            for active in (0, 17, 129, 319, 320):
                ends = torch.tensor([0, active // 2, active], dtype=torch.int64, device=device)
                expected = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(
                    *packed, bank.weight, bank.weight_scale, bank.weight_offset, bank.weight_sum, ends
                )
                actual = resources["projection"](bank, packed, ends)
                torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
                cases += 1
            indices = torch.arange(320, dtype=torch.int32).remainder(x.shape[0])
            indices[17:] = 1 << 29
            ends = torch.tensor([0, 1, 17], dtype=torch.int64, device=device)
            gathered = resources["gather"](packed, indices.to(device), ends)
            for actual, original in zip(gathered, packed):
                torch.testing.assert_close(actual[:17].cpu(), original[:17].cpu(), rtol=0, atol=0)
            cases += 1
    torch.npu.synchronize()
    return {"passed": True, "cases": cases, "exact_bytes": True, "real_weight_serving_gate": False}


def make_manifest(build, resource_source, validation_source, paired_limbs=False):
    provenance = json.loads((build / "provenance.json").read_text())

    def entry(path):
        path = Path(path).resolve(strict=True)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    for name, digest in provenance["sources"].items():
        if (build / name).exists() and entry(build / name)["sha256"] != digest:
            raise ValueError("frozen native source differs from compiled provenance")
    binaries = [provenance["binaries"][name] for name in ("native_grouped_prefill", "native_route_gather")]
    assets = [
        entry(build / "provenance.json"),
        *binaries,
        *[entry(build / name) for name in provenance["sources"] if (build / name).exists()],
        entry(resource_source),
        entry(validation_source),
    ]
    source = (
        "from pathlib import Path\n"
        "def definitions():\n    scope = {'__name__': 'qwen_grouped_prefill_validation'}\n"
        f"    exec(compile(Path({str(validation_source)!r}).read_text(), {str(validation_source)!r}, 'exec'), scope)\n"
        "    return scope\n"
        "def prepare():\n    return definitions()['prepare']("
        f"{str(build)!r}, {str(resource_source)!r}, {paired_limbs!r})\n"
        "def validate(resources):\n    return definitions()['validate'](resources)\n"
    )
    value = {
        "name": provenance["namespace"],
        "libraries": [provenance["bridge"]],
        "assets": assets,
        "operators": [f"{provenance['namespace']}::launch"],
        "validation_source": source,
    }
    NativeManifest(value).verify_files()
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--resource-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-limbs", action="store_true")
    args = parser.parse_args()
    value = make_manifest(
        args.build.resolve(), args.resource_source.resolve(), Path(__file__).resolve(), args.paired_limbs
    )
    with args.output.open("x") as output:
        output.write(json.dumps(value, indent=2) + "\n")
