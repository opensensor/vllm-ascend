# SPDX-License-Identifier: Apache-2.0
"""Freeze BF16 kernel assets and derive operator names from one namespace."""

import hashlib
import json

from .resident_native import NativeManifest


def manifest(root):
    root = root.resolve(strict=True)
    options = json.loads((root / "options.json").read_text())
    namespace, name, bridge = options["namespace"], options["name"], options["bridge"]
    if not namespace.isidentifier() or not name.isidentifier() or "/" in bridge or not bridge.endswith(".so"):
        raise ValueError("invalid BF16 kernel bundle identity")
    for filename in ("gates.json", "compression-gates.json"):
        gates = json.loads((root / filename).read_text())
        if (
            gates.get("complete") is not True
            or not gates.get("records")
            or any(
                record.get("bitwise_equal") is not True or record.get("changed_input_replay") is not True
                for record in gates["records"]
            )
        ):
            raise ValueError("BF16 bundle requires exact conversion/compression replay gates")
        required = ("glm_bf16_cast.bin", bridge, "bf16_cast.py", "candidate.py")
        if any(
            gates.get("binaries", {}).get(name) != hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in required
        ):
            raise ValueError("BF16 gates do not identify these exact binaries and helpers")

    def entry(filename):
        path = root / filename
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    source = f"""def prepare():
    import importlib.util
    from pathlib import Path
    root=Path({str(root)!r})
    spec=importlib.util.spec_from_file_location({(namespace + "_helper")!r},root/'bf16_cast.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {{'bf16_cast_v1':module.NativeBF16Cast(root,{namespace!r})}}

def validate(resources):
    import torch
    convert=resources['bf16_cast_v1']
    value=torch.tensor([0.,-0.,1.00390625,-1.00390625,65504.,70000.,1e-30,-1e-30],device='npu')
    actual=convert(value,torch.bfloat16)
    assert torch.equal(actual.cpu().view(torch.int16),value.bfloat16().cpu().view(torch.int16))
    return {{'passed':True,'cases':1}}
"""
    return NativeManifest(
        {
            "name": name,
            "libraries": [entry(bridge)],
            "assets": [
                entry(filename)
                for filename in (
                    "glm_bf16_cast.bin",
                    "bf16_cast.py",
                    "candidate.py",
                    "options.json",
                    "gates.json",
                    "compression-gates.json",
                )
            ],
            "operators": [namespace + "::launch"],
            "validation_source": source,
        }
    )
