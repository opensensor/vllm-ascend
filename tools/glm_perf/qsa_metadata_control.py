# SPDX-License-Identifier: Apache-2.0
"""Admit only matching signed/strided/replay-qualified QSA metadata builds."""

import hashlib
import json

from .resident_native import NativeManifest


def manifest(build, gate_report):
    build = build.resolve(strict=True)
    provenance = json.loads((build / "provenance.json").read_text())
    options = provenance["_build"]
    gates = json.loads(gate_report.read_text())
    if (
        not options.get("fused_qsa_metadata")
        or gates.get("complete") is not True
        or gates.get("build_options") != options
    ):
        raise ValueError("QSA metadata requires complete matching gates")
    covered = {
        (r["rows"], r["requests"], r["budget"], r["position_bytes"], r["split"])
        for r in gates.get("records", [])
        if all(
            r.get(key) is True
            for key in (
                "passed",
                "changed_replay",
                "signed",
                "guards_checked",
                "owned_padding_checked",
                "strided_inputs",
            )
        )
    }
    required = {
        (rows, requests, budget, width, split)
        for rows in (2, 8, 640)
        for requests in (1, 4)
        for budget in (4, 512)
        for width in (4, 8)
        for split in (1, 20)
    }
    if covered != required:
        raise ValueError("QSA metadata needs all signed, strided and graph replay gates")
    package, namespace = options["helper_package"], options["namespace"]
    bridge = build / f"glm_reconstruction_bridge_v{options['version']}.so"
    binary = build / "glm_qsa_metadata.bin"
    for path in (bridge, binary):
        if gates.get("binaries", {}).get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("QSA gates identify different binaries")
    helpers = [build / package / name for name in provenance["_helpers"]]
    for path in helpers:
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["_helpers"][path.name]:
            raise ValueError("frozen QSA helper changed")

    def entry(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    source = f"""def prepare():
    import sys
    from pathlib import Path
    sys.path.insert(0, {str(build)!r})
    from {package}.qsa_metadata_probe import load
    return load(Path({str(build)!r}))[0]
def validate(native):
    import torch
    from {package}.qsa_metadata_probe import case
    records=[case(native, rows=rows, position_dtype=dtype)
             for rows in (2,8) for dtype in (torch.int32,torch.int64)]
    return {{'passed':True,'cases':len(records),'signed':True,'changed_replay':True}}
"""
    return NativeManifest(
        dict(
            name=f"qsa_metadata_v{options['version']}",
            libraries=[entry(bridge)],
            assets=[entry(path) for path in (binary, *helpers)],
            operators=[namespace + "::launch"],
            validation_source=source,
        )
    )
