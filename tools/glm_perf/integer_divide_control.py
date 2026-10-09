# SPDX-License-Identifier: Apache-2.0
"""Admit only the exact signed/replay-qualified integer metadata bundle."""

import hashlib
import json

from .integer_divide import DIVISION_GATE_COUNTS, DIVISORS, INTEGER_DIVIDE_ENTRY
from .resident_native import NativeManifest


def manifest(build, gate_report):
    build = build.resolve(strict=True)
    provenance = json.loads((build / "provenance.json").read_text())
    options = provenance["_build"]
    gates = json.loads(gate_report.read_text())
    if (
        not options.get("integer_metadata_divide")
        or options.get("integer_divide_entry") != INTEGER_DIVIDE_ENTRY
        or gates.get("complete") is not True
        or gates.get("build_options") != options
    ):
        raise ValueError("integer metadata requires complete matching gates")
    records = gates.get("records", [])
    coverage = {
        (r["dtype"], r["divisor"], r["count"])
        for r in records
        if r.get("passed") is True and r.get("changed_input_replay") is True and r.get("owned_padding_checked") is True
    }
    required = {
        (dtype, divisor, count)
        for dtype in ("torch.int32", "torch.int64")
        for divisor in DIVISORS
        for count in DIVISION_GATE_COUNTS
    }
    if coverage != required or any(r.get("signed_extremes") is not True for r in records):
        raise ValueError("integer metadata requires signed extremes, both widths and changed replay")
    package, namespace = options["helper_package"], options["namespace"]
    assets = (build / "glm_integer_divide.bin", build / f"glm_reconstruction_bridge_v{options['version']}.so")
    for path in assets:
        if gates.get("binaries", {}).get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("integer gates identify different binaries")
    helpers = [build / package / name for name in provenance["_helpers"]]
    for path in helpers:
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["_helpers"][path.name]:
            raise ValueError("frozen integer helper changed")

    def entry(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    source = f"""def prepare():
    import sys
    from pathlib import Path
    sys.path.insert(0, {str(build)!r})
    from {package}.integer_divide_probe import load
    return load(Path({str(build)!r}))[0]
def validate(native):
    import torch
    from {package}.integer_divide_probe import case
    # Worker RPCs run in inference mode, which omits Tensor._base metadata.
    # The standalone padding probe needs that metadata to audit owned storage.
    with torch.inference_mode(False):
        records=[case(native, dtype, divisor, 640)
                 for dtype in (torch.int32,torch.int64) for divisor in {DIVISORS!r}]
    return {{'passed':True,'cases':len(records),'signed_extremes':True,'changed_input_replay':True}}
"""
    return NativeManifest(
        {
            "name": f"integer_divide_v{options['version']}",
            "libraries": [entry(assets[1])],
            "assets": [entry(path) for path in (assets[0], *helpers)],
            "operators": [namespace + "::launch"],
            "validation_source": source,
        }
    )
