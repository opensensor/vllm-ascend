# SPDX-License-Identifier: Apache-2.0
"""Require exact gapped-bank/graph gates before loading selected-state copies."""

import hashlib
import json

from .resident_native import NativeManifest


def manifest(build, gate_report):
    build = build.resolve(strict=True)
    provenance = json.loads((build / "provenance.json").read_text())
    options = provenance["_build"]
    gates = json.loads(gate_report.read_text())
    if not options.get("selected_state_rows") or not gates.get("complete") or gates.get("build_options") != options:
        raise ValueError("state copies require complete matching binary gates")
    covered = {
        (tuple(r["payload_shape"]), r["gap"], r["selected"], r["index_bytes"])
        for r in gates.get("records", [])
        if all(
            r.get(key) is True
            for key in ("passed", "changed_replay", "all_backing_bytes_checked", "negative_slots", "fresh_flags")
        )
    }
    required = {
        (shape, gap, selected, size)
        for shape in ((2, 16, 16), (16, 128, 128))
        for gap in (0, 192)
        for selected in (1, 4)
        for size in (4, 8)
    }
    if covered != required:
        raise ValueError("state copies require both index widths, page gaps and changed graph replay")
    package, namespace = options["helper_package"], options["namespace"]
    helpers = build / package
    bridge = build / f"glm_reconstruction_bridge_v{options['version']}.so"
    binaries = (bridge, build / "glm_state_rows_gather.bin", build / "glm_state_rows_scatter.bin")
    for binary in binaries:
        if gates["binaries"].get(binary.name) != hashlib.sha256(binary.read_bytes()).hexdigest():
            raise ValueError("state-copy gates do not identify these exact binaries")
    for name, digest in provenance["_helpers"].items():
        if hashlib.sha256((helpers / name).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen state-copy helper changed")

    def entry(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    source = f"""def prepare():
    import sys
    sys.path.insert(0, {str(build)!r})
    from {package}.state_rows_probe import load_helper
    native, _ = load_helper(__import__('pathlib').Path({str(build)!r}))
    return native
def validate(native):
    import torch
    from {package}.state_rows_probe import case
    records=[case(native, index_dtype=dtype, selected=count)
             for dtype in (torch.int32,torch.int64) for count in (1,4)]
    return {{'passed':True,'cases':len(records),'full_backing_bitwise_checked':True}}
"""
    return NativeManifest(
        {
            "name": f"state_rows_v{options['version']}",
            "libraries": [entry(bridge)],
            "assets": [entry(p) for p in (*binaries[1:], *(helpers / name for name in provenance["_helpers"]))],
            "operators": [namespace + "::launch"],
            "validation_source": source,
        }
    )
