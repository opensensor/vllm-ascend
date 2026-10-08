# SPDX-License-Identifier: Apache-2.0
"""Admit tiled normalization only after complete matching arithmetic and replay."""

import hashlib
import json

from .resident_native import NativeManifest
from .sinkhorn_tiled import reduction_order
from .sinkhorn_tiled_probe import COUNTS, ITERATIONS, LOGIT_SCALES, verify


def manifest(build, gate_report):
    build = build.resolve(strict=True)
    provenance = verify(build)
    gates = json.loads(gate_report.read_text())
    if (
        gates.get("complete") is not True
        or gates.get("provenance") != provenance
        or gates.get("order") is not None
        or gates.get("row_orders") != {str(rows): reduction_order(rows) for rows in COUNTS}
    ):
        raise ValueError("tiled normalization requires complete matching shape-order gates")
    required = {(rows, iterations, scale) for rows in COUNTS for iterations in ITERATIONS for scale in LOGIT_SCALES}
    records = gates.get("records", [])
    covered = {
        (r["rows"], r["iterations"], r["logit_scale"])
        for r in records
        if all(r.get(k) is True for k in ("passed", "exact_fp32", "changed_replay", "guards_checked"))
    }
    if covered != required or len(records) != len(required):
        raise ValueError("tiled normalization requires all shape, bounds and changed-replay gates")
    namespace, version = provenance["namespace"], provenance["version"]

    def entry(name):
        path = build / name
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    source = f"""def prepare():
    import importlib.util
    from pathlib import Path
    root=Path({str(build)!r})
    spec=importlib.util.spec_from_file_location({namespace + "_frozen_probe"!r},root/'sinkhorn_tiled_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    helper,provenance=module.load(root)
    native=helper.SinkhornTiled(root,provenance['namespace'])
    native.prepare_counts({COUNTS!r})
    return native
def validate(native):
    import torch
    gen=torch.Generator().manual_seed(970)
    for rows in (1,2,8,640):
        mix=torch.softmax(torch.randn(rows,4,4,generator=gen).to(native.device),dim=-1)+1e-6
        expected=mix/(mix.sum(dim=-2,keepdim=True)+1e-6)
        for _ in range(19):
            expected=expected/(expected.sum(dim=-1,keepdim=True)+1e-6)
            expected=expected/(expected.sum(dim=-2,keepdim=True)+1e-6)
        actual=native(mix)
        torch.npu.synchronize()
        if not torch.equal(actual.cpu().view(torch.int32),expected.cpu().view(torch.int32)):
            raise ValueError('normalization FP32 bits changed')
    return {{'passed':True,'cases':4,'exact_fp32':True,'qualified_rows':{COUNTS!r}}}
"""
    bridge = f"glm_sinkhorn_tiled_bridge_v{version}.so"
    return NativeManifest(
        dict(
            name=f"sinkhorn_tiled_v{version}",
            libraries=[entry(bridge)],
            assets=[entry(name) for name in provenance["assets"] if name != bridge],
            operators=[namespace + "::launch"],
            validation_source=source,
        )
    )
