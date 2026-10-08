# SPDX-License-Identifier: Apache-2.0
"""Reject compile-only vector converters until their exact device gates exist."""

import hashlib
import json

from .query_bf16_vector_probe import COUNTS, verify
from .resident_native import NativeManifest


def manifest(build, gate_report):
    build = build.resolve(strict=True)
    provenance = verify(build)
    if provenance.get("reused_bridge") is not False:
        raise ValueError("resident query vector requires a fresh versioned bridge namespace")
    gates = json.loads(gate_report.read_text())
    if gates.get("complete") is not True or gates.get("provenance") != provenance:
        raise ValueError("query vector requires complete matching hardware gates")
    records = gates.get("records", [])
    covered = {
        r.get("count")
        for r in records
        if all(r.get(key) is True for key in ("passed", "changed_replay", "input_guards", "output_padding"))
    }
    if covered != set(COUNTS) or len(records) != len(COUNTS):
        raise ValueError("query vector requires exhaustive FP16, tails, guards and changed replay")

    def entry(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    source = f"""def prepare():
    import importlib.util
    from pathlib import Path
    root=Path({str(build)!r})
    spec=importlib.util.spec_from_file_location('query_vector_resident_native',root/'query_bf16_vector.py')
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.NativeQueryVector(root,{provenance["namespace"]!r})
def validate(native):
    import importlib.util,os,tempfile
    from pathlib import Path
    root=Path({str(build)!r})
    spec=importlib.util.spec_from_file_location('query_vector_resident_gate',root/'query_bf16_vector_probe.py')
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    handle,path=tempfile.mkstemp(prefix='glm-query-vector-gate-',suffix='.json')
    os.close(handle)
    try:
        report=helper.run(root,Path(path),allow_device_gate=True,device=native.device.index)
        native.prepare_counts(helper.COUNTS)
        return {{'passed':report['complete'],'cases':len(report['records']),'changed_replay':True}}
    finally:
        Path(path).unlink(missing_ok=True)
"""
    return NativeManifest(
        dict(
            name=f"query_vector_v{provenance['version']}",
            libraries=[dict(provenance["bridge"])],
            assets=[entry(build / name) for name in provenance["assets"]],
            operators=[provenance["namespace"] + "::launch"],
            validation_source=source,
        )
    )
