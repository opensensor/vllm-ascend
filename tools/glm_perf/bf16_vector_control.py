# SPDX-License-Identifier: Apache-2.0
"""Admit vector BF16 only with matching full precision, bounds and replay gates."""

import hashlib
import json

from .bf16_vector_probe import COUNTS, MODES, verify
from .resident_native import NativeManifest


def manifest(build, gate_report):
    build = build.resolve(strict=True)
    provenance = verify(build)
    gates = json.loads(gate_report.read_text())
    if gates.get("complete") is not True or gates.get("provenance") != provenance:
        raise ValueError("vector BF16 requires complete matching provenance")
    records = gates.get("records", [])
    required = {(count, mode) for count in COUNTS for mode in MODES}
    covered = {
        (record["count"], record["mode"])
        for record in records
        if all(record.get(key) is True for key in ("passed", "exact_bits", "changed_replay", "guards_checked"))
    }
    if covered != required or len(records) != len(required):
        raise ValueError("vector BF16 requires every precision, shape, bounds and replay gate")
    namespace, version = provenance["namespace"], provenance["version"]

    def entry(name):
        path = build / name
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    source = f"""def prepare():
    import importlib.util
    from pathlib import Path
    root=Path({str(build)!r})
    spec=importlib.util.spec_from_file_location({namespace + "_frozen_probe"!r},root/'bf16_vector_probe.py')
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    helper,provenance=probe.load(root)
    native=helper.NativeBf16Vector(root,provenance['namespace'])
    native.prepare_counts((17,65536))
    return native
def validate(native):
    import importlib.util
    from pathlib import Path
    import torch
    root=Path({str(build)!r})
    spec=importlib.util.spec_from_file_location({namespace + "_validation_probe"!r},root/'bf16_vector_probe.py')
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    helper,provenance=probe.load(root)
    for mode in {MODES!r}:
        for count in (17,65536):
            bits=probe.input_bits(count,mode,1)
            source_dtype=torch.bfloat16 if mode==1 else torch.float32
            dtype={{0:torch.bfloat16,1:torch.float32,4:torch.float32,5:torch.float16}}[mode]
            actual=native.convert(bits.view(source_dtype).to(native.device),dtype,mode)
            storage=torch.int32 if dtype==torch.float32 else torch.int16
            if not torch.equal(actual.cpu().view(storage),helper.reference(bits,mode)):
                raise ValueError('vector BF16 validation changed storage bits')
    return {{'passed':True,'cases':8,'exact_bits':True,'qualified_modes':{MODES!r}}}
"""
    bridge = f"glm_bf16_vector_bridge_v{version}.so"
    return NativeManifest(
        dict(
            name=f"bf16_vector_v{version}",
            libraries=[entry(bridge)],
            assets=[entry(name) for name in provenance["assets"] if name != bridge],
            operators=[namespace + "::launch"],
            validation_source=source,
        )
    )
