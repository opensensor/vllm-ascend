# SPDX-License-Identifier: Apache-2.0
"""Manifest gate for the two native fused MoE stages."""

import hashlib
import json

from tools.glm_perf.resident_native import NativeManifest


def manifest(build_dir, gate_report):
    root = build_dir.resolve(strict=True)
    provenance = json.loads((root / "provenance.json").read_text())
    options = provenance["_build"]
    gates = json.loads(gate_report.read_text())
    required = {(weight, activation) for weight in (2, 3, 4) for activation in (4, 8)}
    if gates.get("complete") is not True or gates.get("build_options") != options:
        raise ValueError("fused MoE requires complete matching binary gates")
    for key in ("records", "real_weight_records"):
        rows = gates.get(key, [])
        if not rows or any(
            r.get("passed") is not True
            or not r.get("graph_changed_inputs_routes_weights")
            or r.get("fp16_intermediate_gm_bytes") != 0
            for r in rows
        ):
            raise ValueError("fused MoE requires independent arithmetic, replay and real-weight gates")
        if {(r.get("weight_bits"), r.get("activation_bits")) for r in rows} != required:
            raise ValueError("fused MoE requires all weight and activation precisions")
    if {(r["weight_bits"], r["activation_bits"]) for r in gates["records"] if r["tokens"] > 64} != required:
        raise ValueError("fused MoE requires prefill gates")
    namespace = options["namespace"]
    package = options["helper_package"]
    package_root = root / package
    prepared_layout = options.get("prepared_weight_layout", False)
    bridge = root / f"glm_reconstruction_bridge_v{options['version']}.so"
    binaries = (bridge, root / "glm_fused_gate_up.bin", root / "glm_fused_down.bin", root / "glm_fused_pack.bin")
    for path in binaries:
        if gates.get("binaries", {}).get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("fused gates do not identify these exact binaries")
    helpers = {name: hashlib.sha256((package_root / name).read_bytes()).hexdigest() for name in provenance["_helpers"]}
    if helpers != provenance["_helpers"]:
        raise ValueError("frozen fused helpers differ from build provenance")
    source = f"""def prepare():
    import hashlib, importlib, importlib.util, sys
    from pathlib import Path
    package={package!r}
    root=Path({str(package_root)!r})
    if package not in sys.modules:
        spec=importlib.util.spec_from_file_location(package,root/'__init__.py',submodule_search_locations=[str(root)])
        module=importlib.util.module_from_spec(spec)
        sys.modules[package]=module
        spec.loader.exec_module(module)
    if Path(sys.modules[package].__file__).resolve() != (root/'__init__.py').resolve():
        raise ValueError('fused helper package belongs to another build')
    for name, expected in {helpers!r}.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('fused helper changed: '+name)
    from {package}.glm_fused_moe import NativeFusedMoE
    build=Path({str(root)!r})
    resources={{'fused_int4a'+str(bits): NativeFusedMoE(build,namespace={namespace!r},activation_bits=bits,
                                                     prepared_weight_layout={prepared_layout!r})
               for bits in (8,4)}}
    if {prepared_layout!r}:
        import torch
        from {package}.fused_weight_layout import PreparedWeightLayout
        report=Path({str(root)!r})/('weight-layout-rank'+str(torch.distributed.get_rank())+'.json')
        resources['_weight_layout']=PreparedWeightLayout(torch.npu.synchronize,torch.distributed.get_world_size(),
                                                         report_path=report)
    return resources

def validate(resources):
    from {package}.fused_moe_probe import gate_case
    records=[gate_case(resources['fused_int4a'+str(activation)],bits) for activation in (8,4) for bits in (2,3,4)]
    return {{'passed':True,'cases':len(records),'native_fused_stages':2,
             'input_quantization_passes_per_token':1,
             'fp16_intermediate_gm_bytes':0,'model_quality':'not_evaluated'}}
"""

    def entry(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    return NativeManifest(
        {
            "name": namespace.removeprefix("glm_"),
            "libraries": [entry(bridge)],
            "assets": [entry(p) for p in (*binaries[1:], *(package_root / n for n in helpers))],
            "operators": [namespace + "::launch"],
            "validation_source": source,
        }
    )
