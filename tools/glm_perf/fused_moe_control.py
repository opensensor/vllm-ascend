# SPDX-License-Identifier: Apache-2.0
"""Manifest gate for the two native fused MoE stages."""

import hashlib
import json

from tools.glm_perf.glm_fused_moe import FUSED_REDUCTION_TOKENS
from tools.glm_perf.resident_native import NativeManifest


def manifest(build_dir, gate_report):
    root = build_dir.resolve(strict=True)
    provenance = json.loads((root / "provenance.json").read_text())
    options = provenance["_build"]
    gates = json.loads(gate_report.read_text())
    required = {(weight, activation) for weight in (2, 3, 4) for activation in (4, 8)}
    if gates.get("complete") is not True or gates.get("build_options") != options:
        raise ValueError("fused MoE requires complete matching binary gates")

    def valid_workspace(row):
        if not options.get("fp16_route_workspace"):
            return row.get("fp16_intermediate_gm_bytes") == 0
        dimensions = tuple(row.get(name) for name in ("tokens", "top_k", "hidden"))
        if any(type(value) is not int or value <= 0 for value in dimensions):
            return False
        tokens, top_k, hidden = dimensions
        expected = tokens * top_k * hidden * 2 if tokens > FUSED_REDUCTION_TOKENS else 0
        return (
            row.get("route_workspace_dtype") == "torch.float16"
            and row.get("fp16_gate_up_hidden_gm_bytes") == 0
            and row.get("fp16_intermediate_gm_bytes") == expected
            and row.get("unweighted_fp16_workspace_bytes") == expected
            and row.get("weighted_fp32_workspace_bytes") == 0
        )

    for key in ("records", "real_weight_records"):
        rows = gates.get(key, [])
        if not rows or any(
            r.get("passed") is not True or not r.get("graph_changed_inputs_routes_weights") or not valid_workspace(r)
            for r in rows
        ):
            raise ValueError("fused MoE requires independent arithmetic, replay and real-weight gates")
        if options.get("prerounded_weight_scales") and any(r.get("prerounded_weight_scales") is not True for r in rows):
            raise ValueError("prerounded kernels require explicitly prepared scale gates")
        if options.get("fp16_weight_scales") and any(r.get("fp16_weight_scales") is not True for r in rows):
            raise ValueError("FP16 storage kernels require explicitly matching scale gates")
        if options.get("compact_w4_scratch") and any(r.get("compact_w4_scratch") is not True for r in rows):
            raise ValueError("compact W4 kernels require explicit specialized dispatch gates")
        if {(r.get("weight_bits"), r.get("activation_bits")) for r in rows} != required:
            raise ValueError("fused MoE requires all weight and activation precisions")
    if {(r["weight_bits"], r["activation_bits"]) for r in gates["records"] if r["tokens"] > 64} != required:
        raise ValueError("fused MoE requires prefill gates")
    if any(
        options.get(flag)
        for flag in (
            "compact_w4_scratch",
            "prefill_weight_cache",
            "prerounded_weight_scales",
            "fp16_weight_scales",
            "fp16_route_workspace",
            "share_gate_up_input",
            "cache_gate_up_activations",
            "vector_scale_products",
            "gather_product_matrix",
            "prefill_rows_32",
            "active_cube_rows",
            "direct_w4_l1",
            "prepared_offset_tables",
            "quad_hidden_quant",
            "direct_hidden_gather",
            "route_packed_input",
        )
    ):
        covered = {
            (r["weight_bits"], r["activation_bits"])
            for r in gates["real_weight_records"]
            if r["tokens"] > FUSED_REDUCTION_TOKENS
        }
        if covered != required:
            raise ValueError("prefill memory experiments require real-weight multibatch gates")
    namespace = options["namespace"]
    package = options["helper_package"]
    package_root = root / package
    prepared_layout = options.get("prepared_weight_layout", False)
    lookup_args = ",weight_decode_lut=True" if options.get("weight_decode_lut") else ""
    route_args = ",fp16_route_workspace=True" if options.get("fp16_route_workspace") else ""
    bridge = root / f"glm_reconstruction_bridge_v{options['version']}.so"
    binaries = (bridge, root / "glm_fused_gate_up.bin", root / "glm_fused_down.bin", root / "glm_fused_pack.bin")
    if "glm_fused_reduce.bin" in provenance:
        binaries += (root / "glm_fused_reduce.bin",)
    if options.get("route_packed_input"):
        binaries += (root / "glm_fused_route_input.bin",)
    if options.get("specialize_w3"):
        binaries += (root / "glm_fused_gate_up_w3.bin", root / "glm_fused_down_w3.bin")
    if options.get("compact_w4_scratch"):
        binaries += (root / "glm_fused_gate_up_w4.bin", root / "glm_fused_down_w4.bin")
    for path in binaries:
        if gates.get("binaries", {}).get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("fused gates do not identify these exact binaries")
    helpers = {name: hashlib.sha256((package_root / name).read_bytes()).hexdigest() for name in provenance["_helpers"]}
    if helpers != provenance["_helpers"]:
        raise ValueError("frozen fused helpers differ from build provenance")
    validation_tokens = (2, 17) if "glm_fused_reduce.bin" in provenance else (2,)
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
                                                     prepared_weight_layout={prepared_layout!r}{lookup_args}{route_args})
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
    records=[gate_case(resources['fused_int4a'+str(activation)],bits,tokens=tokens)
             for activation in (8,4) for bits in (2,3,4) for tokens in {validation_tokens!r}]
    return {{'passed':True,'cases':len(records),'native_fused_stages':2,
             'input_quantization_passes_per_token':1,
             'fp16_intermediate_gm_bytes':max(r['fp16_intermediate_gm_bytes'] for r in records),
             'model_quality':'not_evaluated'}}
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
