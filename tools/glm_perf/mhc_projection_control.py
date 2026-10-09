# SPDX-License-Identifier: Apache-2.0
"""Admit a qualified append-only projection; model weights are gated separately."""

import hashlib
import json

from tools.glm_perf.resident_native import NativeManifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(build, gates):
    provenance = json.loads((build / "provenance.json").read_text())
    options = provenance["_build"]
    report = json.loads(gates.read_text())
    kernel = build / "glm_mhc_projection.bin"
    package = options["helper_package"]
    helper = build / package / "mhc_projection.py"
    if (
        digest(kernel) != provenance[kernel.name]["binary_sha256"]
        or digest(helper) != provenance["_helpers"][helper.name]
    ):
        raise ValueError("projection binary or helper differs from provenance")
    if (
        report.get("complete") is not True
        or report.get("binary_sha256") != digest(kernel)
        or report.get("helper_sha256") != digest(helper)
        or {row.get("rows") for row in report.get("cases", [])} != {1, 16, 33, 640, 1280}
        or any(not row.get("finite") or not row.get("graph_changed_inputs") for row in report["cases"])
    ):
        raise ValueError("projection requires complete matching arithmetic and replay gates")
    namespace = options["namespace"]
    source = f"""def prepare():
    import importlib.util,sys
    from pathlib import Path
    import torch
    root=Path({str(build)!r})
    package={package!r}
    if package not in sys.modules:
        spec=importlib.util.spec_from_file_location(package,root/package/'__init__.py',submodule_search_locations=[str(root/package)])
        module=importlib.util.module_from_spec(spec);sys.modules[package]=module;spec.loader.exec_module(module)
    from {package}.mhc_projection import NativeMhcProjection
    return NativeMhcProjection(root,{namespace!r},device=torch.device('npu',torch.npu.current_device()))

def validate(operation):
    import torch
    with torch.inference_mode(False):
        generator=torch.Generator().manual_seed(1011)
        wc=torch.randn(24,16384,generator=generator)*.01
        xc=torch.randn(33,4,4096,generator=generator).half().float()
        x=xc.to(operation.device);w=wc.to(operation.device)
        operation.prepare([w],[33])
        expected=xc.double().reshape(33,-1)@wc.half().double().T
        actual=operation(x,w).cpu().double()
        torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-4)
        operation.weights.clear()
    return dict(passed=True,rows=33,max_abs=float((actual-expected).abs().max()),
                scope='independent FP64 projection gate; loaded model weights gated before binding')
"""
    compile(source, "qualified mHC projection manifest", "exec")
    bridge = build / f"glm_reconstruction_bridge_v{options['version']}.so"
    if digest(bridge) != provenance["reconstruction_bridge.cpp"]["binary_sha256"]:
        raise ValueError("projection bridge differs from provenance")
    initializer = helper.with_name("__init__.py")
    if digest(initializer) != provenance["_helpers"][initializer.name]:
        raise ValueError("projection package initializer differs from provenance")
    return NativeManifest(
        dict(
            name=f"mhc_projection_v{options['version']}",
            libraries=[dict(path=str(bridge), sha256=digest(bridge))],
            assets=[
                dict(path=str(p), sha256=digest(p)) for p in (kernel, helper, initializer, build / "provenance.json")
            ],
            operators=[namespace + "::launch"],
            validation_source=source,
        )
    )
