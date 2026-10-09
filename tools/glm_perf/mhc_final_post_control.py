# SPDX-License-Identifier: Apache-2.0
"""Admit the complete final mixer only after exact-build arithmetic/replay gates."""

import json

from tools.glm_perf.resident_native import NativeManifest, file_digest


def manifest(build, gates):
    report = json.loads(gates.read_text())
    provenance = json.loads((build / "mhc-provenance.json").read_text())
    bridge_provenance = json.loads((build / "provenance.json").read_text())
    namespace = provenance["namespace"]
    version = provenance["version"]
    cases = report.get("cases", [])
    if (
        report.get("complete") is not True
        or {(row.get("rows"), row.get("input_dtype")) for row in cases}
        != {(rows, dtype) for rows in (640, 1280) for dtype in ("torch.float16", "torch.float32")}
        or not all(
            row.get("finite") is True
            and row.get("graph_changed_inputs") is True
            and row.get("fp32_output_unrounded") is True
            for row in cases
        )
        or report.get("provenance") != provenance
        or provenance.get("state_rounding") != "none_fp32"
        or provenance.get("finish_only") is not False
    ):
        raise ValueError("final mixer requires matching complete FP32 arithmetic and changed-input replay gates")
    helper = build / "mhc_post_native.py"
    bridge = build / f"glm_reconstruction_bridge_v{version}.so"
    if (
        file_digest(helper) != provenance["helper_sha256"]
        or report.get("helper_sha256") != provenance["helper_sha256"]
        or bridge_provenance["_build"]["namespace"] != namespace
        or file_digest(bridge) != bridge_provenance["reconstruction_bridge.cpp"]["binary_sha256"]
    ):
        raise ValueError("final mixer helper or bridge differs from qualified provenance")
    binaries = [build / f"mhc_post_fp{bits}.bin" for bits in (16, 32)]
    if any(file_digest(path) != provenance["binaries"][path.name]["sha256"] for path in binaries):
        raise ValueError("final mixer binary differs from qualified provenance")
    validation = f"""def prepare():
 import importlib.util
 from pathlib import Path
 build=Path({str(build)!r})
 spec=importlib.util.spec_from_file_location('frozen_final_post_v{version}',build/'mhc_post_native.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.NativeMhcFinalPost(build,{namespace!r})

def validate(operation):
 import torch
 with torch.inference_mode(False):
  generator=torch.Generator().manual_seed({version})
  cpu=(torch.randn(640,4096,generator=generator),
       torch.randn(640,4,4096,generator=generator).half().float(),
       torch.randn(640,4,1,generator=generator).sigmoid().half().float(),
       torch.randn(640,4,4,generator=generator).softmax(-1).half().float())
  device=torch.device('npu',torch.npu.current_device())
  values=tuple(value.to(device) for value in cpu)
  actual=operation(*values).cpu()
  expected=torch.einsum('nij,nih->njh',cpu[3].double(),cpu[1].double())+cpu[2].double()*cpu[0].double().unsqueeze(1)
  torch.testing.assert_close(actual.double(),expected,rtol=2e-5,atol=2e-5)
  assert actual.dtype==torch.float32 and bool(torch.isfinite(actual).all())
  assert bool((actual!=actual.half().float()).any()),'final output was rounded'
  delta=float((actual.double()-expected).abs().max())
 return dict(passed=True,executed=True,max_abs_fp64=delta,
             scope='complete unrounded FP32 final mixer; operator gate, not model quality')
"""
    compile(validation, "final mixer admission", "exec")

    def entry(path):
        return {"path": str(path), "sha256": file_digest(path)}

    return NativeManifest(
        {
            "name": f"mhc_final_post_v{version}",
            "libraries": [entry(bridge)],
            "assets": [
                entry(path) for path in [helper, *binaries, build / "mhc-provenance.json", build / "provenance.json"]
            ],
            "operators": [f"{namespace}::launch"],
            "validation_source": validation,
        }
    )
