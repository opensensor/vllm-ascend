"""Requalify the prior TP6 HC fusion against this service's loaded weights."""

import hashlib
import json
import sys
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_native import NativeManifest
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
RUN = ROOT / sys.argv[1]
assert RUN.parent == ROOT and RUN.name.startswith("resident-")
client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", expected_workers=6, timeout=1800)
manifest = json.loads(Path("/srv/ai/src/qwen-performance-evidence-20261010/mtp-v2-hc-native-manifest.json").read_text())
for asset in manifest["assets"]:
    old = "/srv/ai/src/qwen-performance-mtp-tp6-v2-20261010-8fce905/"
    if asset["path"].startswith(old):
        asset["path"] = str(ROOT / "runtime" / asset["path"][len(old) :])
        asset["sha256"] = hashlib.sha256(Path(asset["path"]).read_bytes()).hexdigest()
manifest["name"] = "hc_residual_v3"
for name in ("exact_hc_residual.py", "shared_hc_operand.py", "release_hc_projection.py"):
    asset = ROOT / "runtime/tools/qwen4exp/resident_candidates" / name
    manifest["assets"].append(dict(path=str(asset), sha256=hashlib.sha256(asset.read_bytes()).hexdigest()))
manifest["validation_source"] += """
component_validate = validate
def validate(op):
    import gc
    import torch
    from tools.qwen4exp.resident_candidates import exact_hc_residual
    from vllm_ascend.models.qwen4_exp.model import _GatedResidual
    report = component_validate(op)
    workers = [obj for obj in gc.get_objects() if type(obj).__name__ == 'NPUWorker310']
    assert len(workers) == 1
    combine = exact_hc_residual.make_combine(op)
    count = 0
    with torch.inference_mode():
        for wrapper in workers[0]._resident_wrappers():
            for module in wrapper.runnable.modules():
                if not isinstance(module, _GatedResidual) or not module.use_combine:
                    continue
                for rows in (3, 6, 12, 18, 256):
                    generator = torch.Generator().manual_seed(1055 + count)
                    hyper = torch.randn(rows, module.hyper_hidden, generator=generator).half().npu()
                    block = torch.randn(rows, module.hidden_size, generator=generator).half().npu()
                    expected_mix, residual = _GatedResidual.mix(module, hyper)
                    expected = _GatedResidual.combine(module, block, residual)
                    actual_mix, residual = exact_hc_residual.mix(module, hyper)
                    actual = combine(module, block, residual)
                    torch.testing.assert_close(actual_mix.cpu(), expected_mix.cpu(), rtol=0, atol=0)
                    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
                    count += 1
    assert count > 0
    return dict(report, real_weight_cases=count, rows=[3, 6, 12, 18, 256], bitwise_equal=True)
"""
(RUN / "hc-exact-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
result = client.load_native(NativeManifest(manifest))
(RUN / "hc-exact-gate.json").write_text(json.dumps(result, indent=2) + "\n")
source = (ROOT / "runtime/tools/qwen4exp/resident_candidates/exact_hc_residual.py").read_text()
result = client.switch(Control(uuid.uuid4().hex, candidate="tp6_exact_native_hc", source=source, recapture=True))
(RUN / "hc-exact-installed.json").write_text(json.dumps(result, indent=2) + "\n")
print("HC fusion qualified and installed on all six ranks", flush=True)
