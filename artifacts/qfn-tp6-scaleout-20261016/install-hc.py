"""Requalify the prior TP6 HC fusion against this service's loaded weights."""

import hashlib
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_native import NativeManifest
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
RUN = ROOT / "resident-v3"
client = QfnResidentClient("http://127.0.0.1:8001", RUN / "control.token", expected_workers=6, timeout=1800)
manifest = json.loads(Path("/srv/ai/src/qwen-performance-evidence-20261010/mtp-v2-hc-native-manifest.json").read_text())
for asset in manifest["assets"]:
    old = "/srv/ai/src/qwen-performance-mtp-tp6-v2-20261010-8fce905/"
    if asset["path"].startswith(old):
        asset["path"] = str(ROOT / "runtime" / asset["path"][len(old) :])
        asset["sha256"] = hashlib.sha256(Path(asset["path"]).read_bytes()).hexdigest()
manifest["validation_source"] += """
component_validate = validate
def validate(op):
    import gc
    import torch
    from tools.qwen4exp.resident_candidates import fused_hc_projection, fused_hc_residual
    from vllm_ascend.models.qwen4_exp.model import _GatedResidual
    report = component_validate(op)
    workers = [obj for obj in gc.get_objects() if type(obj).__name__ == 'NPUWorker310']
    assert len(workers) == 1
    combine = fused_hc_residual.make_combine(op)
    count = 0
    with torch.inference_mode():
        for wrapper in workers[0]._resident_wrappers():
            for module in wrapper.runnable.modules():
                if not isinstance(module, _GatedResidual) or not module.use_combine:
                    continue
                for rows in (3, 6, 12, 18):
                    generator = torch.Generator().manual_seed(1055 + count)
                    hyper = torch.randn(rows, module.hyper_hidden, generator=generator).half().npu()
                    block = torch.randn(rows, module.hidden_size, generator=generator).half().npu()
                    expected_mix, residual = _GatedResidual.mix(module, hyper)
                    expected = _GatedResidual.combine(module, block, residual)
                    actual_mix, residual = fused_hc_projection.mix(module, hyper)
                    assert isinstance(residual, fused_hc_projection.ProjectionResidual)
                    actual = combine(module, block, residual)
                    torch.testing.assert_close(actual_mix.cpu(), expected_mix.cpu(), rtol=0, atol=0)
                    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
                    count += 1
    assert count > 0
    return dict(report, real_weight_cases=count, rows=[3, 6, 12, 18], bitwise_equal=True)
"""
(ROOT / "hc-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
result = client.load_native(NativeManifest(manifest))
(ROOT / "hc-gate.json").write_text(json.dumps(result, indent=2) + "\n")
source = (ROOT / "runtime/tools/qwen4exp/resident_candidates/fused_hc_residual.py").read_text()
result = client.switch(Control(uuid.uuid4().hex, candidate="tp6_fused_native_hc", source=source, recapture=True))
(ROOT / "hc-installed.json").write_text(json.dumps(result, indent=2) + "\n")
print("HC fusion qualified and installed on all six ranks", flush=True)
