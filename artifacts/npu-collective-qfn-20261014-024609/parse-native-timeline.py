"""Parse recorded traces offline without touching serving devices."""

from pathlib import Path

from tools.qwen4exp.profile_runtime import analyse_npu

root = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/qsa-resident-v1/native-timeline-v1")
for tag in ("decode", "cold-prefill"):
    path = root / tag / "rank0"
    print(analyse_npu(path, processes=1), flush=True)
