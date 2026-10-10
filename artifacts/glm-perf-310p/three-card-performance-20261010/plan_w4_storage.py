"""Header-only admission for the lossless three-card storage variant."""

import json
from pathlib import Path

from tools.glm_perf.w4_storage_checkpoint import plan

root = Path("/srv/ai/storage/artifacts/glm-card-profiles-20261010")
source = Path("/srv/ai/storage/models/GLM-5.3-Flash-three-card-native-candidate")
inventory = plan(source, extra_bytes_per_rank=0)
banks = [row["bank"] for row in inventory["banks"] if row["eligible"]]
proposal = plan(source, banks, extra_bytes_per_rank=6 * 1024**3)
assert proposal["fits_allowance"] and proposal["resident_world_size"] == 6
(root / "w4-storage-pipeline-plan.json").write_text(json.dumps(proposal, indent=2) + "\n")
print(
    json.dumps(
        {
            k: proposal[k]
            for k in ("additional_bytes_per_rank", "fits_allowance", "new_payload_bytes", "precision_changed")
        },
        indent=2,
    )
)
