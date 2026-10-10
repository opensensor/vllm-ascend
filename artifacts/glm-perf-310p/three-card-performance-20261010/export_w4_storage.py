"""Export the admitted lossless variant locally; never modify the source."""

import json
from pathlib import Path

from tools.glm_perf.w4_storage_checkpoint import export

root = Path("/srv/ai/storage/artifacts/glm-card-profiles-20261010")
source = Path("/srv/ai/storage/models/GLM-5.3-Flash-three-card-native-candidate")
output = Path("/srv/ai/storage/models/GLM-5.3-Flash-three-card-W3-in-W4-native-candidate")
proposal = json.loads((root / "w4-storage-pipeline-plan.json").read_text())
assert proposal["fits_allowance"] and proposal["additional_bytes_per_rank"] == [
    2717908992,
    2717908992,
    905969664,
    905969664,
    5435817984,
    5435817984,
]
banks = [row["bank"] for row in proposal["banks"] if row["selected"]]
memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
assert int(memory["MemAvailable"].split()[0]) >= 32 * 1024**2, "Export requires 32 GiB available host RAM"
result = export(
    source,
    output,
    banks,
    extra_bytes_per_rank=6 * 1024**3,
    self_contained=True,
    workers=6,
    resume=True,
)
(root / "w4-storage-export-receipt.json").write_text(
    json.dumps(
        {
            "complete": result["complete"],
            "self_contained": result["self_contained_storage"],
            "output": str(output),
            "banks": banks,
            "added_bytes_per_rank": proposal["additional_bytes_per_rank"],
        },
        indent=2,
    )
    + "\n"
)
print("GLM_LOSSLESS_W4_STORAGE_EXPORT_COMPLETE", flush=True)
