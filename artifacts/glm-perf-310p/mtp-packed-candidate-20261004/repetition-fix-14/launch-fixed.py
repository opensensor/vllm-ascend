"""Start the validated graph configuration with normal serving endpoints."""

import subprocess
from pathlib import Path

root = Path("/home/matteius/experiments/glm-w3-20261004/repetition-fix-14")
command = [
    "bash",
    str(root / "serve-mtp-fixed.sh"),
    "/srv/ai/src/glm-selective-w3-nz-test-20261004",
    "/srv/ai/models/GLM-5.3-Flash-selective-W3-310p",
    "32768",
    "0.70",
    "/srv/ai/src/glm-l1-wide-build-20261004/opp-l1-overlap-20261004",
    "4",
    "graph",
    "640",
    "histogram",
    "",
    "/srv/ai/src/build-only-glm-w3-nz-csrc-20261004/opp-qsa-cube512-candidate",
    "on",
    "off",
    "off",
    "off",
    "",
    "1",
]
with (root / "serve-fixed.log").open("wb") as log:
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
(root / "server.pid").write_text(str(process.pid) + "\n")
print(process.pid)
