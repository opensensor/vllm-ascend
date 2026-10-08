#!/usr/bin/env python3
"""Load and select the qualified HC residual on drained resident workers."""

import json
import time
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest

ROOT = Path("/home/matteius/experiments/qwen38-prefix-npu-20261008")
RUNTIME = Path("/srv/ai/src/qwen38-prefix-bounded-runtime-20261008")
MANIFEST = Path("/home/matteius/experiments/qwen38-decode-next-20261005/native-hc-residual/native-v1.json")


def save(name, value):
    (ROOT / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


def main():
    client = ResidentClient("http://127.0.0.1:8001", 4, 900)
    deadline = time.monotonic() + 900
    while True:
        try:
            models = client.request("/v1/models", method="GET")
            break
        except OSError:
            if time.monotonic() > deadline:
                raise RuntimeError("Faster image profile did not become ready") from None
            time.sleep(3)
    save("fast-image-models", models)
    save("fast-image-before-hc", client.rpc("resident_status"))
    manifest = NativeManifest(json.loads(MANIFEST.read_text()))
    save("fast-image-native-load", client.load_native(manifest))
    control = Control.from_dict(
        {
            "generation": uuid.uuid4().hex,
            "mode": "graph",
            "candidate": "native_hc_residual",
            "recapture": True,
            "source": (RUNTIME / "tools/qwen4exp/resident_candidates/native_hc_residual.py").read_text(),
        }
    )
    save("fast-image-hc-switch", client.switch(control))
    save("fast-image-final-status", client.rpc("resident_status"))
    save("fast-image-paused", client.request("/is_paused", method="GET"))
    print(json.dumps({"native_hc_selected": True, "model": models["data"][0]["id"]}), flush=True)


if __name__ == "__main__":
    main()
