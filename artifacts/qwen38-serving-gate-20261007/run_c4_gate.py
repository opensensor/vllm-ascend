"""Compare original and native HC on the separately launched 9/12 profile."""

import json
import time
from pathlib import Path

from replay_gate import BASE, RUNTIME, concurrent_run, emit, save, serial, smoke, switch

from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest


def main():
    client = ResidentClient(BASE, 4, 300)
    deadline = time.monotonic() + 1800
    while True:
        try:
            model_list = client.request("/v1/models", method="GET")
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError("C4 graph service did not become ready") from None
            time.sleep(5)
    model = model_list["data"][0]["id"]
    save("c4-models", model_list)
    save("c4-startup-status", client.rpc("resident_status"))
    smoke(client, "c4-original", model)
    serial(client, "c4-original", model, count=1)
    manifest_path = Path("/home/matteius/experiments/qwen38-decode-next-20261005/native-hc-residual/native-v1.json")
    manifest = NativeManifest(json.loads(manifest_path.read_text()))
    save("c4-native-load", client.load_native(manifest))
    residual = (RUNTIME / "tools/qwen4exp/resident_candidates/native_hc_residual.py").read_text()
    for repeat in range(3):
        for label, source in ((f"c4_original_{repeat}", ""), (f"c4_residual_{repeat}", residual)):
            switch(client, label if source else "baseline", source)
            # The engine publishes counters every ten seconds. Settle the
            # preceding workload outside measured latency on both arms.
            time.sleep(11)
            concurrent_run(client, label, model, 4)
    smoke(client, "c4-selected-residual", model)
    save("c4-final-status", client.rpc("resident_status"))
    save("c4-final-paused", client.request("/is_paused", method="GET"))
    emit("c4_complete")


if __name__ == "__main__":
    main()
