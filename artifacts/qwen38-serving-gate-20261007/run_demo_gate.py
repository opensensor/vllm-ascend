"""Finish a bounded demo gate without changing model weights or kernel packages."""

import concurrent.futures
import json
import time
from pathlib import Path

from replay_gate import BASE, MODEL_DIR, RUNTIME, cold, concurrent_run, emit, save, serial, smoke, switch
from tokenizers import Tokenizer

from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.qwen4exp.resident_reset import reset_caches
from tools.qwen38_decode_study.mixed_prefill_probe import prompt, stream


def main():
    client = ResidentClient(BASE, 4, 240)
    deadline = time.monotonic() + 480
    while True:
        try:
            models = client.request("/v1/models", method="GET")
            break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("Demo readiness exceeded eight minutes") from None
            time.sleep(3)
    model = models["data"][0]["id"]
    save("demo-models", models)
    save("demo-startup-status", client.rpc("resident_status"))
    emit("demo_ready", model=model)
    smoke(client, "demo-original", model)
    manifest_path = Path("/home/matteius/experiments/qwen38-decode-next-20261005/native-hc-residual/native-v1.json")
    save("demo-native-load", client.load_native(NativeManifest(json.loads(manifest_path.read_text()))))
    source = (RUNTIME / "tools/qwen4exp/resident_candidates/native_hc_residual.py").read_text()
    switch(client, "demo_native_residual", source)
    serial(client, "demo-selected", model, count=1)
    concurrent_run(client, "demo-selected-0", model, 4)
    concurrent_run(client, "demo-selected-1", model, 4)
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    cold(client, "demo-selected", model, tokenizer, 23410, warm=True)
    save("demo-windows-reset", reset_caches(client))
    prompts = [prompt(tokenizer, 1024, f"demo-independent-window-{i}") for i in range(6)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(stream, BASE, model, ids, 64, 180) for ids in prompts]
        results = [future.result() for future in futures]
    save("demo-six-windows", results)
    resumed = [stream(BASE, model, ids, 32, 180) for ids in prompts]
    save("demo-six-windows-resumed", resumed)
    emit("demo_windows", sessions=len(results), cached_on_resume=[row["cached_tokens"] for row in resumed])
    save("demo-final-status", client.rpc("resident_status"))
    save("demo-final-paused", client.request("/is_paused", method="GET"))
    save("demo-health", {"model": model, "completed": True, "graph_profile": [3, 12], "active_scheduler_limit": 4})
    emit("demo_complete")


if __name__ == "__main__":
    main()
