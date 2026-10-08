"""Final bounded generation gate before handing the demo server to its user."""

import concurrent.futures
import json
import time
from pathlib import Path

import regex as re
from replay_gate import BASE, MODEL_DIR, ROOT, RUNTIME, chat, emit, save, smoke, switch
from tokenizers import Tokenizer

from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.qwen4exp.resident_reset import reset_caches
from tools.qwen38_decode_study.benchmark import PROMPTS, stream_completion
from tools.qwen38_decode_study.mixed_prefill_probe import prompt, stream


def main():
    client = ResidentClient(BASE, 4, 180)
    deadline = time.monotonic() + 360
    while True:
        try:
            models = client.request("/v1/models", method="GET")
            break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("Final demo startup timed out") from None
            time.sleep(2)
    log = (ROOT / "server-demo5.log").read_text(errors="replace")
    capacity = re.search(r"NPU KV cache size: ([\d,]+) tokens, Maximum concurrency.*?: ([\d.]+)x", log)
    reserve = re.search(r"post_capture_free=([\d.]+) GiB, reserve=([\d.]+) GiB", log)
    if not capacity or not reserve or float(reserve[1]) < float(reserve[2]):
        raise RuntimeError("Final cache capacity and runtime reserve not verified")
    model = models["data"][0]["id"]
    save("demo5-models", models)
    manifest = Path("/home/matteius/experiments/qwen38-decode-next-20261005/native-hc-residual/native-v1.json")
    save("demo5-native-load", client.load_native(NativeManifest(json.loads(manifest.read_text()))))
    source = (RUNTIME / "tools/qwen4exp/resident_candidates/native_hc_residual.py").read_text()
    switch(client, "demo5_native_residual", source)
    smoke(client, "demo5-selected", model)
    save("demo5-reset", reset_caches(client))
    serial = stream_completion(BASE, chat(model, PROMPTS[0], 128))
    save("demo5-serial128", serial)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(stream_completion, BASE, chat(model, PROMPTS[i % 3], 128)) for i in range(4)]
        results = [future.result() for future in futures]
    save("demo5-c4-128", {"results": results, "wall_s": time.perf_counter() - started})
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    save("demo5-cold-reset", reset_caches(client))
    cold = stream(BASE, model, prompt(tokenizer, 4096, "final-demo-chunked-prefill"), 32, 180)
    if cold["cached_tokens"] != 0:
        raise RuntimeError("Final chunked prefill was not cold")
    save("demo5-cold4096", cold)
    save("demo5-final-status", client.rpc("resident_status"))
    save("demo5-final-paused", client.request("/is_paused", method="GET"))
    save(
        "demo5-health",
        {
            "model": model,
            "completed": True,
            "graph_profile": [3, 12],
            "cache_tokens": int(capacity[1].replace(",", "")),
            "planner_concurrency_256k": float(capacity[2]),
            "post_capture_free_gib": float(reserve[1]),
            "runtime_reserve_gib": float(reserve[2]),
            "active_scheduler_limit": 4,
        },
    )
    emit("demo5_complete")


if __name__ == "__main__":
    main()
