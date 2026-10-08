"""Validate the smaller image-enabled demo and checkpoint reuse on real weights."""

import base64
import hashlib
import io
import json
import time
from pathlib import Path

import regex as re
from PIL import Image, ImageDraw, ImageFont
from replay_gate import BASE, MODEL_DIR, ROOT, RUNTIME, concurrent_run, emit, save, serial, smoke, switch
from tokenizers import Tokenizer

from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.qwen4exp.resident_reset import reset_caches
from tools.qwen38_decode_study.mixed_prefill_probe import prompt, stream


def image_checks(client, model):
    results = []
    for label in ("red", "blue", "ocr"):
        picture = Image.new("RGB", (256, 256), "white" if label == "ocr" else label)
        question = "What color fills this image? Reply with only the color name."
        expected = label
        if label == "ocr":
            draw = ImageDraw.Draw(picture)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            draw.text((40, 50), "DEMO", fill="black", font=font)
            draw.text((90, 120), "42", fill="black", font=font)
            question = "Read the text in this image. Reply with only the text."
            expected = "DEMO 42"
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        png = buffer.getvalue()
        (ROOT / f"vision-{label}.png").write_bytes(png)
        started = time.perf_counter()
        response = client.request(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        actual = response["choices"][0]["message"].get("content") or ""
        passed = expected.lower() in " ".join(actual.lower().split())
        results.append(
            {
                "case": label,
                "expected": expected,
                "pass": passed,
                "image_sha256": hashlib.sha256(png).hexdigest(),
                "elapsed_s": time.perf_counter() - started,
                "response": response,
            }
        )
        save("vision-images", results)
        emit("vision_image", case=label, passed=passed, actual=actual)
    if not all(row["pass"] for row in results):
        raise RuntimeError("Image requests completed but semantic image gate failed")


def main():
    client = ResidentClient(BASE, 4, 360)
    deadline = time.monotonic() + 900
    while True:
        try:
            models = client.request("/v1/models", method="GET")
            break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("Image-enabled startup exceeded fifteen minutes") from None
            time.sleep(3)
    model = models["data"][0]["id"]
    save("vision-models", models)
    save("vision-startup-status", client.rpc("resident_status"))
    emit("vision_ready", model=model)
    image_checks(client, model)
    manifest = Path("/home/matteius/experiments/qwen38-decode-next-20261005/native-hc-residual/native-v1.json")
    save("vision-native-load", client.load_native(NativeManifest(json.loads(manifest.read_text()))))
    source = (RUNTIME / "tools/qwen4exp/resident_candidates/native_hc_residual.py").read_text()
    switch(client, "vision_native_residual", source)
    image_checks(client, model)
    smoke(client, "vision-selected", model)
    serial(client, "vision-selected", model, count=1)
    concurrent_run(client, "vision-selected", model, 4)

    # More retained history than the previous profile's smallest device tier:
    # three independent 32K prefixes, then reuse each without a cache reset.
    # This is a bounded checkpoint-pressure gate, not 3 x 256K qualification.
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    save("vision-history-reset", reset_caches(client))
    before = client.rpc("resident_status")
    prompts = [prompt(tokenizer, 32768, f"vision-history-{i}") for i in range(3)]
    cold = []
    for index, ids in enumerate(prompts):
        cold.append(stream(BASE, model, ids, 32, 360))
        save("vision-history-cold", cold)
        emit("vision_history_cold", session=index, ttft_s=cold[-1]["time_to_first_token_s"])
    warm = []
    for index, ids in enumerate(prompts):
        warm.append(stream(BASE, model, ids, 32, 360))
        save("vision-history-warm", warm)
        emit("vision_history_warm", session=index, cached_tokens=warm[-1]["cached_tokens"])
    after = client.rpc("resident_status")
    save("vision-history-status", {"before": before, "after": after})
    if any(row["cached_tokens"] != 0 for row in cold) or any(not row["cached_tokens"] for row in warm):
        raise RuntimeError("Cold/warm checkpoint history gate did not prove prefix reuse")
    deltas = []
    for old, new in zip(sorted(before, key=lambda r: r["rank"]), sorted(after, key=lambda r: r["rank"]), strict=True):
        for group, tier in new["prefix_mamba"].items():
            deltas.append(
                {
                    "rank": new["rank"],
                    "group": group,
                    "spill_delta": tier["spill_count"] - old["prefix_mamba"][group]["spill_count"],
                    "restore_delta": tier["restore_count"] - old["prefix_mamba"][group]["restore_count"],
                    "archive_slots": tier["archive_slots"],
                }
            )
    save("vision-history-transfers", deltas)
    log = (ROOT / "server-vision.log").read_text(errors="replace")
    capacity = re.search(r"NPU KV cache size: ([\d,]+) tokens, Maximum concurrency.*?: ([\d.]+)x", log)
    if not capacity or not 3 <= float(capacity[2]) <= 4.1:
        raise RuntimeError("Image-enabled planner capacity outside intended three/four-window target")
    save("vision-final-status", after)
    save("vision-final-paused", client.request("/is_paused", method="GET"))
    save(
        "vision-health",
        {
            "completed": True,
            "model": model,
            "images_verified": True,
            "graph_profile": [3, 12],
            "cache_tokens": int(capacity[1].replace(",", "")),
            "planner_concurrency_256k": float(capacity[2]),
            "active_scheduler_limit": 4,
            "history_test_tokens": 3 * 32768,
            "history_spill_delta": sum(row["spill_delta"] for row in deltas),
            "history_restore_delta": sum(row["restore_delta"] for row in deltas),
        },
    )
    emit("vision_complete")


if __name__ == "__main__":
    main()
