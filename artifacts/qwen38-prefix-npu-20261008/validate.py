#!/usr/bin/env python3
"""Run isolated real-weight checks; never target a shared serving port."""

import argparse
import base64
import concurrent.futures
import io
import json
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer

from tools.glm_perf.resident_harness import ResidentClient
from tools.qwen4exp.resident_reset import reset_caches
from tools.qwen38_decode_study.mixed_prefill_probe import prompt, stream

ROOT = Path("/home/matteius/experiments/qwen38-prefix-npu-20261008")
MODEL_ROOT = Path("/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i")
MODEL = "qwen38-prefix-bounded-validation"


def temperatures():
    output = subprocess.check_output(["npu-smi", "info"], text=True, timeout=10)
    return [int(line.split("NA")[1].split()[0]) for line in output.splitlines() if "310P3" in line]


def cool():
    readings = []
    for _ in range(60):
        temps = temperatures()
        readings.append({"time": time.time(), "temperatures_c": temps})
        if max(temps) < 84:
            return readings
        print(json.dumps({"cooling": temps}), flush=True)
        time.sleep(10)
    raise RuntimeError("NPUs did not cool below 84C; no new work submitted")


def pressure(client):
    client.request("/pause?mode=wait&clear_cache=true")
    results = {}
    try:
        for policy in ("baseline", "bounded"):
            results[policy + "_cooling"] = cool()
            results[policy + "_reset"] = reset_caches(client)
            results[policy] = client.rpc("resident_prefix_pressure", policy)
            save("pressure-partial", results)
        results["final_reset"] = reset_caches(client)
    finally:
        client.resume()
    return results


def chat(client, messages, max_tokens=128):
    return client.request(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "temperature": 0,
            "seed": 42,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": messages,
        },
    )


def vision(client):
    results = []
    for label in ("red", "blue", "ocr"):
        cooling = cool()
        image = Image.new("RGB", (256, 256), "white" if label == "ocr" else label)
        if label == "ocr":
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
            draw.text((20, 95), "DEMO42", fill="black", font=font)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        question = (
            "Read the text in the image. Reply with only the text."
            if label == "ocr"
            else "What is the image color? Reply with only the color name."
        )
        response = chat(
            client,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": url}}],
                }
            ],
        )
        actual = (response["choices"][0]["message"].get("content") or "").strip()
        expected = "DEMO42" if label == "ocr" else label
        results.append(
            {
                "label": label,
                "cooling": cooling,
                "response": response,
                "pass": actual.casefold() == expected.casefold(),
                "temperatures_c": temperatures(),
            }
        )
        save("vision-partial", results)
    return results


def sessions(client):
    reset = reset_caches(client)
    tokenizer = Tokenizer.from_file(str(MODEL_ROOT / "tokenizer.json"))
    results = {"reset": reset, "batches": [], "initial_status": client.rpc("resident_status")}
    # Three independent 16K histories exceed the per-group 27-checkpoint budget.
    for length, count, outputs in ((256, 1, 256), (16384, 3, 128), (16384, 3, 128)):
        cooling = cool()
        tokens = [prompt(tokenizer, length, f"bounded-npu-20261008-{length}-{i}") for i in range(count)]
        began = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(stream, client.base_url, MODEL, item, outputs, 900) for item in tokens]
            responses = [future.result() for future in futures]
        elapsed = time.monotonic() - began
        results["batches"].append(
            {
                "length": length,
                "concurrency": count,
                "cooling": cooling,
                "responses": responses,
                "wall_s": elapsed,
                "aggregate_end_to_end_tps": sum(r["completion_tokens"] for r in responses) / elapsed,
                "status": client.rpc("resident_status"),
                "temperatures_c": temperatures(),
            }
        )
        save("sessions-partial", results)
    return results


def mixed(client):
    cooling = cool()
    reset = reset_caches(client)
    tokenizer = Tokenizer.from_file(str(MODEL_ROOT / "tokenizer.json"))
    tokens = prompt(tokenizer, 256, "bounded-mixed-image-20261008")
    ready = threading.Event()
    image = Image.new("RGB", (256, 256), "blue")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        decoding = pool.submit(stream, client.base_url, MODEL, tokens, 256, 900, ready)
        if not ready.wait(timeout=60):
            raise RuntimeError("Decode request did not start")
        response = chat(
            client,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is the image? Reply with only the color."},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
        )
        decode = decoding.result()
    actual = (response["choices"][0]["message"].get("content") or "").strip()
    return {
        "cooling": cooling,
        "reset": reset,
        "decode": decode,
        "image": response,
        "image_pass": actual.casefold() == "blue",
        "status": client.rpc("resident_status"),
        "temperatures_c": temperatures(),
    }


def fast_prefill(client):
    cooling = cool()
    reset = reset_caches(client)
    tokenizer = Tokenizer.from_file(str(MODEL_ROOT / "tokenizer.json"))
    tokens = prompt(tokenizer, 23410, "qwen-latest-matched-23410-v1")
    results = {"cooling": cooling, "reset": reset, "initial_status": client.rpc("resident_status")}
    results["cold"] = stream(client.base_url, MODEL, tokens, 32, 900)
    save("fast-prefill-partial", results)
    results["repeat_cooling"] = cool()
    results["repeat"] = stream(client.base_url, MODEL, tokens, 32, 900)
    results["final_status"] = client.rpc("resident_status")
    results["cold_cached_zero"] = results["cold"]["cached_tokens"] == 0
    results["repeat_prefix_hit"] = results["repeat"]["cached_tokens"] > 0
    results["temperatures_c"] = temperatures()
    return results


def save(name, result):
    (ROOT / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pressure", "vision", "sessions", "mixed", "fast-prefill", "status"))
    args = parser.parse_args()
    client = ResidentClient("http://127.0.0.1:8001", 4, 900)
    result = {
        "pressure": pressure,
        "vision": vision,
        "sessions": sessions,
        "mixed": mixed,
        "fast-prefill": fast_prefill,
        "status": lambda client: client.rpc("resident_status"),
    }[args.action](client)
    save(args.action, result)
    print(
        json.dumps(
            {"action": args.action, "output": str(ROOT / (args.action + ".json")), "temperatures_c": temperatures()}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
