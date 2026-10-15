"""Collect bounded resident decode and cold-prefill device traces."""

import json
import time
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
OUTPUT = ROOT / "qsa-resident-v1/native-timeline-decode-geometry-v1"


def main():
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v3/control.token")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = {"before": client.rpc("resident_status"), "trials": []}
    for tag, delay, text, budget in (
        ("decode", 8, "Explain how a hash table works and discuss collision resolution strategies.", 128),
    ):
        client.request("/pause?mode=wait&clear_cache=false")
        row = {"tag": tag}
        try:
            row["start"] = client.rpc(
                "qfn_cpu_profile_start", json.dumps(dict(output_dir=str(OUTPUT), tag=tag, delay=delay, max=8))
            )
        finally:
            client.request("/resume")
        try:
            started = time.monotonic()
            row["response"] = client.request(
                "/v1/chat/completions",
                dict(
                    model="qwen38-flash-next",
                    messages=[dict(role="user", content=text)],
                    chat_template_kwargs={"enable_thinking": False},
                    temperature=0,
                    max_tokens=budget,
                    min_tokens=budget,
                    ignore_eos=True,
                ),
            )
            row["seconds"] = time.monotonic() - started
        finally:
            client.request("/pause?mode=wait&clear_cache=false")
            try:
                row["stop"] = client.rpc("qfn_cpu_profile_stop")
                assert all(item.get("ownership_released") and item.get("recorded") for item in row["stop"]), row["stop"]
            finally:
                client.request("/resume")
        records["trials"].append(row)
        (OUTPUT / "receipt.json").write_text(json.dumps(records, indent=2) + "\n")
        print(
            json.dumps(dict(tag=tag, seconds=row["seconds"], usage=row["response"]["usage"], stopped=row["stop"])),
            flush=True,
        )
    records["after"] = client.rpc("resident_status")
    (OUTPUT / "receipt.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
