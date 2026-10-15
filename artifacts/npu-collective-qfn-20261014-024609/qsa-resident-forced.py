"""Compare raw model scores across a boundary with identical forced history."""

import argparse
import json
import time
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token")
    fixture = json.loads(args.fixture.read_text())
    request = next(row["request"] for row in fixture["trials"] if row["label"] == "cross_boundary").copy()
    request.update(max_tokens=48, min_tokens=48, allowed_token_ids=[279], logprobs=5)
    assert not client.request("/is_paused", method="GET")["is_paused"]
    assert client.request("/pause?mode=wait&clear_cache=true")["status"] == "paused"
    # No code or graph changes; invalidate prefixes and request state uniformly.
    client.rpc("resident_reset")
    client.resume()
    started = time.monotonic()
    response = client.request("/v1/completions", request)
    elapsed = time.monotonic() - started
    choice = response["choices"][0]
    assert choice["token_ids"] == [279] * 48
    assert any(value < -0.01 for value in choice["logprobs"]["token_logprobs"])
    result = {"request": request, "response": response, "seconds": elapsed, "workers": client.rpc("resident_status")}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "seconds": elapsed, "tokens": len(choice["token_ids"])}))


if __name__ == "__main__":
    main()
