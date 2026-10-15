"""Exercise queued rejection, limits, penalties, structured output and aborts."""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v1")
    args = parser.parse_args()
    if Path(args.label).name != args.label:
        raise ValueError("plain label required")
    output = ROOT / "resident-live-v9/requalification" / f"scheduling-serving-gate-{args.label}.json"
    if output.exists():
        raise FileExistsError(output)
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v9/control.token")
    result = dict(before=client.rpc("resident_status"), trials=[])

    def save(row):
        result["trials"].append(row)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items() if key != "response"}), flush=True)

    def generate(label, tokens, **extra):
        payload = dict(
            model="qwen38-flash-next",
            messages=[dict(role="user", content="Explain hash table collisions.")],
            chat_template_kwargs=dict(enable_thinking=False),
            temperature=0,
            max_tokens=tokens,
            min_tokens=tokens,
            ignore_eos=True,
        )
        payload.update(extra)
        started = time.monotonic()
        response = client.request("/v1/chat/completions", payload)
        assert response["usage"]["completion_tokens"] == tokens
        return dict(label=label, seconds=time.monotonic() - started, response=response, passed=True)

    for tokens in (1, 2, 3):
        save(generate(f"budget-{tokens}", tokens))
    save(generate("penalties", 64, frequency_penalty=0.2, presence_penalty=0.2, repetition_penalty=1.05))
    row = generate("logprobs", 16, logprobs=True, top_logprobs=1)
    assert row["response"]["choices"][0]["logprobs"]["content"]
    save(row)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(generate, f"turnover-{index}", tokens) for index, tokens in enumerate((1, 64, 128, 2))
        ]
        for future in futures:
            save(future.result())
    payload = dict(
        model="qwen38-flash-next",
        messages=[dict(role="user", content="Return the JSON object with answer equal to 2.")],
        chat_template_kwargs=dict(enable_thinking=False),
        temperature=0,
        max_tokens=64,
        response_format=dict(
            type="json_schema",
            json_schema=dict(
                name="answer",
                strict=True,
                schema=dict(
                    type="object",
                    properties=dict(answer=dict(type="integer", enum=[2])),
                    required=["answer"],
                    additionalProperties=False,
                ),
            ),
        ),
    )
    response = client.request("/v1/chat/completions", payload)
    assert json.loads(response["choices"][0]["message"]["content"]) == {"answer": 2}
    save(dict(label="structured-json", response=response, passed=True))

    payload = dict(
        model="qwen38-flash-next",
        messages=[dict(role="user", content="Explain hash tables in detail.")],
        chat_template_kwargs=dict(enable_thinking=False),
        max_tokens=512,
        min_tokens=512,
        ignore_eos=True,
        temperature=0,
        stream=True,
    )
    request = urllib.request.Request(
        client.base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=90) as stream:
        for line in stream:
            if not line.startswith(b"data: {"):
                continue
            chunk = json.loads(line[len(b"data: ") :])
            if any(choice.get("delta", {}).get("content") for choice in chunk.get("choices", ())):
                break
        else:
            raise RuntimeError("stream never produced content before abort")
    save(dict(label="abort-after-first-content", passed=True))
    save(generate("after-abort", 64))
    client.request("/pause?mode=wait&clear_cache=false")
    result["queue"] = client.rpc("qfn_cpu_profile_stop", json.dumps(dict(queued_schedule=True, status=True)))
    result["observer"] = client.rpc("qfn_cpu_profile_stop", json.dumps(dict(output_observer=True, status=True)))
    client.resume()
    result["after"] = client.rpc("resident_status")
    assert all(not row["native_failed"] and not row["graphs_dirty"] for row in result["after"])
    assert all(not row["pending_device_inputs"] and not row["pending_device_history"] for row in result["queue"])
    assert all(not row["failed"] and not row["pending"] for row in result["observer"])
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
