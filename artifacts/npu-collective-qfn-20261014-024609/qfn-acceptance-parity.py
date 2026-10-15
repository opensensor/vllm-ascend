"""Compare complete greedy token sequences with acceptance reuse off/on."""

import json
from pathlib import Path

from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")


def main():
    output = ROOT / "iteration-closure-v1/model-parity-v1.json"
    if output.exists():
        raise FileExistsError(output)
    client = QfnResidentClient("http://127.0.0.1:8001", ROOT / "resident-live-v4/control.token")
    result = dict(cases=[])
    base = client.request(
        "/tokenize",
        dict(
            model="qwen38-flash-next",
            messages=[dict(role="user", content="Explain how a hash table resolves collisions.")],
            chat_template_kwargs={"enable_thinking": False},
        ),
    )["tokens"]
    filler = client.request(
        "/tokenize",
        dict(model="qwen38-flash-next", prompt="A hash table stores keys in buckets. ", add_special_tokens=False),
    )["tokens"]
    for length in (len(base), 255, 2030, 2100):
        extra = length - len(base)
        prompt = base[:3] + (filler * ((extra + len(filler) - 1) // len(filler)))[:extra] + base[3:]
        case = dict(length=length, trials=[])
        for enabled in (False, True):
            client.request("/pause?mode=wait&clear_cache=false")
            try:
                ranks = client.rpc("qfn_cpu_profile_stop", enabled)
                assert all(r.get("passed") for r in ranks)
            finally:
                client.resume()
            response = client.request(
                "/v1/completions",
                dict(
                    model="qwen38-flash-next",
                    prompt=prompt,
                    temperature=0,
                    seed=42,
                    max_tokens=96,
                    min_tokens=96,
                    ignore_eos=True,
                    return_token_ids=True,
                ),
            )
            assert response["usage"]["completion_tokens"] == 96
            case["trials"].append(dict(enabled=enabled, response=response))
        a, b = [row["response"]["choices"][0] for row in case["trials"]]
        assert a["token_ids"] and len(a["token_ids"]) == 96
        case["exact_tokens"] = a["token_ids"] == b["token_ids"]
        case["exact_text"] = a["text"] == b["text"]
        result["cases"].append(case)
        output.write_text(json.dumps(result, indent=2) + "\n")
        assert case["exact_tokens"] and case["exact_text"]
        print(json.dumps({k: v for k, v in case.items() if k != "trials"}), flush=True)
    result["status"] = client.rpc("resident_status")
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
