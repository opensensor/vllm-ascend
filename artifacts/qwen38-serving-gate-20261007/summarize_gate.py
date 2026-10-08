"""Derive serving comparisons from retained HTTP evidence and server timings."""

import json
from pathlib import Path
from statistics import median

import regex as re

ROOT = Path(__file__).resolve().parent
TIMING = re.compile(r"request (\S+) \([^)]*\):.*?decode (\d+) tokens, (\d+) gaps in ([\d.]+) ms")


def read(name):
    return json.loads((ROOT / f"{name}.json").read_text())


def timings(filename):
    return {
        request: {"tokens": int(tokens), "decode_tok_s": int(gaps) * 1000 / float(elapsed)}
        for request, tokens, gaps, elapsed in TIMING.findall((ROOT / filename).read_text(errors="replace"))
        if float(elapsed) > 0
    }


def summarize():
    server = timings("server.log")
    arms = {}
    for label in ("residual_a1", "gdn_b1", "residual_a2", "gdn_b2"):
        serial = read(label + "-serial")
        for row in serial:
            timing = server[row["request_id"]]
            if timing["tokens"] != row["usage"]["completion_tokens"]:
                raise ValueError("Server and client disagree on completion length")
            row["server_decode_tok_s"] = timing["decode_tok_s"]
        arms[label] = {
            "serial": serial,
            "serial_server_median_tok_s": median(row["server_decode_tok_s"] for row in serial),
            "c2_aggregate_tok_s": read(label + "-c2")["aggregate_tok_s"],
            "c4_eager_aggregate_tok_s": read(label + "-c4")["aggregate_tok_s"],
            "cold8192": read(label + "-cold8192"),
            "cold23410": read(label + "-cold23410"),
        }
    pairs = []
    for control, candidate in (("residual_a1", "gdn_b1"), ("residual_a2", "gdn_b2")):
        for before, after in zip(arms[control]["serial"], arms[candidate]["serial"], strict=True):
            pairs.append(
                {
                    "control": control,
                    "candidate": candidate,
                    "prompt_index": before["prompt_index"],
                    "baseline_tok_s": before["server_decode_tok_s"],
                    "candidate_tok_s": after["server_decode_tok_s"],
                    "speed_ratio": after["server_decode_tok_s"] / before["server_decode_tok_s"],
                    "same_text": before["text_sha256"] == after["text_sha256"],
                    "baseline_sampled_acceptance": before["acceptance"],
                    "candidate_sampled_acceptance": after["acceptance"],
                }
            )
    same_text = [pair for pair in pairs if pair["same_text"]]
    result = {
        "arms": arms,
        "serial_pairs": pairs,
        "serial_pooled": {
            "baseline_median_tok_s": median(pair["baseline_tok_s"] for pair in pairs),
            "candidate_median_tok_s": median(pair["candidate_tok_s"] for pair in pairs),
            "median_paired_speed_ratio": median(pair["speed_ratio"] for pair in pairs),
            "same_text_pairs": len(same_text),
            "total_pairs": len(pairs),
            "same_text_median_speed_ratio": median(pair["speed_ratio"] for pair in same_text) if same_text else None,
        },
        "acceptance_limit": (
            "The first four arms sampled periodic counters after one second; deltas may straddle request boundaries. "
            "They are not exact per-request acceptance or round timing."
        ),
    }
    for size in (8192, 23410):
        cold_a = [arms[label][f"cold{size}"] for label in ("residual_a1", "residual_a2")]
        cold_b = [arms[label][f"cold{size}"] for label in ("gdn_b1", "gdn_b2")]
        result[f"cold{size}"] = {
            "baseline_median_ttft_s": median(row["time_to_first_token_s"] for row in cold_a),
            "candidate_median_ttft_s": median(row["time_to_first_token_s"] for row in cold_b),
            "cached_tokens": [row["cached_tokens"] for row in cold_a + cold_b],
            "output_hashes": [row["output_sha256"] for row in cold_a + cold_b],
            "samples_per_arm": 2,
        }
    if (ROOT / "server-c4.log").exists():
        c4_server = timings("server-c4.log")
        groups = {}
        for arm in ("original", "residual"):
            records = [read(f"c4_{arm}_{repeat}-c4") for repeat in range(3)]
            groups[arm] = {
                "aggregate_tok_s": [record["aggregate_tok_s"] for record in records],
                "aggregate_median_tok_s": median(record["aggregate_tok_s"] for record in records),
                "hashes": [[row["text_sha256"] for row in record["results"]] for record in records],
                "server_tok_s": [
                    [c4_server[row["request_id"]]["decode_tok_s"] for row in record["results"]] for record in records
                ],
            }
        groups["median_speed_ratio"] = (
            groups["residual"]["aggregate_median_tok_s"] / groups["original"]["aggregate_median_tok_s"]
        )
        result["c4_graph"] = groups
    if (ROOT / "demo5-health.json").exists():
        c4 = read("demo5-c4-128")
        result["demo_final"] = {
            **read("demo5-health"),
            "paused": read("demo5-final-paused")["is_paused"],
            "serial128_client_decode_tok_s": read("demo5-serial128")["client_decode_tok_s"],
            "c4_128_aggregate_tok_s": sum(row["usage"]["completion_tokens"] for row in c4["results"]) / c4["wall_s"],
            "cold4096": read("demo5-cold4096"),
            "capacity_limit": "Planner capacity only; five full 256K windows were not stress-tested.",
        }
    if (ROOT / "recovery-readiness.json").exists():
        result["active_profile"] = read("recovery-readiness")
    elif (ROOT / "vision-readiness.json").exists():
        result["active_profile"] = read("vision-readiness")
    if (ROOT / "service-status-20261008.json").exists():
        result["service_status"] = read("service-status-20261008")
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "arms"}, indent=2))


if __name__ == "__main__":
    summarize()
