"""Collect compact measurements and provenance without private control credentials."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-hc-epilogues-20261015")
BASE = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
TRACES = Path("/srv/ai/storage/artifacts/qfn-mixed-prefill-20261015")
TOOLS = Path("/srv/ai/storage/artifacts/ple-host-gather-20261015/serving-source-v2/tools/qwen4exp")


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    trials = []
    for label in ("off-a", "on-a", "off-b", "on-b", "affine-shadow", "prefill-off", "prefill-on"):
        directory = TRACES / ("hc-epilogues-" + label)
        trial = dict(label=label, source=str(directory))
        for name in ("prefill", "decode"):
            path = directory / f"{name}-stream.json"
            if path.exists():
                trial[name] = {key: value for key, value in read(path).items() if key != "events"}
        if len(trial) > 2:
            trials.append(trial)
    write("measurements.json", trials)
    gates = {}
    for label in (
        "hc-install",
        "hc-qualified",
        "hc-large-qualified",
        "final-audit",
        "final-audit-after-c4",
        "c4-off",
        "c4-off-after",
        "c4-on",
        "c4-on-after",
    ):
        path = ROOT / label / "receipt.json"
        if path.exists():
            gates[label] = read(path)["applied"]
    write("gates.json", gates)
    paths = [
        ROOT / "build-v3/native_hc_epilogues.cpp",
        ROOT / "build-v3/native_hc_epilogues.bin",
        TOOLS / "hc_epilogues.py",
        TOOLS / "apply_hc_epilogues.py",
        Path("/srv/ai/storage/artifacts/qfn-hc-combine-compile-20261014-v1/build/qwen_hc_combine_v1.so"),
    ]
    write("provenance.json", {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    decode = {}
    for name in ("decode-benchmark", "concurrency"):
        path = BASE / "qsa-resident-v1" / f"{name}-hc-epilogues-v1.json"
        if path.exists():
            data = read(path)
            decode[name] = {key: value for key, value in data.items() if key not in ("before", "after", "workers")}
    write("decode.json", decode)
    c4 = {}
    for action in ("off", "on"):
        path = BASE / "qsa-resident-v1" / f"concurrency-hc-epilogues-c4-{action}.json"
        if path.exists():
            data = read(path)
            c4[action] = {
                key: data[key] for key in ("makespan_s", "aggregate_http_output_tok_s", "median_client_decode_tok_s")
            }
    write("c4-comparison.json", c4)


if __name__ == "__main__":
    main()
