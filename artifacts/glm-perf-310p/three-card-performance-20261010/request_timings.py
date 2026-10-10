"""Extract request-level prefill/decode evidence from archived service logs."""

import argparse
import json
from pathlib import Path

import regex as re

PATTERN = re.compile(
    r"Engine 000 request (?P<request>\S+) \([^)]*\): prompt "
    r"(?P<computed>\d+) computed \+ (?P<cached>\d+) cached in (?P<prefill>[\d.]+) ms .*?; "
    r"decode (?P<tokens>\d+) tokens, (?P<gaps>\d+) gaps in (?P<decode>[\d.]+) ms"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.log.read_text().splitlines():
        match = PATTERN.search(line)
        if match is None:
            continue
        value = match.groupdict()
        gaps = int(value["gaps"])
        decode = float(value["decode"])
        rows.append(
            {
                "request": value["request"],
                "computed_prompt_tokens": int(value["computed"]),
                "cached_prompt_tokens": int(value["cached"]),
                "prefill_ms": float(value["prefill"]),
                "decode_tokens": int(value["tokens"]),
                "decode_gaps": gaps,
                "decode_ms": decode,
                "decode_ms_per_gap": decode / gaps if gaps else None,
                "decode_tokens_per_s": 1000 * gaps / decode if gaps and decode else None,
            }
        )
    assert rows, "no completed request timing evidence"
    args.output.write_text(json.dumps({"source_log": str(args.log), "requests": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
