"""Plot streaming content-update gaps; an update can contain multiple tokens."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    cases = (
        ("decode-alone-v1", "Decode alone"),
        ("mixed-spans-v1", "Default: up to 1,920 prefill + 3 decode tokens"),
        ("mixed-limit256-v1", "Temporary 256-token prefill limit"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, sharey=True)
    for axis, (label, title) in zip(axes, cases):
        receipt = json.loads((ROOT / label / "receipt.json").read_text())
        decode = receipt["decode"]
        start = decode["start_ns"]
        events = [row["time_ns"] for row in decode["events"]]
        x = [(b - start) / 1e9 for b in events[1:]]
        y = [(b - a) / 1e9 for a, b in zip(events, events[1:])]
        axis.plot(x, y, ".-", markersize=3, linewidth=0.8, color="#175a9c")
        if "prefill" in receipt:
            prefill = receipt["prefill"]
            begin = (prefill["start_ns"] - start) / 1e9
            axis.axvspan(begin, begin + prefill["ttft_s"], color="#e8a23b", alpha=0.2)
        axis.set_title(f"{title} | maximum gap {max(y):.3f} s", loc="left", fontsize=11)
        axis.set_yscale("log")
        axis.set_ylim(0.02, 20)
        axis.set_ylabel("Update gap (s)")
        axis.grid(alpha=0.25, which="both")
    axes[-1].set_xlabel("Seconds since the decoding request began")
    figure.suptitle("A long prefill stalls an existing decoder on the retained four-NPU server")
    figure.text(
        0.08,
        0.02,
        "Shading: second request submitted through its first content update. "
        "SSE updates may contain multiple tokens. Single trials.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    figure.savefig(ROOT / "streaming-gaps.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
