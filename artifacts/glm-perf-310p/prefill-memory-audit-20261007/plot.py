"""Plot archived measurements; does not contact or modify a serving process."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent
PREFILL = ARTIFACTS / "native-prefill-20261006/v45-attribution.json"
DECODE = ARTIFACTS / "native-throughput-20261006/trace-v64-decode/attribution.json"
KERNEL = ARTIFACTS / "native-throughput-20261006/v56-source.cpp.txt"
CATEGORIES = (
    ("native_int4_expert", "Native expert projections"),
    ("collective", "Collectives (includes waiting)"),
    ("kda_convolution", "KDA / convolution"),
    ("sparse_attention", "Sparse attention"),
    ("copy_cast_layout", "Separate copy / cast / layout"),
    ("native_activation_pack", "Input quantization"),
)
PIPELINES = (
    ("vec_ratio", "Vector"),
    ("scalar_ratio", "Scalar"),
    ("mte2_ratio", "MTE2"),
    ("mte1_ratio", "MTE1"),
    ("mte3_ratio", "MTE3"),
    ("mac_ratio", "MAC"),
)


def summarize(values):
    return {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}


def main():
    prefill = json.loads(PREFILL.read_text())
    decode = json.loads(DECODE.read_text())
    assert len(prefill["ranks"]) == len(decode["workloads"]["c1"]) == 4
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(CATEGORIES))
    measurements = {"prefill": {}, "decode_pipelines": {}}
    for step, offset, color in ((0, -0.18, "#2864a5"), (1, 0.18, "#d67b27")):
        values = []
        for category, _ in CATEGORIES:
            samples = []
            for rank in prefill["ranks"]:
                record = rank["steps"][step]
                assert record["tokens"] == 640
                samples.append(record["categories"].get(category, {}).get("summed_ms", 0) / 1000)
            values.append(summarize(samples))
        measurements["prefill"][str(step)] = dict(zip((key for key, _ in CATEGORIES), values))
        means = np.array([v["mean"] for v in values])
        errors = np.array([[v["mean"] - v["min"] for v in values], [v["max"] - v["mean"] for v in values]])
        axes[0].barh(
            y + offset, means, height=0.32, xerr=errors, capsize=3, color=color, label=f"640-token chunk {step + 1}"
        )
    axes[0].set_yticks(y, [label for _, label in CATEGORIES])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Summed task duration per rank (seconds)")
    axes[0].set_title("Historical v45 cold prefill, all four ranks\nBars: rank mean; whiskers: rank range")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="x", alpha=0.2)
    for index, (key, label) in enumerate(PIPELINES):
        samples = [
            rank["decode"]["pipeline_metrics"]["native_int4_expert"][key] * 100 for rank in decode["workloads"]["c1"]
        ]
        value = summarize(samples)
        measurements["decode_pipelines"][key] = value
        axes[1].barh(index, value["mean"], color="#3c8d81")
        axes[1].text(value["mean"] + 1, index, f"{value['mean']:.2f}%", va="center")
    axes[1].set_yticks(np.arange(len(PIPELINES)), [label for _, label in PIPELINES])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 76)
    axes[1].set_xlabel("Duration-weighted pipeline activity (%)")
    axes[1].set_title("Historical v64 c1 decode: native experts\nFour-rank mean; pipelines can overlap")
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle("GLM 310P: projection overhead is the profiling target", fontsize=17)
    figure.text(
        0.5,
        0.035,
        "Archived, profiled runs: not the current v56 workload. Task sums are not an additive critical path.\n"
        "Separate copy tasks exclude movement inside native kernels. "
        "Pipeline activity does not measure memory bandwidth.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.93))
    figure.savefig(ROOT / "archived-prefill-and-pipelines.png", dpi=180)
    plt.close(figure)
    measurements["sources"] = {
        str(path.relative_to(ARTIFACTS)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (PREFILL, DECODE, KERNEL)
    }
    measurements["scope"] = "Historical v45 prefill task sums and v64 decode pipeline metrics, not current v56."
    measurements["logical_route_traffic"] = {
        "tokens": 640,
        "top_k": 8,
        "hidden": 4096,
        "bytes_per_value": 4,
        "buffer_mib_per_rank": 640 * 8 * 4096 * 4 / 2**20,
        "write_plus_read_gib_per_rank_42_target_moe_layers": 640 * 8 * 4096 * 4 * 2 * 42 / 2**30,
        "scope": "Static logical traffic estimate, not measured bandwidth; excludes MTP and other buffers.",
    }
    (ROOT / "evidence.json").write_text(json.dumps(measurements, indent=2) + "\n")


if __name__ == "__main__":
    main()
