"""Pack downloaded Strata artifacts and write the two GPQA experiment configs.

Requires the pinned source checkout, Python environment, sm_120 build, IQ2_XS
download, and fetched MTP tensors. No GPU initialization happens in this step.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from experiment import digest, read_cases

STRATA_REVISION = "61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406"
IQ2_REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
IQ4_REVISION = "38bb39ee97821de2c9009abb7e93950eec396e66"
IQ2_HASHES = (
    "92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7",
    "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113",
)
IQ4_HASHES = (
    "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4",
    "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7",
    "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833",
)


def run_tool(source, name, *args):
    subprocess.run(
        [str(source / ".venv/bin/python"), str(source / "tools" / name), *map(str, args)], cwd=source, check=True
    )


def verify_shards(paths, hashes):
    for path, expected in zip(paths, hashes):
        print(f"Verifying {path.name}", flush=True)
        if digest(path) != expected:
            raise ValueError(f"Model checksum mismatch: {path}")


def make_config(root, arm, native, model_hashes, revision):
    source = root / "Strata"
    pack = root / "data/packs" / arm
    runtime = [p for directory in (pack, root / "data/mtp/rt") for p in sorted(directory.iterdir()) if p.is_file()]
    args = [
        "--pack",
        str(pack),
        "--native",
        str(native),
        "--expert-profile",
        str(source / "data/expert-profile.bin"),
        "--expert-cache",
        "auto",
        "--prefill",
        "auto",
        "--spec",
        "4",
        "--spec-min-p",
        "0.5",
        "--mtp",
        str(root / "data/mtp/rt"),
        "--max-context",
        "262144",
        "--kv",
        "int8",
        "--vram-reserve-mib",
        "4096",
    ]
    if arm == "iq2":
        args += ["--ple-gguf", str(native.with_name(native.name.replace("00001-of", "00002-of")))]
    else:
        args += ["--resident-budget-gib", "60"]
    return {
        "exe": str(source / "build/strata"),
        "cwd": str(source),
        "args": args,
        "tokenizer": str(pack / "tokenizer"),
        "model_name": f"qwen38-strata-{arm}",
        "log": str(root / "results" / f"engine-{arm}.log"),
        "lib_dirs": ["/usr/local/cuda/lib64"],
        "gpu": 0,
        "gpus_asked": True,
        "host": "127.0.0.1",
        "port": 8096,
        "parallel": 1,
        "reasoning_effort": "high",
        "reasoning_budget_tokens": 0,
        "reasoning_loop_recovery": False,
        "tool_call_recovery": False,
        "env": {"STRATA_ROUTE_TAIL_SKIP": "7"},
        "experiment": {
            "model_revision": revision,
            "model_sha256": model_hashes,
            "strata_revision": STRATA_REVISION,
            "model_files": [
                {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                for path in sorted(native.parent.glob("*.gguf"))
            ],
            "runtime_files": [{"path": str(path), "sha256": digest(path)} for path in runtime],
            "status": "offline prepared; GPU validation pending",
            "sampling": "frozen October 2 GPQA replay; serial throughput differs from old C3 run",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--iq4-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "Strata"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != STRATA_REVISION:
        raise ValueError("Strata source differs from the pinned experiment")
    read_cases(args.reference / "cases.json")
    if not (source / "build/strata").is_file():
        raise ValueError("Build the sm_120 Strata engine first")
    iq2_dir = root / "data/models/IQ2_XS"
    iq2 = [iq2_dir / f"Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-{i:05d}-of-00002.gguf" for i in (1, 2)]
    iq4 = [args.iq4_dir.resolve() / f"Qwen3.8-Flash-Next-UD-IQ4_XS-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    verify_shards(iq2, IQ2_HASHES)
    verify_shards(iq4, IQ4_HASHES)
    for arm, paths in (("iq2", iq2), ("iq4", iq4)):
        pack = root / "data/packs" / arm
        if not (pack / "native_experts.txt").exists():
            run_tool(
                source, "iq_pack.py", "--gguf", paths[0], "--out", pack, *(["--compat-bf16"] if arm == "iq4" else [])
            )
    mtp = root / "data/mtp"
    run_tool(source, "mtp_fetch.py", "verify", "--out", mtp)
    if not (mtp / "rt/experts.bin").exists():
        run_tool(source, "mtp_pack.py", "--src", mtp, "--experts", "q2_0", "--out", mtp / "mtp-q2_0.gguf")
        run_tool(source, "mtp_rt.py", "--gguf", mtp / "mtp-q2_0.gguf", "--out", mtp / "rt")
    shutil.copy2(source / "data/draft_vocab.bin", mtp / "rt/draft_vocab.bin")
    (root / "configs").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    for arm, paths, hashes, revision in (
        ("iq2", iq2, IQ2_HASHES, IQ2_REVISION),
        ("iq4", iq4, IQ4_HASHES, IQ4_REVISION),
    ):
        config = make_config(root, arm, paths[0], hashes, revision)
        (root / "configs" / f"{arm}.json").write_text(json.dumps(config, indent=2))
        config["env"]["STRATA_ROUTE_TAIL_SKIP"] = "0"
        (root / "configs" / f"{arm}-no-tail-skip.json").write_text(json.dumps(config, indent=2))
    for name in ("cases.json", "results.jsonl", "ascend-complete-results.jsonl", "manifest.json"):
        target = root / "reference" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(args.reference / name, target)
    print("Offline preparation complete. Run the GPU smoke gate before scoring.", flush=True)


if __name__ == "__main__":
    main()
