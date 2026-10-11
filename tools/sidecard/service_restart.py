# SPDX-License-Identifier: Apache-2.0
"""Stage a fresh API deployment while retaining authenticated CUDA banks.

The old API must release its peer before the replacement connects. Use the
normal launcher with --replace-pid: its import preflight and receipt-checked
SIGTERM happen after this reversible staging step.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from .native_expert_probe import digest_file


def prepare_restart(previous_root, output, *, npu_devices=None):
    previous_root, output = (Path(path).resolve() for path in (previous_root, output))
    if output.exists() or output.is_relative_to(previous_root):
        raise ValueError("restart requires a fresh directory outside the previous deployment")
    config = json.loads((previous_root / "service.json").read_bytes())
    if npu_devices is not None:
        from .one_card import validate_device_pair

        if not config.get("one_card", False):
            raise ValueError("physical NPU pair selection requires one-card mode")
        validate_device_pair(npu_devices)
    ready = json.loads(Path(config["worker_ready_path"]).read_bytes())
    if (
        ready["config_sha256"] != digest_file(previous_root / "cuda-worker.json")
        or ready["config_sha256"] != config["worker_config_sha256"]
        or ready["checkpoint_sha256"] != config["checkpoint_sha256"]
        or ready["resident_layers"] != 42
        or ready["resident_experts"] != (6048 if config.get("one_card", False) else 3024)
        or ready.get("max_sessions", 1) <= 1
    ):
        raise ValueError("restart requires authenticated reconnectable CUDA banks")
    os.kill(ready["pid"], 0)
    previous = json.loads((previous_root / "service-process.json").read_bytes())
    if digest_file(Path(config["checkpoint_root"]) / "native-layout.json") != config["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after CUDA preparation")
    output.mkdir(mode=0o700)
    shutil.copytree(
        previous_root / "source", output / "source", ignore=shutil.ignore_patterns("__pycache__", "kernel_meta")
    )
    package = output / "source" / "tools" / "sidecard"
    shutil.rmtree(package)
    shutil.copytree(Path(__file__).parent, package, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("cuda-worker.json", "runtime-environment.json"):
        shutil.copyfile(previous_root / name, output / name)
    if npu_devices is not None:
        runtime = json.loads((output / "runtime-environment.json").read_bytes())
        runtime["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, npu_devices))
        (output / "runtime-environment.json").write_text(json.dumps(runtime, indent=2) + "\n")
        config["npu_devices"] = list(npu_devices)
    config["report_path"] = str(output / "model-boundary-metrics.json")
    (output / "service.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "previous-service.json").write_text(json.dumps(previous, indent=2) + "\n")
    receipt = {
        "previous_root": str(previous_root),
        "output": str(output),
        "cuda_pid": ready["pid"],
        "replace_pid": previous["pid"],
    }
    (output / "cuda-reuse.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npu-devices", choices=("0,1", "2,3"))
    args = parser.parse_args()
    devices = tuple(map(int, args.npu_devices.split(","))) if args.npu_devices else None
    print(json.dumps(prepare_restart(args.previous_root, args.output, npu_devices=devices)), flush=True)


if __name__ == "__main__":
    main()
