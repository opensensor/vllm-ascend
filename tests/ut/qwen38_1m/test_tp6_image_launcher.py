# SPDX-License-Identifier: Apache-2.0
"""Check the actual rendered TP6 command without loading weights or CANN."""

import json
import shlex
import subprocess
import sys
from pathlib import Path


def test_tp6_launcher_retains_images_and_decode_optimizations(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = (root / "artifacts/qfn-tp6-scaleout-20261016/start-tp6.sh").read_text()
    source = source.replace(
        "experiment_root=/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016",
        f"experiment_root={shlex.quote(str(tmp_path))}",
    ).replace(
        "python_bin=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python",
        f"python_bin={shlex.quote(sys.executable)}",
    )
    launcher = tmp_path / "start.sh"
    launcher.write_text(source)
    (tmp_path / "collective-serving.json").write_text(
        json.dumps({"execution_mode": "graph", "graph_replay_tracking": False})
    )
    run_root = tmp_path / "fresh run"
    command = shlex.split(
        subprocess.check_output(["bash", str(launcher), "--run-root", str(run_root), "--show"], text=True)
    )

    def option(name):
        return command[command.index(name) + 1]

    assert "--language-model-only" not in command
    assert option("--mm-encoder-tp-mode") == "data"
    assert json.loads(option("--limit-mm-per-prompt")) == {"image": 1, "video": 0}
    assert json.loads(option("--mm-processor-kwargs")) == {"min_pixels": 65536, "max_pixels": 1048576}
    assert option("--tensor-parallel-size") == "6"
    assert option("--max-model-len") == "262144"
    assert option("--max-num-seqs") == "6"
    assert "--enable-expert-parallel" in command
    assert "--enable-prefix-caching" in command
    assert json.loads(option("--speculative-config")) == {"method": "mtp", "num_speculative_tokens": 2}
    graphs = json.loads(option("--compilation-config"))
    assert graphs["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert graphs["cudagraph_capture_sizes"] == [3, 6, 12, 18]
    collective = json.loads(option("--additional-config"))["npu_collective"]
    assert collective["execution_mode"] == "graph"
    assert collective["graph_replay_tracking"] is False
    assert collective["output_root"] == str(run_root / "owners")
    assert not run_root.exists(), "--show must not allocate runtime owners"
