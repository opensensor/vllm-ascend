# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[3] / "artifacts/qwen38-decode-next-20261005/run-component-batch.py"


def batch(tmp_path, failures):
    (tmp_path / "torch.py").write_text(
        "class Npu:\n    def synchronize(self): pass\n    def empty_cache(self): pass\nnpu = Npu()\n"
    )
    (tmp_path / "fake_probe.py").write_text(
        "import argparse, json, os\nfrom pathlib import Path\n"
        "def main():\n"
        "    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--error',default='')\n"
        "    a=p.parse_args()\n"
        "    record={'pid':os.getpid(),'error':a.error} if a.error else {'pid':os.getpid()}\n"
        "    Path(a.output).write_text(json.dumps(record))\n"
        "    if a.error: raise RuntimeError(a.error)\n"
    )
    jobs = [
        {"id": str(i), "module": "fake_probe", "output": str(tmp_path / f"{i}.json"), "args": ["--error", error]}
        for i, error in enumerate(failures)
    ]
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps(jobs))
    progress = tmp_path / "progress.json"

    def execute():
        return subprocess.run(
            [sys.executable, str(RUNNER), "--jobs", str(jobs_path), "--progress", str(progress)],
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
            capture_output=True,
            text=True,
            check=False,
        )

    return execute, progress


def test_successful_probes_share_process_and_resume_preserves_receipts(tmp_path):
    execute, progress = batch(tmp_path, ["", ""])
    assert execute().returncode == 0
    a, b = [json.loads((tmp_path / f"{i}.json").read_text()) for i in range(2)]
    assert a["pid"] == b["pid"]
    saved = (tmp_path / "0.json").read_bytes()
    assert execute().returncode == 0
    assert (tmp_path / "0.json").read_bytes() == saved
    assert len(json.loads(progress.read_text())["jobs"]) == 2


@pytest.mark.parametrize("error,code", [("unsupported capture", 20), ("device execution 507015", 21)])
def test_failure_stops_submission_before_next_job(tmp_path, error, code):
    execute, progress = batch(tmp_path, [error, ""])
    assert execute().returncode == code
    assert not (tmp_path / "1.json").exists()
    assert len(json.loads(progress.read_text())["jobs"]) == 1
    assert error in (tmp_path / "0.log").read_text()
    if code == 20:
        assert execute().returncode == 0
        assert (tmp_path / "1.json").exists()


def test_unfinished_log_is_preserved_and_blocks_execution(tmp_path):
    execute, progress = batch(tmp_path, [""])
    log = tmp_path / "0.log"
    log.write_text("unfinished evidence")
    assert execute().returncode != 0
    assert log.read_text() == "unfinished evidence"
    assert not progress.exists()
