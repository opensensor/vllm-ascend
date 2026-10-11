# SPDX-License-Identifier: Apache-2.0
"""Reversible staging pins existing banks and preserves previous receipts."""

import json

import pytest

from tools.sidecard.native_expert_probe import digest_file
from tools.sidecard.service_restart import prepare_restart


def restart_fixture(root):
    root.mkdir()
    package = root / "source" / "tools" / "sidecard"
    package.mkdir(parents=True)
    (package / "old.py").write_text("old package")
    (root / "source" / "sidecard_worker.py").write_text("worker extension")
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "native-layout.json").write_text("pinned checkpoint")
    (root / "cuda-worker.json").write_text("{}")
    ready = dict(
        pid=123,
        config_sha256=digest_file(root / "cuda-worker.json"),
        checkpoint_sha256=digest_file(checkpoint / "native-layout.json"),
        resident_layers=42,
        resident_experts=3024,
        max_sessions=16,
    )
    (root / "cuda-ready.json").write_text(json.dumps(ready))
    config = dict(
        checkpoint_root=str(checkpoint),
        checkpoint_sha256=ready["checkpoint_sha256"],
        worker_ready_path=str(root / "cuda-ready.json"),
        worker_config_sha256=ready["config_sha256"],
        cuda_socket=str(root / "cuda.sock"),
        report_path=str(root / "metrics.json"),
    )
    (root / "service.json").write_text(json.dumps(config))
    (root / "service-process.json").write_text(json.dumps(dict(pid=456, command=["python", "serve", "/model"])))
    (root / "runtime-environment.json").write_text("{}")


def test_restart_keeps_bank_identity_and_uses_a_fresh_metrics_path(tmp_path, monkeypatch):
    root, output = tmp_path / "old", tmp_path / "new"
    restart_fixture(root)
    monkeypatch.setattr("tools.sidecard.service_restart.os.kill", lambda pid, sig: None)
    before = (root / "service.json").read_bytes()
    receipt = prepare_restart(root, output)
    assert receipt["cuda_pid"] == 123
    assert receipt["replace_pid"] == 456
    config = json.loads((output / "service.json").read_bytes())
    assert config["cuda_socket"] == str(root / "cuda.sock")
    assert config["worker_ready_path"] == str(root / "cuda-ready.json")
    assert config["report_path"] == str(output / "model-boundary-metrics.json")
    assert (root / "service.json").read_bytes() == before
    assert digest_file(root / "cuda-worker.json") == digest_file(output / "cuda-worker.json")
    assert not (output / "source" / "tools" / "sidecard" / "old.py").exists()
    assert (output / "source" / "tools" / "sidecard" / "service_restart.py").exists()


@pytest.mark.parametrize("change", ["sessions", "digest", "checkpoint", "output"])
def test_restart_refuses_stale_or_nonreconnectable_banks_before_creating_files(tmp_path, monkeypatch, change):
    root, output = tmp_path / "old", tmp_path / "new"
    restart_fixture(root)
    monkeypatch.setattr("tools.sidecard.service_restart.os.kill", lambda pid, sig: None)
    if change == "sessions":
        path = root / "cuda-ready.json"
        ready = json.loads(path.read_text())
        ready["max_sessions"] = 1
        path.write_text(json.dumps(ready))
    elif change == "digest":
        (root / "cuda-worker.json").write_text("changed")
    elif change == "checkpoint":
        (root / "checkpoint" / "native-layout.json").write_text("changed")
    else:
        output.mkdir()
    with pytest.raises(ValueError):
        prepare_restart(root, output)
    assert not (output / "service.json").exists()
