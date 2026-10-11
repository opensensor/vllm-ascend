# SPDX-License-Identifier: Apache-2.0
"""Admission and real-service route invariants without accelerator contexts."""

import json
from unittest.mock import Mock

import pytest

from tests.ut.sidecard.test_native_expert_probe import native_fixture
from tools.sidecard.native_expert_probe import digest_file
from tools.sidecard.resident import CudaResidentBackend, NativeResidentPlan
from tools.sidecard.service_adapter import SidecardStatusMixin, local_request, service_placement
from tools.sidecard.service_launch import serving_command, stop_previous
from tools.sidecard.service_prepare import prepare
from tools.sidecard.service_validate import check_metrics, completion_text
from tools.sidecard.service_validate import run as validate_service
from tools.sidecard.wire import PacketLimits, WireIdentity


def authenticated_fixture(root, bits=4):
    native_fixture(root, bits)
    path = root / "native-layout.json"
    manifest = json.loads(path.read_bytes())
    digest = digest_file(root / "native.safetensors")
    for record in (*manifest["native_shards"], *manifest["scale_shards"]):
        record["sha256"] = digest
    path.write_text(json.dumps(manifest))
    return digest_file(path)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_resident_plan_authenticates_payload_and_accounts_unpacked_cuda_bytes(tmp_path, bits):
    digest = authenticated_fixture(tmp_path, bits)
    plan = NativeResidentPlan(tmp_path, ((3, 72),), digest)
    assert plan.by_layer == {3: (72,)}
    assert plan.required_cuda_bytes == 3 * 4096 * 2048 + 3 * 128 * 64 * 2
    with pytest.raises(RuntimeError, match="authenticated"):
        next(plan.items())
    plan.authenticate()
    assert plan.authenticated
    plan.check_unchanged()
    with (tmp_path / "native.safetensors").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ValueError, match="changed"):
        plan.check_unchanged()
    with pytest.raises(ValueError, match="checksum"):
        plan.authenticate()


def test_resident_plan_rejects_metadata_mutation_and_wrong_identity(tmp_path):
    digest = authenticated_fixture(tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        NativeResidentPlan(tmp_path, ((3, 72),), "a" * 64)
    plan = NativeResidentPlan(tmp_path, ((3, 72),), digest)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata changed"):
        plan.authenticate()


@pytest.mark.parametrize("keys", [(), ((45, 72),), ((3, 288),), ((3, 72), (3, 72))])
def test_resident_plan_excludes_draft_foreign_and_duplicate_keys(tmp_path, keys):
    with pytest.raises(ValueError, match="keys"):
        NativeResidentPlan(tmp_path, keys, "a" * 64)


def test_service_placement_matches_native_ep_banks_in_every_main_layer():
    placement = service_placement()
    assert len(placement.layers) == 42
    assert placement.owners == ("cuda:0", "npu:1", "npu:2", "npu:3")
    for layer in range(3, 45):
        assert placement.layer(layer) == ("cuda:0",) * 72 + ("npu:1",) * 72 + ("npu:2",) * 72 + ("npu:3",) * 72


def test_worker_extension_has_no_native_lifecycle_method_conflicts():
    assert not any(hasattr(SidecardStatusMixin, name) for name in ("shutdown", "load_model", "execute_model"))


def test_service_preserves_nonzero_original_slots_and_compacts_tokens():
    hidden = ((0.25,) * 4096,) * 3
    ids = ((0, 0, 71, 0, 5, 0, 0, 0), (0,) * 8, (6, 0, 0, 0, 0, 0, 0, 7))
    weights = ((0, 0, 0.5, 0, 0.25, 0, 0, 0), (0,) * 8, (0.5, 0, 0, 0, 0, 0, 0, 0.5))
    request = local_request("request", 8, hidden, ids, weights)
    assert request.token_ids == (0, 2)
    assert tuple((route.token, route.slot, route.expert) for route in request.routes) == (
        (0, 2, 71),
        (0, 4, 5),
        (2, 0, 6),
        (2, 7, 7),
    )
    assert request.hidden == (hidden[0], hidden[2])
    assert local_request("idle", 3, hidden, ((0,) * 8,) * 3, ((0,) * 8,) * 3) is None


@pytest.mark.parametrize("expert,weight", [(72, 0.5), (1, -1), (1, float("nan")), (True, 0.5)])
def test_service_rejects_foreign_negative_or_nonfinite_routes(expert, weight):
    with pytest.raises(ValueError, match="route"):
        local_request("bad", 3, ((0.25,) * 4096,), ((expert, 0, 0, 0, 0, 0, 0, 0),), ((weight, 0, 0, 0, 0, 0, 0, 0),))


def test_service_rejects_duplicate_active_experts():
    with pytest.raises(ValueError, match="duplicate"):
        local_request("bad", 3, ((0.25,) * 4096,), ((1, 1, 0, 0, 0, 0, 0, 0),), ((0.5, 0.5, 0, 0, 0, 0, 0, 0),))


def test_resident_backend_rejects_geometry_before_device_initialization():
    placement = service_placement()
    identity = WireIdentity.for_placement(placement, "a" * 64)
    specification = dict(
        checkpoint_root="/unused",
        library="/unused",
        library_sha256="b" * 64,
        cuda_runtime="/unused",
        cuda_memory_budget_bytes=82 * 1024**3,
    )
    with pytest.raises(ValueError, match="specification"):
        CudaResidentBackend(specification, "cuda:0", placement, identity, PacketLimits(4096, 33, 264, 288, 8))


def test_serving_command_preserves_qualified_model_and_requires_eager_target_only():
    previous = [
        "python",
        "serve",
        "/model",
        "--host",
        "127.0.0.1",
        "--served-model-name",
        "baseline",
        "--compilation-config",
        "{}",
        "--tensor-parallel-size",
        "4",
    ]
    snapshot = list(previous)
    command = serving_command(previous, host="0.0.0.0", port=8001)
    assert previous == snapshot
    assert command[2] == "/model"
    assert command[command.index("--served-model-name") + 1] == "glm-sidecard"
    assert command[command.index("--worker-extension-cls") + 1] == "sidecard_worker.SidecardWorkerExtension"
    assert json.loads(command[command.index("--compilation-config") + 1]) == {"cudagraph_mode": "NONE", "mode": 0}
    assert command.count("--enforce-eager") == 1
    with pytest.raises(ValueError, match="MTP"):
        serving_command(previous + ["--speculative-config", "{}"])


def test_launcher_rejects_unidentified_replacement_pid_before_signalling():
    with pytest.raises(ValueError, match="PID differs"):
        stop_previous(42, {"pid": 43, "command": []})


def test_context_launch_overrides_preserve_other_qualified_arguments():
    previous = ["python", "serve", "/model", "--max-model-len", "8192", "--max-num-seqs", "4"]
    command = serving_command(previous, max_model_len=-1, kv_cache_memory_bytes=8 * 1024**3)
    assert command[command.index("--max-model-len") + 1] == "-1"
    assert command[command.index("--kv-cache-memory-bytes") + 1] == str(8 * 1024**3)
    assert command[command.index("--max-num-seqs") + 1] == "4"
    assert command.count("--max-model-len") == 1
    assert previous[previous.index("--max-model-len") + 1] == "8192"


def test_capacity_first_launch_uses_compact_live_state_and_one_sequence():
    previous = [
        "python",
        "serve",
        "/model",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--max-num-seqs",
        "4",
    ]
    command = serving_command(previous, capacity_first=True)
    assert "--enable-prefix-caching" not in command
    assert command.count("--no-enable-prefix-caching") == 1
    assert command[command.index("--mamba-cache-mode") + 1] == "none"
    assert command[command.index("--max-num-seqs") + 1] == "1"
    assert serving_command(command, capacity_first=True).count("--no-enable-prefix-caching") == 1
    assert "--enable-prefix-caching" in previous


def test_coding_profile_restores_prefix_reuse_and_larger_prefill():
    previous = ["python", "serve", "/model", "--no-enable-prefix-caching", "--mamba-cache-mode", "none"]
    command = serving_command(previous, prefix_caching=True, max_num_seqs=1, prefill_tokens=512)
    assert "--no-enable-prefix-caching" not in command
    assert command[command.index("--mamba-cache-mode") + 1] == "align"
    assert command[command.index("--max-num-batched-tokens") + 1] == "512"
    assert command[command.index("--max-num-seqs") + 1] == "1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefill_tokens": 0},
        {"prefill_tokens": 641},
        {"max_num_seqs": 0},
        {"capacity_first": True, "prefix_caching": True},
    ],
)
def test_invalid_coding_profile_fails_before_launch(kwargs):
    with pytest.raises(ValueError):
        serving_command(["python", "serve", "/model"], **kwargs)


@pytest.mark.parametrize("length,budget", [(0, None), (-2, None), (True, None), (None, 0), (None, -1), (None, True)])
def test_invalid_context_launch_bounds_fail_before_process_creation(length, budget):
    with pytest.raises(ValueError):
        serving_command(["python", "serve", "/model"], max_model_len=length, kv_cache_memory_bytes=budget)


def test_deployment_preparation_preserves_venv_interpreter_symlink(tmp_path, monkeypatch):
    qualified = tmp_path / "qualified"
    qualified.mkdir()
    (qualified / "qualified-marker").write_text("unchanged")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    authenticated_fixture(checkpoint)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"pid": 42, "command": ["python", "serve", str(checkpoint)]}))
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({"PYTHONPATH": str(qualified)}))
    library = tmp_path / "oracle.so"
    library.write_bytes(b"test library")
    system_python = tmp_path / "system-python"
    system_python.write_bytes(b"test interpreter")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(system_python)
    popen = Mock(return_value=Mock(pid=123))
    monkeypatch.setattr("tools.sidecard.service_prepare.subprocess.Popen", popen)
    output = tmp_path / "deployment"
    result = prepare(qualified, baseline, environment, checkpoint, library, venv_python, output)
    assert result["command"][0] == str(venv_python)
    assert result["pid"] == 123
    assert (qualified / "qualified-marker").read_text() == "unchanged"
    worker = json.loads((output / "cuda-worker.json").read_text())
    assert worker["backend"]["library_sha256"] == digest_file(library)
    assert worker["limits"]["max_routes"] == 256
    assert len(worker["placement"]["layers"]) == 42
    with pytest.raises(ValueError, match="fresh"):
        prepare(qualified, baseline, environment, checkpoint, library, venv_python, output)
    assert popen.call_count == 1


def test_service_validation_rejects_startup_only_or_missing_layer_evidence():
    value = dict(
        cuda_worker_pid=42,
        profile="ascend-backbone-cuda-experts",
        cuda_attention=False,
        retains_npu_shadow_banks=True,
        layers={
            str(layer): dict(
                real_calls=3, real_cuda_routes=2, shadow_checks=3, shadow_max_absolute_error=0, shadow_max_nrmse=0
            )
            for layer in range(3, 45)
        },
    )
    assert check_metrics(value)["real_input_shadow_checks"] == 126
    value["layers"]["8"]["real_cuda_routes"] = 0
    with pytest.raises(AssertionError, match="real CUDA routes"):
        check_metrics(value)
    del value["layers"]["8"]
    with pytest.raises(AssertionError, match="every main-model"):
        check_metrics(value)


@pytest.mark.parametrize("text,tokens", [("", 1), (" ", 1), ("hello", 0)])
def test_service_validation_requires_generated_text_and_token_usage(text, tokens):
    result = {"response": {"choices": [{"text": text}], "usage": {"completion_tokens": tokens}}}
    with pytest.raises(AssertionError, match="no generated text"):
        completion_text(result)


@pytest.mark.parametrize("require_repeat", [False, True])
def test_repeat_failure_is_retained_after_all_operational_checks(tmp_path, monkeypatch, require_repeat):
    metrics = tmp_path / "metrics.json"
    value = dict(
        cuda_worker_pid=42,
        profile="ascend-backbone-cuda-experts",
        cuda_attention=False,
        retains_npu_shadow_banks=True,
        layers={
            str(layer): dict(
                real_calls=3, real_cuda_routes=2, shadow_checks=3, shadow_max_absolute_error=0, shadow_max_nrmse=0
            )
            for layer in range(3, 45)
        },
    )
    metrics.write_text(json.dumps(value))
    repeated = []

    def fake_request(base_url, endpoint, body=None):
        if endpoint == "/v1/models":
            result = {"data": [{"id": "glm-sidecard"}]}
        elif endpoint == "/v1/chat/completions":
            result = {"choices": [{"message": {"content": "4"}}]}
        else:
            text = "hello"
            if body["prompt"] == "The capital of France is":
                repeated.append(True)
                text = "first" if len(repeated) == 1 else "changed"
            result = {
                "choices": [{"text": text}],
                "usage": {"completion_tokens": 4, "prompt_tokens": 243 if "Hello world." in body["prompt"] else 5},
            }
        return {"status": 200, "elapsed_seconds": 1, "response": result}

    class Stream:
        def __enter__(self):
            for row in value["layers"].values():
                row["real_cuda_routes"] += 1
            metrics.write_text(json.dumps(value))
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"text":"hello"}]}\n\n'
            yield b"data: [DONE]\n\n"

    monkeypatch.setattr("tools.sidecard.service_validate.request", fake_request)
    monkeypatch.setattr("tools.sidecard.service_validate.urllib.request.urlopen", lambda *args, **kwargs: Stream())
    output = tmp_path / "report.json"
    if require_repeat:
        with pytest.raises(AssertionError, match="operational requests passed"):
            validate_service("http://localhost", metrics, output, require_repeat=True)
    else:
        validate_service("http://localhost", metrics, output, require_repeat=False)
    report = json.loads(output.read_text())
    assert report["operational_passed"]
    assert not report["passed"]
    assert not report["deterministic_repeat_equal"]
    assert len(report["cases"]) == 7
    assert set(report["actual_api_cuda_route_deltas"].values()) == {1}
