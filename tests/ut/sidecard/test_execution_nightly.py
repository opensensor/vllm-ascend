# SPDX-License-Identifier: Apache-2.0
"""Authenticated gate orchestration with mocked subprocesses, never devices."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard import execution_nightly as nightly
from tools.sidecard.execution_kernel_build import digest
from tools.sidecard.execution_model_gate import canonical_sha256, config_identity


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def packet(path, value):
    return write(path, json.dumps(value, sort_keys=True) + "\n")


def reference(path):
    return dict(path=str(path), sha256=digest(path))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "phase/source"
    records = {}
    for module in (*nightly.MODULES.values(), "tools.sidecard.__init__"):
        relative = module.replace(".", "/") + ".py"
        records[relative] = digest(write(source / relative, "# authenticated frozen source fixture\n"))
    sources = packet(tmp_path / "sources.json", records)
    npu = write(tmp_path / "venv-npu/bin/python", "pinned NPU executable")
    cuda = write(tmp_path / "venv-cuda/bin/python", "pinned CUDA executable")
    npu.chmod(0o755)
    cuda.chmod(0o755)
    base = write(tmp_path / "base.so", "qualified base fixture")
    build_root = tmp_path / "native-build"
    addon = write(build_root / "addon/libsidecard_execution.so", "native addon fixture")
    vendor = build_root / "opp/packages/vendors/private-test-vendor"
    vendor.mkdir(parents=True)
    abi = dict(
        paths=dict(
            python_executable=str(npu),
            cann_root=str(tmp_path / "toolkit"),
            torch_root=str(tmp_path / "torch"),
            torch_npu_root=str(tmp_path / "torch_npu"),
        )
    )
    abi_path = packet(tmp_path / "abi.json", abi)
    env = packet(
        tmp_path / "npu-private-env.json",
        {
            "ASCEND_RT_VISIBLE_DEVICES": "2,3",
            "CUDA_VISIBLE_DEVICES": "0",
            "CANN_CUSTOM_OPP_PATH": str(vendor),
            "PYTHONPATH": "/pinned/sdk/python",
            "SDK_MARKER": "private-sdk-value",
        },
    )
    checkpoint = tmp_path / "checkpoint"
    layout = packet(checkpoint / "native-layout.json", {"identity": "checkpoint fixture"})
    native_ref = dict(root=str(build_root), manifest_sha256="a" * 64)
    native = dict(
        base_ops_library=str(base),
        addon_library="addon/libsidecard_execution.so",
        opp_path="opp/packages/vendors/private-test-vendor",
        abi_manifest_sha256=digest(abi_path),
    )
    kda = packet(
        tmp_path / "phase/kda.json",
        dict(
            library=str(addon),
            library_sha256=digest(addon),
            base_library=str(base),
            base_library_sha256=digest(base),
            physical_chips=[2, 3],
        ),
    )
    body = dict(
        schema_version=1,
        exclusive_hardware=True,
        physical_chips=[2, 3],
        cuda_device=0,
        coverage=["kernels"],
        selected_candidates=["kda_decode"],
        source=dict(root=str(source), manifest=reference(sources)),
        abi=reference(abi_path),
        npu_environment=reference(env),
        interpreters=dict(npu=reference(npu), cuda=reference(cuda)),
        native_build=native_ref,
        output=str(tmp_path / "nightly-output"),
        cases=[
            dict(name=f"kda-rank{rank}", kind="kda", config=reference(kda), output=f"kda-rank{rank}.json", rank=rank)
            for rank in (0, 1)
        ],
    )
    manifest = packet(tmp_path / "nightly.json", body)

    def authenticate_abi(path, sha):
        assert path == abi_path and sha == digest(abi_path)
        return abi

    def authenticate_native(root, sha):
        assert Path(root) == build_root and sha == native_ref["manifest_sha256"]
        return native

    monkeypatch.setattr(nightly, "authenticate_abi", authenticate_abi)
    monkeypatch.setattr(nightly, "authenticate_build", authenticate_native)
    monkeypatch.setattr(nightly, "validate_options", nightly.parse_options)
    return SimpleNamespace(
        root=tmp_path,
        manifest=manifest,
        body=body,
        source=source,
        env=env,
        abi=abi,
        checkpoint=checkpoint,
        layout=layout,
        base=base,
        npu=npu,
        native_ref=native_ref,
    )


def save(setup):
    packet(setup.manifest, setup.body)


def executor(setup, monkeypatch, *, fail=None):
    calls = []

    def execute(argv, *, cwd, env, stdout, stderr, check):
        module = argv[argv.index("-m") + 1]
        target = Path(argv[argv.index("--output") + 1])
        calls.append(dict(argv=argv, cwd=cwd, environment=env))
        assert str(cwd) == str(setup.source)
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "2,3" and env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["PYTHONPATH"] == str(cwd) + os.pathsep + "/pinned/sdk/python"
        assert stderr == subprocess.STDOUT and check is False
        stdout.write(b"gate output without private environment\n")
        if fail == "exit":
            return SimpleNamespace(returncode=7)
        if module.endswith("npu_dsa_probe"):
            result = dict(
                ok=True,
                coordinated_tp2=True,
                physical_chips=[2, 3],
                owners=[dict(report=dict(rank=rank, ok=True)) for rank in (0, 1)],
            )
        elif module.endswith("execution_model_gate"):
            variant = "candidate" if "--reference" in argv else "reference"
            result = dict(
                passed=True,
                reference_qualified=variant == "reference",
                workloads=[dict(name=name) for name in ("2K", "8K", "full-code")],
            )
        else:
            rank = int(argv[argv.index("--rank") + 1])
            result = dict(
                rank=rank, physical_chip=rank + 2, **{"ok" if module.endswith("kda_probe") else "passed": True}
            )
        if fail == "receipt":
            result["ok" if module.endswith(("kda_probe", "npu_dsa_probe")) else "passed"] = False
        if module.endswith("multi_route_probe"):
            target = target / "summary.json"
        packet(target, result)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nightly.subprocess, "run", execute)
    return calls


def model_cases(setup):
    setup.body["coverage"] = ["model"]
    contract = packet(
        setup.root / "precision.json",
        dict(
            fixed_before_native_device_tests=True,
            model_bounds={
                "same_math": {"all_tensors": "bitwise"},
                "changed_math": {
                    "floating_state": {"atol": 1e-5, "rtol": 1e-5},
                    "vocabulary": {"atol": 1e-5, "rtol": 1e-5},
                    "selectors": "bitwise",
                    "integer_boolean_state": "bitwise",
                    "nonfinite": "reject",
                    "unknown_tensor_kind": "reject",
                },
            },
        ),
    )
    fixtures = packet(
        setup.root / "fixtures.json",
        dict(
            schema_version=1,
            fixtures=[
                dict(
                    name=name,
                    request={"prompt": name},
                    request_sha256=canonical_sha256({"prompt": name}),
                    expected_prompt_tokens=tokens,
                    expected_tool_count=0,
                )
                for name, tokens in (("2K", 2048), ("8K", 8192), ("full-code", 23820))
            ],
        ),
    )
    cases = []
    for variant in ("reference", "candidate"):
        options = {} if variant == "reference" else dict(kda_decode=True, native_build=setup.native_ref)
        config_path = setup.root / "phase" / (variant + ".json")
        auth_path = setup.root / (variant + "-auth.json")
        spec = dict(
            architecture="cuda-prefix-ascend-middle-cuda-suffix",
            checkpoint=str(setup.checkpoint),
            checkpoint_sha256=digest(setup.layout),
            npu_python=str(setup.npu),
            npu_environment=str(setup.env),
            npu_ops_library=str(setup.base),
            execution_candidates=options,
            execution_diagnostics=dict(enabled=True, authentication=dict(path=str(auth_path), sha256="0" * 64)),
        )
        identity = dict(
            source_sha256=setup.body["source"]["manifest"]["sha256"],
            checkpoint_sha256=digest(setup.layout),
            native_sha256="b" * 64,
        )
        authentication = dict(
            identity=identity,
            architecture=spec["architecture"],
            config_sha256=config_identity(spec),
            files=[
                dict(
                    path=setup.body["source"]["manifest"]["path"], sha256=identity["source_sha256"], category="source"
                ),
                dict(path=str(setup.layout), sha256=digest(setup.layout), category="checkpoint"),
                dict(path=str(setup.base), sha256=digest(setup.base), category="native"),
            ],
        )
        if variant == "reference":
            evidence = packet(
                setup.root / "reference-evidence.json",
                dict(
                    passed=True,
                    default_off=True,
                    identity=identity,
                    architecture=spec["architecture"],
                    config_sha256=config_identity(spec),
                ),
            )
            authentication["qualification"] = dict(passed=True, default_off=True, evidence=reference(evidence))
        packet(auth_path, authentication)
        spec["execution_diagnostics"]["authentication"]["sha256"] = digest(auth_path)
        packet(config_path, spec)
        case = dict(
            name=variant,
            kind="model",
            variant=variant,
            config=reference(config_path),
            fixtures=reference(fixtures),
            precision_contract=reference(contract),
            output=variant + ".json",
        )
        if variant == "candidate":
            case["reference"] = "reference"
        cases.append(case)
    setup.body["cases"] = cases
    save(setup)


def test_validates_all_inputs_without_creating_outputs_or_importing_devices(setup, monkeypatch):
    monkeypatch.setattr(nightly.subprocess, "run", lambda *args, **kwargs: pytest.fail("validation executed tool"))
    result = nightly.validate_manifest(setup.manifest)
    assert len(result["cases"]) == 2 and not Path(setup.body["output"]).exists()


def test_fixed_rank_commands_preserve_lexical_venv_and_private_sdk_values(setup, monkeypatch):
    before = dict(os.environ)
    calls = executor(setup, monkeypatch)
    result = nightly.run(setup.manifest)
    assert result["passed"] is True and len(calls) == 2
    assert all(call["argv"][0] == str(setup.npu) for call in calls)
    assert [call["argv"][-1] for call in calls] == ["0", "1"]
    assert dict(os.environ) == before
    assert "private-sdk-value" not in json.dumps(result)
    with pytest.raises(ValueError, match="new directory"):
        nightly.run(setup.manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "exclusive",
        "chips",
        "cuda",
        "missing-rank",
        "duplicate-rank",
        "unknown-kind",
        "shell",
        "environment",
        "output-escape",
        "config-drift",
        "source-drift",
        "abi-drift",
        "interpreter-drift",
        "source-missing",
        "native-unused",
        "reserved-output",
    ],
)
def test_invalid_manifest_or_drift_rejected_before_any_subprocess(setup, monkeypatch, mutation):
    if mutation == "exclusive":
        setup.body["exclusive_hardware"] = False
    elif mutation == "chips":
        setup.body["physical_chips"] = [0, 1]
    elif mutation == "cuda":
        setup.body["cuda_device"] = 1
    elif mutation == "missing-rank":
        setup.body["cases"].pop()
    elif mutation == "duplicate-rank":
        setup.body["cases"][1]["rank"] = 0
    elif mutation == "unknown-kind":
        setup.body["cases"][1]["kind"] = "arbitrary_module"
    elif mutation == "shell":
        setup.body["cases"][0]["command"] = "sh -c injected"
    elif mutation == "environment":
        setup.body["env"] = {"LD_PRELOAD": "injected"}
    elif mutation == "output-escape":
        setup.body["cases"][0]["output"] = "../escape.json"
    elif mutation == "config-drift":
        write(Path(setup.body["cases"][0]["config"]["path"]), "{}")
    elif mutation == "source-drift":
        write(setup.source / "tools/sidecard/kda_probe.py", "changed")
    elif mutation == "abi-drift":
        write(Path(setup.body["abi"]["path"]), "changed")
    elif mutation == "interpreter-drift":
        write(setup.npu, "changed")
    elif mutation == "source-missing":
        records = json.loads(Path(setup.body["source"]["manifest"]["path"]).read_bytes())
        del records["tools/sidecard/kda_probe.py"]
        path = packet(setup.root / "missing-source.json", records)
        setup.body["source"]["manifest"] = reference(path)
    elif mutation == "reserved-output":
        setup.body["cases"][0]["output"] = "execution-nightly.json"
    else:
        setup.body["selected_candidates"] = ["dsa_decode_graphs"]
    save(setup)
    monkeypatch.setattr(nightly.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid input executed tool"))
    with pytest.raises(ValueError):
        nightly.run(setup.manifest)
    assert not Path(setup.body["output"]).exists()


@pytest.mark.parametrize("failure", ["exit", "receipt"])
def test_failure_records_evidence_without_launching_remaining_cases(setup, monkeypatch, failure):
    calls = executor(setup, monkeypatch, fail=failure)
    with pytest.raises((RuntimeError, ValueError)):
        nightly.run(setup.manifest)
    assert len(calls) == 1
    report = json.loads((Path(setup.body["output"]) / "execution-nightly.json").read_bytes())
    assert report["passed"] is False and report["error"]


def test_joint_dsa_gate_uses_one_command_and_checks_both_owners(setup, monkeypatch):
    setup.body.update(selected_candidates=["dsa_decode_graphs"], native_build=None)
    config = packet(
        setup.root / "dsa.json",
        dict(
            checkpoint=str(setup.checkpoint),
            checkpoint_sha256=digest(setup.layout),
            ops_library=str(setup.base),
            ops_library_sha256=digest(setup.base),
            physical_chips=[2, 3],
        ),
    )
    setup.body["cases"] = [dict(name="dsa-joint", kind="dsa", config=reference(config), output="dsa.json")]
    save(setup)
    calls = executor(setup, monkeypatch)
    assert nightly.run(setup.manifest)["passed"] and len(calls) == 1
    assert "--rank" not in calls[0]["argv"] and "--worker-rank" not in calls[0]["argv"]


def test_mhc_uses_real_rank_cli_and_native_policy_admission(setup, monkeypatch):
    setup.body["selected_candidates"] = ["mhc_pre"]
    config = packet(
        setup.root / "phase/mhc.json",
        dict(
            checkpoint=str(setup.checkpoint),
            checkpoint_sha256=digest(setup.layout),
            npu_ops_library=str(setup.base),
            execution_candidates=dict(mhc_pre_rows=[1], native_build=setup.native_ref),
        ),
    )
    setup.body["cases"] = [
        dict(name=f"mhc{rank}", kind="mhc", config=reference(config), output=f"mhc{rank}.json", rank=rank)
        for rank in (0, 1)
    ]
    save(setup)
    calls = executor(setup, monkeypatch)
    assert nightly.run(setup.manifest)["passed"]
    assert len(calls) == 2 and calls[0]["argv"][2] == "tools.sidecard.mhc_probe"


@pytest.mark.parametrize("duplicate_name", [False, True])
def test_multi_route_directory_outputs_and_clipped_override_authentication(setup, monkeypatch, duplicate_name):
    from tools.sidecard.native_multi_route import adapter

    setup.body.update(selected_candidates=["multi_route"], native_build=None)
    root = setup.root / "multi-route-build"
    candidate = dict(root=str(root), manifest_sha256="c" * 64)
    monkeypatch.setattr(
        adapter,
        "authenticate_candidate",
        lambda path, sha: (
            None if path == root and sha == candidate["manifest_sha256"] else pytest.fail("candidate identity changed")
        ),
    )
    gates = {
        name: digest(write(setup.root / "clipped" / name, name))
        for name in ("glm_fused_gate_up.bin", "glm_fused_gate_up_w4.bin")
    }
    override = packet(setup.root / "clipped/build.json", dict(native_gate_files=gates, swiglu_limit=10.0))
    config = packet(
        setup.root / "multi.json",
        dict(
            checkpoint=str(setup.checkpoint),
            checkpoint_sha256=digest(setup.layout),
            multi_route_candidate=candidate,
            native_gate_override=str(override),
        ),
    )
    setup.body["cases"] = [
        dict(
            name=f"multi{rank}",
            kind="multi_route",
            config=reference(config),
            output=f"multi{rank}",
            inputs=[reference(override)],
            rank=rank,
        )
        for rank in (0, 1)
    ]
    if duplicate_name:
        setup.body["cases"][1]["name"] = setup.body["cases"][0]["name"]
    save(setup)
    calls = executor(setup, monkeypatch)
    if duplicate_name:
        with pytest.raises(ValueError, match="nonunique"):
            nightly.run(setup.manifest)
        assert not calls
        return
    assert nightly.run(setup.manifest)["passed"] and len(calls) == 2
    assert (Path(setup.body["output"]) / "multi0/summary.json").exists()


def test_model_reference_then_candidate_uses_cuda_and_fresh_reference_digest(setup, monkeypatch):
    model_cases(setup)
    calls = executor(setup, monkeypatch)
    assert nightly.run(setup.manifest)["passed"]
    assert len(calls) == 2 and calls[0]["argv"][0] == setup.body["interpreters"]["cuda"]["path"]
    argv = calls[1]["argv"]
    ref = Path(argv[argv.index("--reference") + 1])
    assert digest(ref) == argv[argv.index("--reference-sha256") + 1]


@pytest.mark.parametrize(
    "mutation", ["order", "missing-candidate", "different-fixtures", "different-precision", "source-root"]
)
def test_model_scope_requires_prior_matched_reference_and_full_candidate_coverage(setup, monkeypatch, mutation):
    model_cases(setup)
    if mutation == "order":
        setup.body["cases"].reverse()
    elif mutation == "missing-candidate":
        setup.body["cases"].pop()
    elif mutation == "source-root":
        setup.body["cases"][1]["source"] = dict(setup.body["source"], root=str(setup.root / "other-source"))
    else:
        key = "fixtures" if mutation == "different-fixtures" else "precision_contract"
        old = Path(setup.body["cases"][1][key]["path"])
        new = write(setup.root / (key + "-other.json"), old.read_text() + "\n")
        setup.body["cases"][1][key] = reference(new)
    save(setup)
    monkeypatch.setattr(nightly.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid model executed tool"))
    with pytest.raises((ValueError, FileNotFoundError)):
        nightly.run(setup.manifest)


def test_default_nightly_collection_skips_and_help_does_not_import_accelerators():
    root = Path(__file__).resolve().parents[3]
    script = (
        "import sys; import tools.sidecard.execution_nightly; "
        "assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], cwd=root, check=True)
    result = subprocess.run(
        [sys.executable, "-m", "tools.sidecard.execution_nightly", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--validate-only" in result.stdout
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--confcutdir=tests/e2e/nightly/sidecard",
            "tests/e2e/nightly/sidecard/test_execution_kernels.py",
            "tests/e2e/nightly/sidecard/test_execution_acceptance.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "4 skipped" in result.stdout
