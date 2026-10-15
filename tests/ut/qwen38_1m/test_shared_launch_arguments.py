# SPDX-License-Identifier: Apache-2.0
"""Compile and execute ownership tests using only the host C++ standard library."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re


def test_bridge_retains_arguments_if_command_setup_or_queue_submission_throws():
    root = Path(__file__).resolve().parents[3]
    source = (root / "tools/qwen4exp/native_owned_bridge.cpp").read_text()
    # The standalone ownership test cannot instantiate the NPU OpCommand.
    # Require its constructor, handler setup and submission to share the guard.
    assert re.search(
        r"try\s*\{\s*at_npu::native::OpCommand command;"
        r"\s*command.Name\([^;]+;\s*command.SetCustomHandler\(handler\);"
        r"\s*command.Run\(\);(?:[^{}]*?)\}\s*catch\s*\(\.\.\.\)\s*\{"
        r"(?:\s*//[^\n]*\n)*\s*kernel->RetainFailed\(owner\);\s*throw;",
        source,
    )


def test_copied_handlers_release_success_and_retain_failure(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    executable = tmp_path / "shared-launch-arguments"
    source = Path(__file__).parent / "cpp/test_shared_launch_arguments.cpp"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            f"-I{root}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize(
    "owned,version,flush", [(False, 1, False), (True, 2, False), (True, 1, False), (True, 3, True)]
)
def test_build_freezes_and_compiles_explicit_bridge(monkeypatch, tmp_path, owned, version, flush):
    import torch.utils.cpp_extension

    from tools.qwen4exp import build_route_finalize as builder

    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "build"
    calls = []
    actual_find = builder.importlib.util.find_spec
    monkeypatch.setattr(
        builder.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(origin=str(tmp_path / "torch_npu/__init__.py"))
            if name == "torch_npu"
            else actual_find(name)
        ),
    )
    monkeypatch.setattr(builder, "freeze_headers", lambda *args: {})
    monkeypatch.setattr(torch.utils.cpp_extension, "include_paths", lambda: [])
    monkeypatch.setattr(torch.utils.cpp_extension, "library_paths", lambda: [])

    def compile_only_mock(command, **kwargs):
        calls.append(command)
        destination = Path(command[command.index("-o") + 1]) if command[0] == "c++" else Path(command[2])
        destination.write_bytes(b"mock compiled artifact")

    monkeypatch.setattr(builder.subprocess, "run", compile_only_mock)
    source = root / "tools/qwen4exp/native_owned_bridge.cpp" if owned else None
    if owned and version == 1:
        with pytest.raises(ValueError, match="fresh namespace"):
            builder.build(output, tmp_path, root, tmp_path, version, source)
        assert not calls
        return
    result = builder.build(output, tmp_path, root, tmp_path, version, source, flush_host_queue=flush)
    expected_source = "native_owned_bridge.cpp" if owned else "reconstruction_bridge.cpp"
    assert result["bridge_source"] == expected_source
    assert str(output / expected_source) in calls[-1]
    assert result["namespace"] == f"qwen_route_finalize_v{version}"
    assert result["flush_host_queue"] is flush
    assert ("-DQWEN_NATIVE_FLUSH_HOST_QUEUE=1" in calls[-1]) is flush
    if owned:
        assert "shared_launch_arguments.h" in result["sources"]
        assert (output / "shared_launch_arguments.h").read_bytes() == (
            root / "tools/qwen4exp/shared_launch_arguments.h"
        ).read_bytes()
        assert not (output / "reconstruction_bridge.cpp").exists()


@pytest.mark.parametrize(
    "version,owned,flush", [(1, True, True), (2, True, True), (4, True, True), (3, False, True), (3, True, 1)]
)
def test_queue_flush_build_rejects_wrong_version_source_or_flag(monkeypatch, tmp_path, version, owned, flush):
    from tools.qwen4exp import build_route_finalize as builder

    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    root = Path(__file__).resolve().parents[3]
    source = root / "tools/qwen4exp/native_owned_bridge.cpp" if owned else None
    destination = tmp_path / "must-not-create"
    with pytest.raises(ValueError, match="boolean|version 3"):
        builder.build(destination, tmp_path, root, tmp_path, version, source, flush_host_queue=flush)
    assert not destination.exists()


def test_queue_flush_follows_submission_on_producer_only():
    root = Path(__file__).resolve().parents[3]
    source = (root / "tools/qwen4exp/native_owned_bridge.cpp").read_text()
    assert "const auto currentStream = c10_npu::getCurrentNPUStream();" in source
    assert "const auto stream = currentStream.stream(false);" in source
    handler = source.split("auto handler =", 1)[1].split("at_npu::native::OpCommand command;", 1)[0]
    assert "stream(true)" not in handler
    command = source.split("command.Run();", 1)[1].split("} catch", 1)[0]
    assert "#if defined(QWEN_NATIVE_FLUSH_HOST_QUEUE) && QWEN_NATIVE_FLUSH_HOST_QUEUE" in command
    assert "TORCH_CHECK(currentStream.stream(true) == stream," in command
    assert "#endif" in command
