# SPDX-License-Identifier: Apache-2.0
"""Real Linux containment tests using ordinary files, never NPU driver probes."""

import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def helper(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    executable = tmp_path_factory.mktemp("sandbox-helper") / "sandbox"
    subprocess.run(
        [
            compiler,
            "-static",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(ROOT / "tools/qwen4exp/host_compile_sandbox.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    probe = subprocess.run([str(executable), "--probe"], capture_output=True, text=True)
    if probe.returncode == 125 and "Landlock ABI >=3 unavailable" in probe.stderr:
        pytest.skip(f"kernel Landlock unsupported: {probe.stderr.strip()}")
    assert probe.returncode == 0, probe.stderr
    assert json.loads(probe.stdout)["landlock_abi"] >= 3
    return executable


def command(helper, allowed, script, *args, writable=True):
    roots = ["/usr", "/lib", "/lib64", "/etc"]
    argv = [str(helper)]
    for root in roots:
        if Path(root).exists():
            argv += ["--allow-read", root]
    argv += [
        "--allow-write" if writable else "--allow-read",
        str(allowed),
        "--allow-write",
        "/dev/null",
        "--",
        "/usr/bin/python3",
        "-I",
        "-c",
        script,
        *map(str, args),
    ]
    return argv


def run(helper, allowed, script, *args, writable=True, **kwargs):
    result = subprocess.run(
        command(helper, allowed, script, *args, writable=writable), capture_output=True, text=True, **kwargs
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stderr.splitlines()[0])
    assert receipt["restricted"] and receipt["nonstandard_fds_closed"]
    assert receipt["driver_ioctls_and_sockets_denied"]
    return json.loads(result.stdout)


def test_real_allowed_read_write_exec_and_denied_outside_file(helper, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "input").write_text("source fixture")
    forbidden = tmp_path / "outside"
    forbidden.write_text("not permitted")
    script = """
import errno,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); denied=pathlib.Path(sys.argv[2])
assert (root/'input').read_text()=='source fixture'
(root/'result').write_text('compiled fixture')
try:
    denied.read_text()
except OSError as error:
    assert error.errno==errno.EACCES
else:
    raise AssertionError('outside path was readable')
print(json.dumps({'allowed_write':True,'denied_read':True}))
"""
    assert run(helper, allowed, script, allowed, forbidden) == {"allowed_write": True, "denied_read": True}
    assert (allowed / "result").read_text() == "compiled fixture"


def test_symlink_alias_cannot_escape_allowed_root(helper, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    denied = tmp_path / "outside"
    denied.write_text("outside")
    alias = allowed / "davinci_manager-alias-fixture"
    alias.symlink_to(denied)
    script = """
import errno,json,pathlib,sys
try:
    pathlib.Path(sys.argv[1]).read_bytes()
except OSError as error:
    assert error.errno==errno.EACCES
else:
    raise AssertionError('symlink alias escaped rules')
print(json.dumps({'alias_denied':True}))
"""
    assert run(helper, allowed, script, alias)["alias_denied"]


def test_nonstandard_inherited_fd_is_closed_before_child(helper, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("inherited secret fixture")
    fd = os.open(outside, os.O_RDONLY)
    try:
        script = """
import errno,json,os,sys
try:
    os.read(int(sys.argv[1]),1)
except OSError as error:
    assert error.errno==errno.EBADF
else:
    raise AssertionError('inherited nonstandard FD survived')
print(json.dumps({'fd_closed':True}))
"""
        assert run(helper, allowed, script, fd, pass_fds=(fd,))["fd_closed"]
    finally:
        os.close(fd)


def test_read_only_roots_cannot_write_or_truncate(helper, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "read_only"
    target.write_text("keep these bytes")
    script = """
import errno,json,os,pathlib,sys
target=pathlib.Path(sys.argv[1]); assert target.read_text()=='keep these bytes'
for action in (lambda: target.write_text('overwrite'),lambda: os.truncate(target,0)):
    try:
        action()
    except OSError as error:
        assert error.errno==errno.EACCES
    else:
        raise AssertionError('read-only path mutated')
print(json.dumps({'write_and_truncate_denied':True}))
"""
    assert run(helper, allowed, script, target, writable=False)["write_and_truncate_denied"]
    assert target.read_text() == "keep these bytes"


def test_driver_daemon_socket_and_fd_transfer_paths_denied(helper, tmp_path):
    script = """
import errno,json,socket
for action in (lambda:socket.socket(socket.AF_UNIX),lambda:socket.socket(socket.AF_INET),lambda:socket.socketpair()):
    try:
        action()
    except OSError as error:
        assert error.errno==errno.EACCES
    else:
        raise AssertionError('socket creation was allowed')
print(json.dumps({'socket_paths_denied':True}))
"""
    assert run(helper, tmp_path, script)["socket_paths_denied"]


@pytest.mark.parametrize("root", ["/", "/dev", "/srv", "/home", "/root", "/run", "/var"])
def test_broad_roots_fail_closed_before_child(helper, root):
    result = subprocess.run([str(helper), "--allow-read", root, "--", "/usr/bin/true"], capture_output=True, text=True)
    assert result.returncode == 125 and "allow root rejected" in result.stderr


def test_allowed_root_symlink_resolves_before_broad_path_check(helper, tmp_path):
    alias = tmp_path / "alias-root"
    alias.symlink_to("/")
    result = subprocess.run(
        [str(helper), "--allow-read", str(alias), "--", "/usr/bin/true"], capture_output=True, text=True
    )
    assert result.returncode == 125 and "allow root rejected" in result.stderr


def test_explicit_safe_device_exception_and_ioctl_filter(helper, tmp_path):
    # /dev/null is an explicit harmless exception. No NPU driver is touched.
    script = """
import errno,fcntl,json,os
fd=os.open('/dev/null',os.O_RDWR)
assert os.read(fd,1)==b''
assert os.write(fd,b'x')==1
try:
    fcntl.ioctl(fd,0)
except OSError as error:
    assert error.errno==errno.EACCES
else:
    raise AssertionError('driver ioctl was permitted')
os.close(fd)
print(json.dumps({'safe_device_read_write':True,'ioctl_denied':True}))
"""
    assert run(helper, tmp_path, script)["ioctl_denied"]


def test_socket_as_standard_fd_rejected_without_child(helper, tmp_path):
    first, second = socket.socketpair()
    try:
        result = subprocess.run(
            command(helper, tmp_path, "raise AssertionError('child executed')"),
            stdin=first,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 125 and "inherited as standard" in result.stderr
    finally:
        first.close()
        second.close()


def test_missing_or_unallowed_exec_fails_closed(helper, tmp_path):
    result = subprocess.run([str(helper), "--", "/usr/bin/true"], capture_output=True, text=True)
    assert result.returncode == 125 and "explicit allow roots" in result.stderr
    result = subprocess.run(
        [str(helper), "--allow-read", str(tmp_path), "--", "/usr/bin/true"], capture_output=True, text=True
    )
    assert result.returncode == 125 and "exec failed" in result.stderr


def test_static_bootstrap_and_clean_child_environment(helper, tmp_path):
    dynamic = subprocess.run(["readelf", "-l", str(helper)], capture_output=True, text=True, check=True)
    assert "INTERP" not in dynamic.stdout
    script = """
import json,os
assert 'LD_PRELOAD' not in os.environ
assert 'LD_AUDIT' not in os.environ
assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']=='0'
print(json.dumps({'clean_child_environment':True}))
"""
    environment = dict(
        os.environ, LD_PRELOAD="/nonexistent-preload-fixture.so", LD_AUDIT="/nonexistent-audit-fixture.so"
    )
    assert run(helper, tmp_path, script, env=environment)["clean_child_environment"]
