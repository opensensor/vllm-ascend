# SPDX-License-Identifier: Apache-2.0
"""Authentication and ownership tests for the optional local reducer wrapper."""

import ctypes as c
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard import pair_add as module


def specification(tmp_path, *, blocks=1):
    values = {}
    for name in ("library", "binary"):
        path = tmp_path / name
        path.write_bytes(("authenticated local " + name).encode())
        values[name] = str(path)
        values[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return module.LocalAddSpec(**values, blocks=blocks)


@pytest.mark.parametrize("blocks", (1, 2, 4, 8))
def test_authenticated_spec_grid_and_peer_identity_are_immutable(tmp_path, blocks):
    spec = specification(tmp_path, blocks=blocks)
    expected = hashlib.sha256(
        json.dumps(
            dict(
                library=spec.library_sha256,
                binary=spec.binary_sha256,
                blocks=blocks,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert spec.identity == expected
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.blocks = 2


@pytest.mark.parametrize("blocks", (True, False, 0, 3, 5, 9, 1.0, "1", None))
def test_invalid_grid_never_coerces(tmp_path, blocks):
    with pytest.raises(ValueError, match="grid"):
        specification(tmp_path, blocks=blocks)


@pytest.mark.parametrize(
    "field,value",
    (
        ("library", "relative.so"),
        ("binary", "relative.bin"),
        ("library", None),
        ("binary", 1),
        ("library_sha256", None),
        ("binary_sha256", True),
        ("library_sha256", "0" * 64),
        ("binary_sha256", "a"),
    ),
)
def test_invalid_path_or_checksum_rejected(tmp_path, field, value):
    spec = specification(tmp_path)
    values = dataclasses.asdict(spec)
    values[field] = value
    with pytest.raises(ValueError):
        module.LocalAddSpec(**values)


@pytest.mark.parametrize("name", ("library", "binary"))
def test_source_drift_fails_authentication_before_library_loading(tmp_path, monkeypatch, name):
    spec = specification(tmp_path)
    Path(getattr(spec, name)).write_bytes(b"changed after spec admission")
    loaded = []
    monkeypatch.setattr(module.c, "CDLL", lambda *args: loaded.append(args))
    with pytest.raises(ValueError, match="SHA256"):
        module.LocalAdd(spec, None)
    assert not loaded


def test_identity_tracks_grid_and_content_without_binding_host_paths(tmp_path):
    first = specification(tmp_path)
    other = tmp_path / "another-host"
    other.mkdir()
    second = specification(other)
    assert first.identity == second.identity
    assert dataclasses.replace(first, blocks=2).identity != first.identity
    Path(second.binary).write_bytes(b"another authenticated device kernel")
    changed = dataclasses.replace(second, binary_sha256=hashlib.sha256(Path(second.binary).read_bytes()).hexdigest())
    assert changed.identity != first.identity


class Function:
    def __init__(self, operation):
        self.operation = operation

    def __call__(self, *args):
        return self.operation(*args)


class Library:
    def __init__(self):
        self.calls, self.abi, self.create_error, self.destroy_error = [], 1, 0, 0
        self.sidecard_add_abi = Function(lambda: self.abi)
        self.sidecard_add_create = Function(self.create)
        self.sidecard_add_destroy = Function(self.destroy)
        # Real ctypes function pointer permits testing the real callback cast.
        self.sidecard_add_enqueue = c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p)(lambda handle, stream: 0)

    def create(self, *args):
        self.calls.append(("create", args))
        args[-1]._obj.value = 0xADD0  # Retain a partial handle even on SDK failure.
        return self.create_error

    def destroy(self, *args):
        self.calls.append(("destroy", args))
        if not self.destroy_error:
            args[0]._obj.value = None
        return self.destroy_error


def wrapper(tmp_path, monkeypatch, *, rows=1):
    spec, library, calls = specification(tmp_path, blocks=4), Library(), []
    setter = Function(lambda *args: 0)
    owner = SimpleNamespace(
        config=SimpleNamespace(rows=rows, elements=rows * 4096),
        input=c.c_void_p(0x1000),
        inbox=c.c_void_p(0x2000),
        output=c.c_void_p(0x3000),
        handle=c.c_void_p(0x4000),
        native=SimpleNamespace(sidecard_pair_set_local_add=setter),
        fail=None,
    )

    def call(name, *args):
        calls.append((name, args))
        if owner.fail == name:
            raise RuntimeError("owner context or native callback failure")
        if name == "aclrtGetCurrentContext":
            args[0]._obj.value = 0x5000

    owner._call = call
    monkeypatch.setattr(module.c, "CDLL", lambda path: library)
    return spec, library, owner, calls


@pytest.mark.parametrize("rows", (1, 4, 32))
def test_real_wrapper_binds_exact_abi_and_local_fixed_buffers_then_registers_callback(tmp_path, monkeypatch, rows):
    spec, library, owner, calls = wrapper(tmp_path, monkeypatch, rows=rows)
    add = module.LocalAdd(spec, owner)
    assert library.sidecard_add_abi.argtypes == [] and library.sidecard_add_abi.restype is c.c_uint64
    assert library.sidecard_add_enqueue.argtypes == [c.c_void_p, c.c_void_p]
    assert library.sidecard_add_destroy.argtypes == [c.POINTER(c.c_void_p), c.c_uint32]
    assert len(library.sidecard_add_create.argtypes) == 8
    add.attach()
    name, args = library.calls[0]
    assert name == "create" and args[0] is add.image
    assert args[1:4] == (len(add.image) - 1, rows * 4096, 4)
    assert all(a is b for a, b in zip(args[4:7], (owner.input, owner.inbox, owner.output), strict=True))
    assert add.handle.value == 0xADD0
    assert calls[0][0] == "sidecard_pair_set_local_add"
    assert calls[0][1][0] is owner.handle and calls[0][1][1] is add.handle
    assert calls[0][1][2].value == c.cast(library.sidecard_add_enqueue, c.c_void_p).value
    assert owner.native.sidecard_pair_set_local_add.argtypes == [c.c_void_p] * 3


def test_wrapper_rejects_wrong_abi_before_create_or_callback(tmp_path, monkeypatch):
    spec, library, owner, calls = wrapper(tmp_path, monkeypatch)
    library.abi = 2
    with pytest.raises(ValueError, match="ABI"):
        module.LocalAdd(spec, owner)
    assert not library.calls and not calls


@pytest.mark.parametrize("rows", (2, 3, 31))
def test_unsupported_shape_does_not_call_native_create(tmp_path, monkeypatch, rows):
    spec, library, owner, calls = wrapper(tmp_path, monkeypatch, rows=rows)
    add = module.LocalAdd(spec, owner)
    with pytest.raises(ValueError, match="fixed binaries"):
        add.attach()
    assert not library.calls and not calls and add.handle.value is None


def test_partial_create_and_setter_failure_retain_code_handle_without_automatic_unload(tmp_path, monkeypatch):
    spec, library, owner, calls = wrapper(tmp_path, monkeypatch)
    add = module.LocalAdd(spec, owner)
    library.create_error = 310010
    with pytest.raises(RuntimeError, match="310010.*retain loaded code"):
        add.attach()
    assert add.handle.value == 0xADD0 and add.image is not None and not calls
    assert [name for name, _ in library.calls] == ["create"]
    library.create_error, owner.fail = 0, "sidecard_pair_set_local_add"
    # A distinct wrapper models a separate quarantined setup attempt.
    another = module.LocalAdd(spec, owner)
    with pytest.raises(RuntimeError, match="callback failure"):
        another.attach()
    assert another.handle.value == 0xADD0
    assert all(name != "destroy" for name, _ in library.calls)


def test_explicit_close_checks_owner_context_and_passes_exact_drained_proof(tmp_path, monkeypatch):
    spec, library, owner, calls = wrapper(tmp_path, monkeypatch)
    add = module.LocalAdd(spec, owner)
    add.attach()
    calls.clear()
    owner.fail = "aclrtGetCurrentContext"
    with pytest.raises(RuntimeError, match="context"):
        add.close()
    assert [name for name, _ in library.calls] == ["create"] and add.handle.value == 0xADD0
    owner.fail = None
    library.destroy_error = 310011
    with pytest.raises(RuntimeError, match="310011.*retain loaded code"):
        add.close()
    assert add.handle.value == 0xADD0 and library.calls[-1][1][1] == 1
    assert add.library is library and add.image is not None
    library.destroy_error = 0
    add.close()
    assert add.handle.value is None
    assert calls[-1][0] == "aclrtGetCurrentContext"
