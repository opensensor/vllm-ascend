# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard import pair_collective_acl as module
from tools.sidecard import pair_collective_build as builder
from tools.sidecard.pair_collective import PairConfig


class Function:
    def __init__(self, name, sdk):
        self.name, self.sdk = name, sdk

    def __call__(self, *args):
        sdk, name = self.sdk, self.name
        sdk.calls.append((name, args))
        if name == "sidecard_pair_abi":
            return 2
        if name == "sidecard_pair_bindings_bytes":
            return 104
        if sdk.fail == name:
            return 77
        if name == "aclrtGetCurrentContext":
            args[0]._obj.value = sdk.context
        elif name == "aclrtDeviceGetBareTgid":
            args[0]._obj.value = 5678
        elif name in (
            "aclrtCreateStream",
            "aclrtMalloc",
            "aclrtCreateNotify",
            "aclrtIpcMemImportByKey",
            "aclrtNotifyImportByKey",
        ):
            sdk.next_pointer += 1024 * 1024
            args[0]._obj.value = sdk.next_pointer
        elif name in ("aclrtIpcMemGetExportKey", "aclrtNotifyGetExportKey"):
            args[2 if name == "aclrtIpcMemGetExportKey" else 1].value = f"key-{args[0].value}".encode()
        elif name == "sidecard_pair_create":
            args[1]._obj.value = 99
        return 0


class Sdk:
    def __init__(self):
        self.functions, self.calls = {}, []
        self.context, self.next_pointer, self.fail = 123, 1024 * 1024, None

    def __getattr__(self, name):
        if name not in self.functions:
            self.functions[name] = Function(name, self)
        return self.functions[name]


def backend(tmp_path, monkeypatch, *, local_add=None):
    paths = [tmp_path / name for name in ("acl.so", "pair.so")]
    for path in paths:
        path.write_bytes(b"private-binary-fixture")
    sdk = Sdk()
    monkeypatch.setattr(module.c, "CDLL", lambda *args, **kwargs: sdk)
    checksum = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    result = module.AclPairBackend(*paths, acl_sha256=checksum, native_sha256=checksum, local_add=local_add)
    return sdk, result


@pytest.mark.parametrize("link_mode", ("hccs", "pcie"))
def test_actual_backend_exports_with_pid_allowlists_and_distinct_import_flags(tmp_path, monkeypatch, link_mode):
    sdk, owner = backend(tmp_path, monkeypatch)
    assert owner.bare_pid() == 5678
    exported = owner.prepare(PairConfig("c" * 64, link_mode=link_mode), 4567)
    assert len(set(exported.values())) == 3
    owner.connect({"data": "peer-memory", "ready": "peer-ready", "consumed": "peer-consumed"})
    owner.enqueue(30)
    owner.synchronize()
    owner.release_imports()
    owner.close()
    calls = sdk.calls
    assert all(args[-1] == 3 for name, args in calls if name == "aclrtMalloc")
    assert all(args[-1] == 0 for name, args in calls if name.endswith("GetExportKey"))
    for name, args in calls:
        if name in ("aclrtIpcMemSetImportPid", "aclrtNotifySetImportPid"):
            assert args[1]._obj.value == 4567 and args[2] == 1
        if name == "aclrtIpcMemImportByKey":
            assert args[-1] == (1 if link_mode == "pcie" else 0)
        if name == "aclrtNotifyImportByKey":
            assert args[-1] == (2 if link_mode == "pcie" else 0)
    assert not any("Reset" in name or "Init" in name for name, _ in calls)
    assert [name for name, _ in calls].count("sidecard_pair_enqueue") == 1
    assert not owner.imports


def test_context_switch_rejected_before_native_submission(tmp_path, monkeypatch):
    sdk, owner = backend(tmp_path, monkeypatch)
    sdk.context = 999
    with pytest.raises(RuntimeError, match="context changed"):
        owner.enqueue(30)
    assert not any(name == "sidecard_pair_enqueue" for name, _ in sdk.calls)


@pytest.mark.parametrize("link_mode", ("hccs", "pcie"))
def test_ping_pong_binds_and_releases_both_inboxes_with_one_notify_pair(tmp_path, monkeypatch, link_mode):
    sdk, owner = backend(tmp_path, monkeypatch)
    keys = owner.prepare(PairConfig("c" * 64, handshake="ping-pong", link_mode=link_mode), 4567)
    assert set(keys) == {"data", "data_next", "ready", "consumed"}
    assert len(set(keys.values())) == 4 and len(owner.allocations) == 5
    with pytest.raises(ValueError, match="invalid or duplicate"):
        owner.connect({k: "peer-" + k for k in keys if k != "data_next"})
    assert not owner.imports
    owner.connect({k: "peer-" + k for k in keys})
    names = [name for name, _ in sdk.calls]
    assert names.count("sidecard_pair_enable_ping_pong") == 1
    assert names.count("aclrtCreateNotify") == 2
    imports = [args for name, args in sdk.calls if name == "aclrtIpcMemImportByKey"]
    assert len(imports) == 2 and all(args[-1] == (1 if link_mode == "pcie" else 0) for args in imports)
    owner.synchronize()
    owner.release_imports()
    owner.close()
    closed = [args[0] for name, args in sdk.calls if name == "aclrtIpcMemClose"]
    assert closed == [b"peer-data", b"peer-data_next", keys["data"].encode(), keys["data_next"].encode()]
    assert [name for name, _ in sdk.calls].count("aclrtFree") == 5


def test_ping_pong_rejects_local_callback_before_resource_creation(tmp_path, monkeypatch):
    spec = local_add_spec(tmp_path)
    sdk, owner = backend(tmp_path, monkeypatch, local_add=spec)
    with pytest.raises(ValueError, match="ACLNN"):
        owner.prepare(PairConfig("c" * 64, handshake="ping-pong", reducer_identity=spec.identity), 4567)
    assert not owner.allocations and not any(name == "aclrtCreateStream" for name, _ in sdk.calls)


def test_partial_setup_failure_retains_created_allocation_owners(tmp_path, monkeypatch):
    sdk, owner = backend(tmp_path, monkeypatch)
    sdk.fail = "aclrtIpcMemSetImportPid"
    with pytest.raises(RuntimeError, match="77"):
        owner.prepare(PairConfig("c" * 64), 4567)
    assert len(owner.allocations) == 4 and owner.export_key is not None
    assert not any(name == "aclrtFree" for name, _ in sdk.calls)


def test_duplicate_peer_keys_rejected_before_import(tmp_path, monkeypatch):
    sdk, owner = backend(tmp_path, monkeypatch)
    owner.prepare(PairConfig("c" * 64), 4567)
    with pytest.raises(ValueError, match="distinct"):
        owner.connect(dict.fromkeys(("data", "ready", "consumed"), "same-key"))
    assert not owner.imports
    assert not any(name.endswith("ImportByKey") for name, _ in sdk.calls)


@pytest.mark.parametrize("ping_pong", (False, True))
def test_memory_and_notify_keys_use_separate_namespaces(tmp_path, monkeypatch, ping_pong):
    sdk, owner = backend(tmp_path, monkeypatch)
    options = {"handshake": "ping-pong"} if ping_pong else {}
    owner.prepare(PairConfig("c" * 64, **options), 4567)
    keys = dict(data="first", ready="first", consumed="second")
    if ping_pong:
        keys["data_next"] = "second"
    owner.connect(keys)
    owner.synchronize()
    owner.release_imports()
    owner.close()
    calls = [name for name, _ in sdk.calls]
    assert calls.count("aclrtIpcMemImportByKey") == (2 if ping_pong else 1)
    assert calls.count("aclrtNotifyImportByKey") == 2


def test_duplicate_ping_pong_memory_keys_rejected_before_import(tmp_path, monkeypatch):
    sdk, owner = backend(tmp_path, monkeypatch)
    owner.prepare(PairConfig("c" * 64, handshake="ping-pong"), 4567)
    with pytest.raises(ValueError, match="distinct"):
        owner.connect(dict(data="same", data_next="same", ready="ready", consumed="consumed"))
    assert not any(name.endswith("ImportByKey") for name, _ in sdk.calls)


def test_binary_mismatch_fails_before_loading(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(module.c, "CDLL", lambda *args, **kwargs: called.append(args))
    with pytest.raises(ValueError, match="SHA256"):
        module.AclPairBackend(tmp_path / "missing", tmp_path / "missing", acl_sha256="a", native_sha256="b")
    assert not called


def local_add_spec(tmp_path):
    values = {}
    for name in ("library", "binary"):
        path = tmp_path / ("local-add-" + name)
        path.write_bytes(("authenticated " + name).encode())
        values[name], values[name + "_sha256"] = str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    return module.LocalAddSpec(**values)


def test_remote_read_rejected_before_any_stream_allocation_export_or_native_tensor_creation(tmp_path, monkeypatch):
    sdk, owner = backend(tmp_path, monkeypatch)
    before = list(sdk.calls)
    with pytest.raises(ValueError, match="remote vector reads are disabled"):
        owner.prepare(PairConfig("c" * 64, exchange_mode="remote-read"), 4567)
    assert sdk.calls == before
    assert owner.config is None and not owner.allocations and not owner.notifications and owner.stream.value is None


@pytest.mark.parametrize("selected_local_add", (False, True))
def test_reducer_identity_mismatch_rejected_before_any_resource_allocation(tmp_path, monkeypatch, selected_local_add):
    spec = local_add_spec(tmp_path) if selected_local_add else None
    sdk, owner = backend(tmp_path, monkeypatch, local_add=spec)
    before = list(sdk.calls)
    wrong_identity = "aclnn" if selected_local_add else "a" * 64
    with pytest.raises(ValueError, match="reducer differs"):
        owner.prepare(PairConfig("c" * 64, reducer_identity=wrong_identity), 4567)
    assert sdk.calls == before and owner.config is None and not owner.allocations


@pytest.mark.parametrize("rows", (2, 3, 31))
def test_local_add_fixed_shape_rejected_before_any_resource_allocation(tmp_path, monkeypatch, rows):
    spec = local_add_spec(tmp_path)
    sdk, owner = backend(tmp_path, monkeypatch, local_add=spec)
    before = list(sdk.calls)
    with pytest.raises(ValueError, match="rows"):
        owner.prepare(PairConfig("c" * 64, rows=rows, reducer_identity=spec.identity), 4567)
    assert sdk.calls == before and owner.config is None and not owner.allocations


@pytest.mark.parametrize("changed", ("library", "binary"))
def test_local_add_drift_reauthenticated_before_any_resource_allocation(tmp_path, monkeypatch, changed):
    spec = local_add_spec(tmp_path)
    sdk, owner = backend(tmp_path, monkeypatch, local_add=spec)
    Path(getattr(spec, changed)).write_bytes(b"changed after backend construction")
    before = list(sdk.calls)
    with pytest.raises(ValueError, match="SHA256"):
        owner.prepare(PairConfig("c" * 64, reducer_identity=spec.identity), 4567)
    assert sdk.calls == before and owner.config is None and not owner.allocations


def test_local_add_attach_failure_retains_imports_buffers_and_partial_wrapper(tmp_path, monkeypatch):
    spec = local_add_spec(tmp_path)
    sdk, owner = backend(tmp_path, monkeypatch, local_add=spec)

    class PartialAdd:
        def __init__(self, selected, backend):
            assert selected is spec and backend is owner

        def attach(self):
            raise RuntimeError("partial native add handle retained")

        def close(self):
            pytest.fail("failed setup must not automatically unload native code")

    monkeypatch.setattr(module, "LocalAdd", PartialAdd)
    owner.prepare(PairConfig("c" * 64, reducer_identity=spec.identity), 4567)
    with pytest.raises(RuntimeError, match="partial native add"):
        owner.connect({"data": "peer-memory", "ready": "peer-ready", "consumed": "peer-consumed"})
    assert isinstance(owner.local_add, PartialAdd) and owner.handle.value == 99
    assert len(owner.allocations) == 4 and set(owner.imports) == {"data", "ready", "consumed"}
    assert not any(name in ("aclrtFree", "sidecard_pair_destroy", "aclrtDestroyStream") for name, _ in sdk.calls)


def test_local_add_unloads_after_pair_and_peer_import_release_before_buffer_free(tmp_path, monkeypatch):
    spec = local_add_spec(tmp_path)
    sdk, owner = backend(tmp_path, monkeypatch, local_add=spec)

    class LocalCode:
        def __init__(self, selected, backend):
            assert selected is spec and backend is owner

        def attach(self):
            sdk.calls.append(("local-add-attach", ()))

        def close(self):
            assert not owner.imports
            sdk.calls.append(("local-add-unload", ()))

    monkeypatch.setattr(module, "LocalAdd", LocalCode)
    owner.prepare(PairConfig("c" * 64, reducer_identity=spec.identity), 4567)
    owner.connect({"data": "peer-memory", "ready": "peer-ready", "consumed": "peer-consumed"})
    with pytest.raises(RuntimeError, match="mappings remain live"):
        owner.close()
    assert not any(name == "local-add-unload" for name, _ in sdk.calls)
    owner.synchronize()
    owner.release_imports()
    owner.close()
    names = [name for name, _ in sdk.calls]
    assert names.index("sidecard_pair_destroy") < names.index("local-add-unload") < names.index("aclrtFree")
    peer_close = next(
        i for i, (name, args) in enumerate(sdk.calls) if name == "aclrtIpcMemClose" and args == (b"peer-memory",)
    )
    assert peer_close < names.index("local-add-unload")


def test_local_add_unload_failure_retains_buffer_and_notification_owners(tmp_path, monkeypatch):
    sdk, owner = backend(tmp_path, monkeypatch)
    owner.prepare(PairConfig("c" * 64), 4567)
    owner.connect({"data": "peer-memory", "ready": "peer-ready", "consumed": "peer-consumed"})
    owner.release_imports()

    def failed_unload():
        raise RuntimeError("native code unload failed; retain owner")

    owner.local_add = SimpleNamespace(close=failed_unload)
    with pytest.raises(RuntimeError, match="unload failed"):
        owner.close()
    assert len(owner.allocations) == 4 and len(owner.notifications) == 2
    assert not any(name in ("aclrtFree", "aclrtDestroyStream") for name, _ in sdk.calls)


@pytest.mark.parametrize("returncode", (0, 1))
def test_private_builder_freezes_inputs_and_records_failed_build_without_device_calls(
    tmp_path, monkeypatch, returncode
):
    sdk = tmp_path / "sdk"
    for rel in (
        "lib64/libascendcl.so",
        "lib64/libopapi.so",
        "lib64/libnnopbase.so",
        "include/acl/acl_rt.h",
        "include/aclnnop/aclnn_add.h",
        "include/aclnn/acl_meta.h",
    ):
        path = sdk / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"authenticated sdk fixture")
    compiler = tmp_path / "compiler"
    compiler.write_bytes(b"compiler identity")
    monkeypatch.setattr(builder.shutil, "which", lambda _: str(compiler))
    commands = []

    def compile_only(command, **kwargs):
        commands.append(command)
        if returncode == 0:
            Path(command[-1]).write_bytes(b"compiled library fixture")
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(builder.subprocess, "run", compile_only)
    output = tmp_path / "output"
    if returncode:
        with pytest.raises(RuntimeError, match="build failed"):
            builder.build(sdk, output)
    else:
        builder.build(sdk, output)
    receipt = json.loads((output / "build.json").read_text())
    assert receipt["build_complete"] is (returncode == 0)
    assert receipt["hardware_validated"] is False
    assert set(receipt["source_sha256"]) == {"pair_sum.cpp", "pair_sum.h"}
    assert len(commands) == 1 and "-Wl,-z,defs" in commands[0]
    with pytest.raises(FileExistsError):
        builder.build(sdk, output)
