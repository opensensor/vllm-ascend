# SPDX-License-Identifier: Apache-2.0
"""Portable strict/lazy execution-candidate admission and prefill-tail checks."""

import builtins
import copy
import dataclasses
import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools.sidecard import execution_options as options

CHECKSUM = "a" * 64
REFERENCE = {"root": "/absolute/build", "manifest_sha256": CHECKSUM}


def native_spec(**candidates):
    return {
        "npu_ops_library": "/absolute/qualified-ops.so",
        "execution_candidates": {"native_build": dict(REFERENCE), **candidates},
    }


def test_default_options_are_immutable_and_accelerator_import_free(monkeypatch):
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if any(part in name for part in ("torch", "execution_kernel_build", "native_multi_route", "npu_mhc")):
            raise AssertionError("baseline attempted candidate/device import: " + name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    for spec in (
        {},
        {"execution_candidates": {}},
        {"npu_ops_library": "unused-path", "execution_candidates": {"multi_route": None}},
    ):
        parsed = options.validate_options(spec)
        assert not parsed.any_enabled and not parsed.native_required
        assert options.load_native(spec, None) == (parsed, None)
        assert options.make_mhc(parsed, None) is None
        assert options.make_multi_route(None, parsed) is None
        with pytest.raises(dataclasses.FrozenInstanceError):
            parsed.kda_decode = True


@pytest.mark.parametrize("raw", (None, False, [], "on", {"unknown": True}))
def test_option_object_is_strict(raw):
    with pytest.raises(ValueError, match="known options"):
        options.parse_options({"execution_candidates": raw})


@pytest.mark.parametrize("name", ("kda_decode", "dsa_decode_graphs"))
@pytest.mark.parametrize("value", (None, 0, 1, "true", [], {}))
def test_booleans_never_coerce(name, value):
    with pytest.raises(ValueError, match="boolean"):
        options.parse_options({"execution_candidates": {name: value}})


@pytest.mark.parametrize("name", ("mhc_pre_rows", "mhc_post_rows"))
@pytest.mark.parametrize("value", (None, (), {}, "1", [True], [1.0], [0], [33], [1, 1]))
def test_row_lists_reject_invalid_types_ranges_and_duplicates(name, value):
    with pytest.raises(ValueError, match=name):
        options.parse_options(native_spec(**{name: value}))


def test_parser_copies_rows_and_references_without_hashing_nonexistent_builds():
    spec = native_spec(kda_decode=True, mhc_pre_rows=[1, 4], mhc_post_rows=[1, 32])
    parsed = options.parse_options(spec)
    spec["execution_candidates"]["mhc_pre_rows"].append(8)
    spec["execution_candidates"]["native_build"]["root"] = "/another/build"
    assert parsed.mhc_pre_rows == frozenset((1, 4)) and parsed.native_build.root == Path("/absolute/build")
    assert options.required_operators(parsed) == {
        "sidecard_kda_decode_310",
        "sidecard_mhc_pre_310",
        "sidecard_mhc_post_310",
    }


@pytest.mark.parametrize("name", ("multi_route", "native_build"))
@pytest.mark.parametrize(
    "reference",
    (
        False,
        {},
        {"root": "/build"},
        {"root": "/build", "manifest_sha256": CHECKSUM, "extra": 1},
        {"root": "relative", "manifest_sha256": CHECKSUM},
        {"root": Path("/build"), "manifest_sha256": CHECKSUM},
        {"root": "/build", "manifest_sha256": "A" * 64},
        {"root": "/build", "manifest_sha256": "0" * 63},
        {"root": "/build", "manifest_sha256": True},
        {"root": "/build\0evil", "manifest_sha256": CHECKSUM},
    ),
)
def test_build_reference_admission_is_strict(name, reference):
    with pytest.raises(ValueError):
        options.parse_options({"execution_candidates": {name: reference}})


@pytest.mark.parametrize("candidate", ({"kda_decode": True}, {"mhc_pre_rows": [1]}, {"mhc_post_rows": [32]}))
def test_native_candidates_need_build_and_qualified_base_library(candidate):
    with pytest.raises(ValueError, match="require native_build"):
        options.parse_options({"execution_candidates": candidate})
    with pytest.raises(ValueError, match="npu_ops_library"):
        options.parse_options({"execution_candidates": {**candidate, "native_build": REFERENCE}})
    assert options.parse_options(native_spec(**candidate)).native_required


def test_unused_native_build_is_rejected():
    with pytest.raises(ValueError, match="unused native_build"):
        options.parse_options(native_spec())
    with pytest.raises(ValueError, match="unused native_build"):
        options.parse_options(native_spec(multi_route=REFERENCE))


@pytest.mark.parametrize("candidate", ({"kda_decode": True}, {"mhc_pre_rows": [1]}, {"mhc_post_rows": [1]}))
def test_expert_only_rejects_unused_block_math(candidate):
    spec = native_spec(**candidate)
    spec["architecture"] = options.EXPERT_ARCHITECTURE
    with pytest.raises(ValueError, match="multi_route only"):
        options.parse_options(spec)


def test_dsa_requires_block_and_decode_graphs_true():
    spec = {"execution_candidates": {"dsa_decode_graphs": True}}
    for invalid in (None, False, 1, "true"):
        spec["npu_decode_graphs"] = invalid
        with pytest.raises(ValueError, match="npu_decode_graphs"):
            options.parse_options(spec)
    spec["npu_decode_graphs"] = True
    assert options.parse_options(spec).dsa_decode_graphs
    spec["architecture"] = options.EXPERT_ARCHITECTURE
    with pytest.raises(ValueError, match="multi_route only"):
        options.parse_options(spec)


@pytest.mark.parametrize("architecture", (options.BLOCK_ARCHITECTURE, options.EXPERT_ARCHITECTURE))
def test_multi_route_supported_for_both_architectures(architecture):
    assert options.parse_options(
        {"architecture": architecture, "execution_candidates": {"multi_route": REFERENCE}}
    ).multi_route


def installed_build(tmp_path, monkeypatch):
    addon, opp, base = tmp_path / "addon.so", tmp_path / "opp", tmp_path / "base.so"
    addon.write_bytes(b"private addon fixture")
    opp.mkdir()
    base.write_bytes(b"qualified base fixture")
    manifest = dict(
        schema_version=1,
        kind="sidecard-execution-build-v1",
        build_complete=True,
        addon_library="addon.so",
        opp_path="opp",
        base_ops_library=str(base),
        operators=["SidecardKdaV310", "SidecardMhcPreV310", "SidecardMhcPostV310"],
        torch_operators=["sidecard_kda_decode_310", "sidecard_mhc_pre_310", "sidecard_mhc_post_310"],
    )
    calls = []
    addon_hash = hashlib.sha256(addon.read_bytes()).hexdigest()

    def authenticate(root, checksum):
        calls.append((root, checksum))
        if checksum != CHECKSUM or hashlib.sha256(addon.read_bytes()).hexdigest() != addon_hash:
            raise ValueError("build receipt/payload drift")
        return copy.deepcopy(manifest)

    fake = ModuleType("tools.sidecard.execution_kernel_build")
    fake.authenticate_build = authenticate
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    spec = native_spec(kda_decode=True, mhc_pre_rows=[1], mhc_post_rows=[1, 4])
    spec["npu_ops_library"] = str(base)
    spec["native_opp_path"] = str(opp)
    spec["execution_candidates"]["native_build"]["root"] = str(tmp_path)
    return spec, manifest, calls, addon


def test_validation_reauthenticates_and_loads_only_private_addon(tmp_path, monkeypatch):
    spec, _, calls, addon = installed_build(tmp_path, monkeypatch)
    loaded = []
    operator = lambda *args: None
    torch_stub = SimpleNamespace(
        ops=SimpleNamespace(
            load_library=loaded.append,
            _C_ascend=SimpleNamespace(
                sidecard_kda_decode_310=operator,
                sidecard_mhc_pre_310=operator,
                sidecard_mhc_post_310=operator,
            ),
        )
    )
    parsed = options.validate_options(spec)
    assert len(calls) == 1 and not loaded
    assert options.load_native(spec, torch_stub)[0] == parsed
    assert len(calls) == 2 and loaded == [str(addon)]
    addon.write_bytes(b"changed after successful admission")
    with pytest.raises(ValueError, match="drift"):
        options.load_native(spec, torch_stub)
    assert len(loaded) == 1


@pytest.mark.parametrize(
    "field,value",
    (
        ("build_complete", False),
        ("schema_version", 2),
        ("schema_version", True),
        ("kind", "other"),
        ("torch_operators", []),
        ("torch_operators", ["sidecard_kda_decode_310"]),
        ("torch_operators", ["sidecard_kda_decode_310", "sidecard_kda_decode_310"]),
        ("addon_library", "../escape.so"),
        ("addon_library", "/absolute/addon.so"),
        ("opp_path", "../opp"),
        ("opp_path", "missing"),
        ("addon_library", "missing.so"),
    ),
)
def test_native_receipt_structure_and_payload_paths_fail_closed(tmp_path, monkeypatch, field, value):
    spec, manifest, _, _ = installed_build(tmp_path, monkeypatch)
    manifest[field] = value
    with pytest.raises(ValueError):
        options.validate_options(spec)


def test_native_build_must_use_same_base_file_and_private_opp_metadata(tmp_path, monkeypatch):
    spec, _, _, _ = installed_build(tmp_path, monkeypatch)
    other = tmp_path / "other.so"
    other.write_bytes(b"different base")
    spec["npu_ops_library"] = str(other)
    with pytest.raises(ValueError, match="base ops library differs"):
        options.validate_options(spec)
    spec["npu_ops_library"] = str(tmp_path / "base.so")
    spec["native_opp_path"] = str(tmp_path / "different-opp")
    with pytest.raises(ValueError, match="OPP launch metadata"):
        options.validate_options(spec)


def test_multiroute_reauthenticates_on_every_validation(tmp_path, monkeypatch):
    from tools.sidecard.native_multi_route import adapter

    calls = []
    monkeypatch.setattr(adapter, "authenticate_candidate", lambda *args: calls.append(args))
    spec = {"execution_candidates": {"multi_route": {"root": str(tmp_path), "manifest_sha256": CHECKSUM}}}
    options.validate_options(spec)
    options.validate_options(spec)
    assert calls == [(tmp_path, CHECKSUM)] * 2


def test_mhc_factory_uses_only_explicit_rows_and_selected_ops():
    pre, sinkhorn = lambda *args: None, lambda *args: None
    parsed = options.parse_options(native_spec(mhc_pre_rows=[1, 4]))
    stub = SimpleNamespace(
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(sidecard_mhc_pre_310=pre, mhc_sinkhorn_310=sinkhorn))
    )
    mhc = options.make_mhc(parsed, stub)
    assert mhc.pre_rows == frozenset((1, 4)) and not mhc.post_rows
    assert mhc.pre_op is pre and mhc.sinkhorn is sinkhorn
    with pytest.raises(RuntimeError, match="unselected"):
        mhc.post_op()


@pytest.mark.parametrize("rows", range(2, 17))
def test_multiroute_factory_and_prefill_tails_preserve_original_padding(tmp_path, monkeypatch, rows):
    from tools.sidecard.native_multi_route import adapter

    calls = []

    class Owner:
        stable_rows = 17

        def __call__(self, layer, x, ids, weights):
            calls.append((layer, x.shape[0], self.stable_rows))
            return "qualified padded tail"

    actor = Owner()
    actor.native = object()
    original_native = actor.native
    real_adapter = adapter.MultiRouteDecode

    def make(owner, root, checksum):
        assert owner is actor and root == tmp_path and checksum == CHECKSUM
        candidate = object.__new__(real_adapter)
        candidate.actor = owner
        return candidate

    monkeypatch.setattr(adapter, "MultiRouteDecode", make)
    parsed = options.parse_options(
        {"execution_candidates": {"multi_route": {"root": str(tmp_path), "manifest_sha256": CHECKSUM}}}
    )
    candidate = options.make_multi_route(actor, parsed)
    # Invoke the real adapter method: every tail bypasses candidate native and
    # enters the exact same original actor with its deterministic padding intact.
    x, ids, weights = torch.zeros((rows, 4096)), torch.zeros((rows, 8)), torch.zeros((rows, 8))
    assert candidate(18, x, ids, weights) == "qualified padded tail"
    assert calls == [(18, rows, 17)] and actor.stable_rows == 17 and actor.native is original_native


def test_multiroute_factory_rejects_globally_disabled_padding(monkeypatch):
    from tools.sidecard.native_multi_route import adapter

    monkeypatch.setattr(
        adapter, "MultiRouteDecode", lambda *args: pytest.fail("must reject before candidate construction")
    )
    parsed = options.parse_options({"execution_candidates": {"multi_route": REFERENCE}})
    with pytest.raises(ValueError, match="17-row padding"):
        options.make_multi_route(SimpleNamespace(stable_rows=1), parsed)


def test_selected_native_and_multiroute_receipts_are_both_rechecked(tmp_path, monkeypatch):
    from tools.sidecard.native_multi_route import adapter

    spec, _, calls, _ = installed_build(tmp_path, monkeypatch)
    spec["npu_decode_graphs"] = True
    spec["execution_candidates"].update(multi_route=REFERENCE, dsa_decode_graphs=True)
    checked = []
    monkeypatch.setattr(adapter, "authenticate_candidate", lambda *args: checked.append(args))
    parsed = options.validate_options(spec)
    assert len(calls) == len(checked) == 1
    assert parsed.kda_decode and parsed.dsa_decode_graphs and parsed.multi_route is not None


def test_private_addon_registration_failure_does_not_fallback(tmp_path, monkeypatch):
    spec, _, _, _ = installed_build(tmp_path, monkeypatch)
    loaded = []
    stub = SimpleNamespace(ops=SimpleNamespace(load_library=loaded.append, _C_ascend=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="not registered"):
        options.load_native(spec, stub)
    assert len(loaded) == 1


def test_bad_multiroute_receipt_propagates_before_any_native_load(monkeypatch):
    from tools.sidecard.native_multi_route import adapter

    def reject(*args):
        raise ValueError("candidate receipt drift")

    monkeypatch.setattr(adapter, "authenticate_candidate", reject)
    spec = {"execution_candidates": {"multi_route": REFERENCE}}
    with pytest.raises(ValueError, match="receipt drift"):
        options.load_native(spec, None)


def test_base_library_hardlink_is_same_identity(tmp_path, monkeypatch):
    spec, _, _, _ = installed_build(tmp_path, monkeypatch)
    link = tmp_path / "base-link.so"
    link.hardlink_to(tmp_path / "base.so")
    spec["npu_ops_library"] = str(link)
    assert options.validate_options(spec).native_required
