# SPDX-License-Identifier: Apache-2.0

import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard.pcie.peer import (
    ASCEND_TO_CUDA,
    CUDA_TO_ASCEND,
    EXECUTE,
    VERSION,
    DmaBuffer,
    Driver,
    KernelPeer,
    QuarantinedDMA,
    Request,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("before", [0, 1])
def test_imported_alias_checks_sync_memops_once_before_use(before):
    driver = object.__new__(Driver)
    state, calls = [before], []

    def call(name, data, attribute, pointer):
        assert attribute == 6 and pointer == 12345
        calls.append(name)
        value = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint))
        if name == "cuPointerSetAttribute":
            state[0] = value[0]
        else:
            value[0] = state[0]

    driver.call = call
    allocation = endpoint(SimpleNamespace(kind="cuda"))
    assert driver.ensure_sync_memops(allocation) is bool(before)
    assert state == [1]
    assert calls == (
        ["cuPointerGetAttribute"]
        if before
        else ["cuPointerGetAttribute", "cuPointerSetAttribute", "cuPointerGetAttribute"]
    )


def endpoint(runtime, *, size=65536, host=False):
    return SimpleNamespace(runtime=runtime, size=size, host=host, pointer=12345, leases=0)


def scenario(tmp_path, mutate=None):
    cuda, ascend = SimpleNamespace(kind="cuda"), SimpleNamespace(kind="ascend")
    calls = []
    gpu = SimpleNamespace(fd=4, allocation=endpoint(cuda), driver=SimpleNamespace(flush=lambda: calls.append("flush")))
    npu, guard = endpoint(ascend), endpoint(ascend, host=True)

    def ioctl(fd, command, body, mutable):
        assert command == EXECUTE and mutable
        value = Request.from_buffer_copy(body)
        assert value.version == VERSION and value.physical_chip == 2
        assert value.guard_va == guard.pointer and value.pin_length == npu.size
        calls.append("dma")
        value.cache_id, value.gpu_segments, value.gpu_mapped_bytes = 2, 1, gpu.allocation.size
        if mutate is not None:
            mutate(value)
        body[:] = bytes(value)

    path = tmp_path / "device"
    path.write_bytes(b"")
    peer = KernelPeer(2, path, ioctl=ioctl)

    def event(runtime):
        return SimpleNamespace(
            runtime=runtime, leases=0, require_recorded=lambda: None, wait=lambda: calls.append("ready")
        )

    return peer, gpu, npu, guard, event, calls


@pytest.mark.parametrize("operation", [CUDA_TO_ASCEND, ASCEND_TO_CUDA])
def test_source_event_and_gpu_flush_order_and_lifetimes(tmp_path, operation):
    peer, gpu, npu, guard, event, calls = scenario(tmp_path)
    ready = event(gpu.allocation.runtime if operation == CUDA_TO_ASCEND else npu.runtime)
    result = peer.copy(operation, gpu, npu, guard, 8256, source_ready=ready)
    assert calls == (["ready", "dma"] if operation == CUDA_TO_ASCEND else ["ready", "dma", "flush"])
    assert result["length"] == 8256 and "npu_va" not in result
    assert all(resource.leases == 0 for resource in (gpu.allocation, npu, guard, ready))
    peer.close()
    peer.close()


def test_persistent_mapping_retains_gpu_owner_until_clean_peer_close(tmp_path):
    peer, gpu, npu, guard, event, calls = scenario(tmp_path)
    peer.persistent_gpu_mappings = True
    for _ in range(3):
        peer.copy(CUDA_TO_ASCEND, gpu, npu, guard, 8256, source_ready=event(gpu.allocation.runtime))
        assert gpu.allocation.leases == 1
        assert npu.leases == guard.leases == 0
        assert peer.exports == [gpu]
    peer.close()
    assert gpu.allocation.leases == 0 and peer.exports == []


def test_persistent_mapping_rejection_still_retains_owner_until_peer_close(tmp_path):
    peer, gpu, npu, guard, event, _ = scenario(tmp_path, lambda value: setattr(value, "map_status", -22))
    peer.persistent_gpu_mappings = True
    with pytest.raises(RuntimeError, match="rejected"):
        peer.copy(CUDA_TO_ASCEND, gpu, npu, guard, 8256, source_ready=event(gpu.allocation.runtime))
    assert gpu.allocation.leases == 1
    peer.close()
    assert gpu.allocation.leases == 0


def test_persistent_mapping_fault_preserves_owner_and_active_leases(tmp_path):
    peer, gpu, npu, guard, event, _ = scenario(tmp_path, lambda value: setattr(value, "retained", 1))
    peer.persistent_gpu_mappings = True
    with pytest.raises(QuarantinedDMA):
        peer.copy(CUDA_TO_ASCEND, gpu, npu, guard, 8256, source_ready=event(gpu.allocation.runtime))
    assert gpu.allocation.leases == 2 and npu.leases == guard.leases == 1
    with pytest.raises(QuarantinedDMA):
        peer.close()
    os.close(peer.fd)  # Fake device only.


def test_persistent_mapping_capacity_rejects_third_export_before_submission(tmp_path):
    peer, gpu, npu, guard, event, calls = scenario(tmp_path)
    peer.persistent_gpu_mappings = True
    for _ in range(2):
        gpu = SimpleNamespace(**vars(gpu))
        gpu.allocation = endpoint(gpu.allocation.runtime)
        peer.copy(CUDA_TO_ASCEND, gpu, npu, guard, 8256, source_ready=event(gpu.allocation.runtime))
    before = list(calls)
    with pytest.raises(ValueError, match="mapping capacity"):
        peer.copy(
            CUDA_TO_ASCEND, SimpleNamespace(**vars(gpu)), npu, guard, 8256, source_ready=event(gpu.allocation.runtime)
        )
    assert calls == before
    owners = [exported.allocation for exported in peer.exports]
    peer.close()
    assert all(owner.leases == 0 for owner in owners)


@pytest.mark.parametrize("size", [0, -1, 65537, True, 0.5])
def test_bounds_fail_before_event_or_ioctl(tmp_path, size):
    peer, gpu, npu, guard, event, calls = scenario(tmp_path)
    with pytest.raises(ValueError, match="capacity"):
        peer.copy(CUDA_TO_ASCEND, gpu, npu, guard, size, source_ready=event(gpu.allocation.runtime))
    assert calls == []
    peer.close()


def test_wrong_guard_and_source_event_are_rejected(tmp_path):
    peer, gpu, npu, guard, event, calls = scenario(tmp_path)
    with pytest.raises(ValueError, match="readiness"):
        peer.copy(CUDA_TO_ASCEND, gpu, npu, guard, 64, source_ready=event(npu.runtime))
    guard.host = False
    with pytest.raises(ValueError, match="guard"):
        peer.copy(CUDA_TO_ASCEND, gpu, npu, guard, 64, source_ready=event(gpu.allocation.runtime))
    assert calls == []
    peer.close()


def test_setup_rejection_releases_leases_without_claiming_dma(tmp_path):
    peer, gpu, npu, guard, event, _ = scenario(tmp_path, lambda value: setattr(value, "pin_status", -22))
    ready = event(gpu.allocation.runtime)
    with pytest.raises(RuntimeError, match="rejected"):
        peer.copy(CUDA_TO_ASCEND, gpu, npu, guard, 64, source_ready=ready)
    assert not peer.failed
    assert all(resource.leases == 0 for resource in (gpu.allocation, npu, guard, ready))
    peer.close()


@pytest.mark.parametrize("failure", ["retained", "identity", "geometry", "syscall"])
def test_uncertain_completion_preserves_native_lifetimes(tmp_path, failure):
    def mutate(value):
        if failure == "retained":
            value.retained, value.dma_status = 1, -110
        elif failure == "identity":
            value.length += 1
        elif failure == "geometry":
            value.gpu_mapped_bytes = 0
        else:
            raise OSError("response copy failed")

    peer, gpu, npu, guard, event, calls = scenario(tmp_path, mutate)
    ready = event(npu.runtime)
    with pytest.raises((QuarantinedDMA, OSError)):
        peer.copy(ASCEND_TO_CUDA, gpu, npu, guard, 64, source_ready=ready)
    assert peer.failed and peer.retained is not None and "flush" not in calls
    assert all(resource.leases == 1 for resource in (gpu.allocation, npu, guard, ready))
    with pytest.raises(QuarantinedDMA):
        peer.close()
    os.close(peer.fd)  # Fake test device only; real failed workers retain this fd.


def test_owner_and_assigned_card_checks(tmp_path):
    for chip in (0, 1, True):
        with pytest.raises(ValueError, match="restricted"):
            KernelPeer(chip, tmp_path / "missing")
    peer, _, _, _, _, _ = scenario(tmp_path)
    peer.pid += 1
    with pytest.raises(RuntimeError, match="creating process"):
        peer.check_owner()
    os.close(peer.fd)


def test_compiled_kernel_uapi_matches_python_layout(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler is required for kernel UAPI qualification")
    source = tmp_path / "abi.c"
    source.write_text(
        '#include <stddef.h>\n#include <stdio.h>\n#include "sidecard_peer.h"\n'
        'int main(void) { printf("{\\"size\\":%zu,\\"command\\":%lu,\\"guard\\":%zu,'
        '\\"retained\\":%zu}", sizeof(struct sc_peer_request), (unsigned long)SC_PEER_EXECUTE, '
        "offsetof(struct sc_peer_request, guard_va), offsetof(struct sc_peer_request, retained)); return 0; }\n"
    )
    binary = tmp_path / "abi"
    subprocess.run(
        [compiler, "-I", str(ROOT / "tools/sidecard/pcie/kernel"), str(source), "-o", str(binary)], check=True
    )
    actual = json.loads(subprocess.check_output([str(binary)], text=True))
    assert actual == dict(
        size=ctypes.sizeof(Request), command=EXECUTE, guard=Request.guard_va.offset, retained=Request.retained.offset
    )


def test_export_requires_dedicated_gpu_memory_and_rdma_capability():
    runtime = SimpleNamespace(kind="cuda")
    allocation = SimpleNamespace(runtime=runtime, host=False, imported=True, pointer=12345, size=65536, leases=0)
    driver = object.__new__(Driver)
    with pytest.raises(ValueError, match="owned dedicated"):
        driver.export(allocation)
    allocation.imported = False

    def call(name, pointer, attribute, address):
        assert name == "cuPointerGetAttribute" and address == allocation.pointer
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint))[0] = 2 if attribute == 2 else 0

    driver.call = call
    with pytest.raises(ValueError, match="RDMA-capable"):
        driver.export(allocation)
    allocation.leases = 1
    with pytest.raises(ValueError, match="already leased"):
        driver.export(allocation)


def test_dma_buf_close_preserves_exported_and_active_allocation_lifetimes():
    read_fd, write_fd = os.pipe()
    allocation = SimpleNamespace(runtime=SimpleNamespace(check_owner=lambda: None), leases=0)
    exported = DmaBuffer(None, allocation, read_fd)
    assert allocation.leases == 1
    allocation.leases += 1
    with pytest.raises(RuntimeError, match="held by an active"):
        exported.close()
    os.fstat(read_fd)
    allocation.leases -= 1
    exported.close()
    exported.close()
    assert allocation.leases == 0
    os.close(write_fd)
