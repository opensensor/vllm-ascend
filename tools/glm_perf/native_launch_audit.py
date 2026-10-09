# SPDX-License-Identifier: Apache-2.0
"""Identify native launches without reading route data or device tensors."""

import hashlib

KERNEL_FILES = (
    ("pack_kernel", "glm_fused_pack.bin"),
    ("route_input_kernel", "glm_fused_route_input.bin"),
    ("gate_kernel", "glm_fused_gate_up.bin"),
    ("down_kernel", "glm_fused_down.bin"),
    ("gate_w3_kernel", "glm_fused_gate_up_w3.bin"),
    ("down_w3_kernel", "glm_fused_down_w3.bin"),
    ("gate_w4_kernel", "glm_fused_gate_up_w4.bin"),
    ("down_w4_kernel", "glm_fused_down_w4.bin"),
    ("reduce_kernel", "glm_fused_reduce.bin"),
)


class NativeLaunchAudit:
    """Attach only while paused; hashes are computed before measured requests.

    Counts cover actual host submissions to the captured resource's bridge.
    They exclude preparation when ``begin_requests`` starts the live interval.
    They are not device completion times or instruction-level profiler traces.
    """

    def __init__(self, native, root, log_first=None):
        original = getattr(native.launch, "__native_launch_original__", native.launch)
        self.operator = str(original)
        self.active = False
        self.counts = {}
        self.binaries = {}
        labels = {}
        for attribute, filename in KERNEL_FILES:
            kernel = getattr(native, attribute, None)
            if kernel is not None:
                labels[id(kernel)] = attribute
                path = root / filename
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                self.binaries[attribute] = {"path": str(path), "sha256": digest}

        def launch(kernel, tensors, cores):
            label = labels.get(id(kernel))
            if label is None:
                raise ValueError("launch does not belong to the audited native resource")
            result = original(kernel, tensors, cores)
            if self.active:
                self.counts[label] = self.counts.get(label, 0) + 1
                if self.counts[label] == 1 and log_first is not None:
                    log_first(
                        "GLM native launch: operator=%s kernel=%s binary_sha256=%s",
                        self.operator,
                        label,
                        self.binaries[label]["sha256"],
                    )
            return result

        launch.__native_launch_original__ = original
        native.launch = launch

    def begin_requests(self):
        if not self.active:
            self.counts.clear()
            self.active = True

    def snapshot(self):
        return {
            "operator": self.operator,
            "counts": dict(self.counts),
            "binaries": self.binaries,
            "active": self.active,
        }
