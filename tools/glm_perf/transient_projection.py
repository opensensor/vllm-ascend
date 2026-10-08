# SPDX-License-Identifier: Apache-2.0
"""Release experimental projection scratch after retiring the old graphs."""


def release_projection_capture_buffers(resources, retained_operators):
    """Call only while idle, after graph retirement and before the next capture.

    Release all projection scratch, including active scratch from retired pools.
    Keep registered kernels, weights and active prepared descriptors intact.
    NativeFusedMoE objects in frozen helper packages have distinct class objects,
    so match their class name and the two explicit transient buffer dictionaries.
    """
    retained = {id(operator) for operator in retained_operators}
    seen = set()
    counts = dict(operators=0, scratch_entries=0, config_entries=0)

    def visit(value):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif type(value).__name__ == "NativeFusedMoE":
            state = vars(value)
            buffers = [state.get(name) for name in ("scratch", "configs")]
            if not all(isinstance(buffer, dict) for buffer in buffers):
                raise ValueError("native projection transient buffer contract changed")
            counts["operators"] += 1
            for name, buffer in zip(("scratch_entries", "config_entries"), buffers):
                if name == "config_entries" and identity in retained:
                    continue
                counts[name] += len(buffer)
                buffer.clear()

    visit(resources)
    return counts
