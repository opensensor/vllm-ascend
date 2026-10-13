# SPDX-License-Identifier: Apache-2.0
"""Keep optional FLA operators out of unrelated model import failures."""

from importlib import import_module


def load_fla_op(name: str):
    try:
        module = import_module("fla_npu.ops.ascendc")
    except ModuleNotFoundError as error:
        # A broken dependency inside an installed backend is a real error;
        # only absence of the optional backend itself permits a light import.
        if error.name is None or error.name.split(".")[0] != "fla_npu":
            raise

        def unavailable(*args, **kwargs):
            raise RuntimeError(
                f"FLA operator {name!r} requires the optional fla_npu backend. "
                "Select the native Ascend 310P implementation for supported models, "
                "or install a backend matching the selected hardware and runtime."
            )

        return unavailable
    return getattr(module, name)
