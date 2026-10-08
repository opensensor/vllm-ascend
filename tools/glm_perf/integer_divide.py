# SPDX-License-Identifier: Apache-2.0
"""Private exact metadata division; keep unsupported PyTorch calls unchanged."""

import ast
import types

import torch

DIVISORS = (4, 160, 640)
DIVISION_CORES = 8
DIVISION_TILE = 64
DMA_BYTES = 32


def prepare_counts(native, counts):
    """Allocate immutable launch metadata outside all serving graph pools."""
    counts = tuple(counts)
    if any(type(count) is not int or count <= 0 for count in counts):
        raise ValueError("prepared integer counts must be positive host integers")
    counts = tuple(sorted(set(counts)))
    keys = [
        (count, divisor, dtype)
        for count in counts
        for divisor in DIVISORS
        for dtype in (torch.int32, torch.int64)
        if (count, divisor, dtype) not in native.configs
    ]
    if not keys:
        return
    # One small transfer while paused, rather than device-fill producers
    # inside the first graph to encounter each descriptor shape.
    values = torch.tensor([(count, divisor, dtype.itemsize) for count, divisor, dtype in keys], dtype=torch.int64).to(
        native.device
    )
    for index, key in enumerate(keys):
        native.configs[key] = values[index]


class NativeIntegerDivide:
    def __init__(self, build, namespace):
        self.device = torch.device("npu", torch.npu.current_device())
        torch.empty(1, device=self.device)
        self.kernel = getattr(torch.classes, namespace).Kernel(
            str(build / "glm_integer_divide.bin"), "glm_integer_divide_v1"
        )
        self.launch = getattr(torch.ops, namespace).launch
        self.configs = {}
        self.calls = 0

    def __call__(self, value, divisor):
        if value.device != self.device or value.dtype not in (torch.int32, torch.int64) or divisor not in DIVISORS:
            raise ValueError("native division requires same-device INT32/INT64 and a qualified divisor")
        # The final DMA block has explicitly owned padding, including tiny
        # decode shapes; no store crosses this allocation's physical span.
        lanes = DMA_BYTES // value.element_size()
        padded = (value.numel() + lanes - 1) // lanes * lanes
        backing = torch.empty(padded, device=self.device, dtype=value.dtype)
        output = backing[: value.numel()].reshape(value.shape)
        if not value.numel():
            return output
        key = (value.numel(), divisor, value.dtype)
        if key not in self.configs:
            # Capture-safe device fills: no CPU tensor copy in serving.
            config = torch.empty(3, device=self.device, dtype=torch.int64)
            for slot, number in enumerate((value.numel(), divisor, value.element_size())):
                config[slot].fill_(number)
            self.configs[key] = config
        cores = min(DIVISION_CORES, (value.numel() + DIVISION_TILE - 1) // DIVISION_TILE)
        self.launch(self.kernel, [value.contiguous(), output, self.configs[key]], cores)
        self.calls += 1
        return output


class DivisionTorch:
    """Delegate every other attribute to the unchanged PyTorch module."""

    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(torch, name)

    def div(self, value, divisor, *, rounding_mode=None, out=None):
        if (
            out is None
            and rounding_mode == "floor"
            and isinstance(value, torch.Tensor)
            and value.dtype in (torch.int32, torch.int64)
            and value.device == self.native.device
            and type(divisor) is int
            and divisor in DIVISORS
        ):
            return self.native(value, divisor)
        return torch.div(value, divisor, rounding_mode=rounding_mode, out=out)

    def remainder(self, value, divisor):
        if (
            isinstance(value, torch.Tensor)
            and value.dtype in (torch.int32, torch.int64)
            and value.device == self.native.device
            and type(divisor) is int
            and divisor in DIVISORS
        ):
            return value - self.native(value, divisor) * divisor
        return torch.remainder(value, divisor)


def private_divisions(original, proxy):
    """Clone a function's globals; preserve code, defaults and closure values."""
    namespace = dict(original.__globals__, torch=proxy)
    clone = types.FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    clone.__kwdefaults__ = original.__kwdefaults__
    clone.__dict__.update(original.__dict__)
    return clone


def rewrite_pool_remainders(original, authoritative_source, proxy):
    """Reuse slot quotients in the private, permanently bound pool writer.

    Read the deployed authoritative source because native writers were built
    with exec and have no inspectable source file. Reject changed expressions
    instead of silently installing a partially rewritten method.
    """
    tree = ast.parse(authoritative_source)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SparseAttnIndexerKpool")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_write_pools")
    source = ast.unparse(method)
    replacements = (
        ("safe_state_slots % pool_size", "(safe_state_slots - state_blocks * pool_size)"),
        ("pool_slots % block_size", "(pool_slots - pool_blocks * block_size)"),
        ("torch.div(pool_slots, block_size, rounding_mode='floor')", "pool_blocks"),
        (
            "block_size = key_cache.shape[1]",
            "block_size = key_cache.shape[1]\n"
            "    pool_blocks = torch.div(pool_slots, block_size, rounding_mode='floor')",
        ),
        ("(positions + 1) % pool_size", "_integer_remainder(positions + 1, pool_size)"),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("authoritative pool writer expressions changed")
        source = source.replace(before, after)
    namespace = dict(original.__globals__, torch=proxy, _integer_remainder=proxy.remainder)
    if "_aicore_convert" in namespace:
        # Retain the same permanent converter, its descriptor cache, and
        # BF16 rounding directly into the existing FP16 key cache.
        source = source.replace(
            "compress_kpool(pool_keys, pool_gates, ape)",
            "compress_kpool(pool_keys, pool_gates, ape, out_dtype=key_cache.dtype)",
        ).replace("compressed.to(key_cache.dtype)", "_aicore_convert(compressed, key_cache.dtype)")
    exec(compile(source, "<pool-writer-native-remainders>", "exec"), namespace)
    result = namespace["_write_pools"]
    result.__glm_remainder_original__ = original
    return result
