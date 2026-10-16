# SPDX-License-Identifier: Apache-2.0
"""Private, opt-in one-row KDA recurrence contract; no accelerator import.

The native operator mutates FP32 carry in place and returns FP16 output. It
receives original FP16 q/k, rather than prematurely rounded normalized inputs.
Registration is deliberately left to the integration task.
"""

from dataclasses import dataclass

HEAD_DIM = 128
VALUE_GROUP = 16
L2_EPS = 1e-6
OPERATOR_NAME = "sidecard_kda_decode_310"
OPERATOR_NAMESPACE = "_C_ascend"
SCHEMA = (
    "sidecard_kda_decode_310(Tensor query, Tensor key, Tensor value, "
    "Tensor gate, Tensor beta, Tensor(a!) state) -> Tensor"
)


@dataclass(frozen=True)
class NumericalBounds:
    """Proposed bounds, fixed before device qualification; not measured results."""

    state_atol: float = 5e-5
    state_rtol: float = 5e-5
    output_atol: float = 2e-3
    output_rtol: float = 2e-3
    sequential_state_atol: float = 1e-3
    sequential_state_rtol: float = 1e-3


BOUNDS = NumericalBounds()


def validate_inputs(query, key, value, gate, beta, state, torch):
    shape = tuple(query.shape)
    if len(shape) != 3 or shape[0] != 1 or shape[2] != HEAD_DIM:
        raise ValueError("native KDA requires one row [1, H, 128]")
    heads = shape[1]
    if not 8 <= heads <= 128 or heads % 8:
        raise ValueError("native KDA requires 8..128 heads, divisible by eight")
    specs = (
        (query, shape, torch.float16),
        (key, shape, torch.float16),
        (value, shape, torch.float16),
        (gate, shape, torch.float32),
        (beta, (1, heads), torch.float32),
        (state, (heads, HEAD_DIM, HEAD_DIM), torch.float32),
    )
    for tensor, expected_shape, dtype in specs:
        if tuple(tensor.shape) != expected_shape or tensor.dtype != dtype:
            raise ValueError("native KDA input shape/dtype violates the FP32 recurrence contract")
        if tensor.device != query.device or not tensor.is_contiguous():
            raise ValueError("native KDA inputs must be contiguous on the same device")
    state_begin = state.data_ptr()
    state_end = state_begin + state.numel() * state.element_size()
    for tensor in (query, key, value, gate, beta):
        begin = tensor.data_ptr()
        end = begin + tensor.numel() * tensor.element_size()
        if begin < state_end and state_begin < end:
            raise ValueError("native KDA read-only inputs must not overlap mutable carry")
    return heads


def reference(query, key, value, gate, beta, state, torch):
    """Independent boundary reference; mutates carry exactly as the runtime does."""
    validate_inputs(query, key, value, gate, beta, state, torch)
    q = query[0].float()
    k = key[0].float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + L2_EPS)
    q = q * (HEAD_DIM**-0.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + L2_EPS)
    state.mul_(gate[0].exp()[:, None, :])
    prediction = (state * k[:, None, :]).sum(-1)
    innovation = (value[0].float() - prediction) * beta[0, :, None]
    state.add_(innovation[:, :, None] * k[:, None, :])
    return (state * q[:, None, :]).sum(-1).half().unsqueeze(0)


def resolve_operator(torch):
    """Resolve the existing custom-op namespace without importing device modules."""
    try:
        namespace = getattr(torch.ops, OPERATOR_NAMESPACE)
        return getattr(namespace, OPERATOR_NAME)
    except AttributeError as error:
        raise RuntimeError("private _C_ascend::sidecard_kda_decode_310 operator is not registered") from error


def native(query, key, value, gate, beta, state, torch, *, operator=None):
    """Dispatch only when explicitly installed; never silently downcast or fall back."""
    validate_inputs(query, key, value, gate, beta, state, torch)
    if query.device.type != "npu":
        raise ValueError("native KDA dispatch requires NPU inputs")
    if operator is None:
        operator = resolve_operator(torch)
    return operator(query, key, value, gate, beta, state)
