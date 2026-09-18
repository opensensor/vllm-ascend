# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""System-One structured-decision runtime (host-side, CPU-only, Triton-free).

This package hosts the OSS System-One runtime: given ``(context, output schema)``
it emits a typed value with per-field calibrated confidence in a single NPU
forward, abstaining to the W2 MoE (System-Two) when unsure.

Only the on-card execution paths depend on ``torch_npu``. Everything in this
subpackage is designed to import and unit-test host-side with no NPU, no Triton,
and no heavy vLLM machinery. ``schema_ir`` in particular is pure-Python.
"""
