# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline candidate: submit qualified TP2/TP4/TP6 QFN draft before host bookkeeping.

Only the pinned runner's existing early-PP predicate changes. Its source hash
and exact seam must match; all unsupported requests retain original ordering.
"""

import hashlib
import inspect
import textwrap
from types import FunctionType

TARGET = "vllm_ascend.worker.model_runner_v1:NPUModelRunner.sample_tokens"
QUALIFIED_SOURCE_SHA256 = "25dc0ac0921d87f03c8bb1974972b73b91c5acf868b1b014185410b6eb837dbc"
QUALIFIED_MODEL_MODULE = "vllm_ascend.models.qwen4_exp.model"
QUALIFIED_MODEL_CLASS = "AscendQwen4ExpForCausalLM"
QUALIFIED_CONDITIONAL_CLASS = "AscendQwen4ExpForConditionalGeneration"
CONDITION = """        early_pp_padded_drafter = (
            use_pp_spec_decode
            and not self.use_async_scheduling
            and use_padded_batch
        )"""
REPLACEMENT = """        early_pp_padded_drafter = (
            (use_pp_spec_decode or _early_tp_allowed(self, spec_decode_metadata, batch_desc, scheduler_output))
            and not self.use_async_scheduling
            and use_padded_batch
        )"""


def eligible(runner, spec_metadata, batch_descriptor, scheduler):
    """Host-only admission; exclude every dependency not audited for this run."""
    config = getattr(runner, "speculative_config", None)
    parallel = getattr(runner, "parallel_config", None)
    drafter = getattr(runner, "drafter", None)
    if (
        config is None
        or getattr(config, "method", None) != "mtp"
        or getattr(runner, "num_spec_tokens", None) != 2
        or getattr(scheduler, "num_spec_tokens_to_schedule", None) != 2
        or getattr(runner, "use_async_scheduling", True)
        or getattr(config, "disable_padded_drafter_batch", True)
        or getattr(parallel, "tensor_parallel_size", None) not in (2, 4, 6)
        or getattr(parallel, "pipeline_parallel_size", None) != 1
        or getattr(parallel, "data_parallel_size", None) != 1
        or getattr(runner, "use_dcp", True)
        or getattr(runner, "num_discarded_requests", None) != 0
        or getattr(runner, "num_prompt_logprobs", True)
        or spec_metadata is None
        or not getattr(batch_descriptor, "uniform", False)
        or getattr(drafter, "_enable_probabilistic_draft_probs", True)
    ):
        return False
    model = runner.get_model()
    if type(model).__module__ == QUALIFIED_MODEL_MODULE and type(model).__name__ == QUALIFIED_CONDITIONAL_CLASS:
        model = getattr(model, "language_model", None)
    if type(model).__module__ != QUALIFIED_MODEL_MODULE or type(model).__name__ != QUALIFIED_MODEL_CLASS:
        return False
    return getattr(getattr(model, "model", None), "_mtp_hidden_buffer", None) is not None


def build_forward(original, *, source=None):
    """Compile the one reviewed change while preserving live module globals."""
    original = getattr(original, "_qfn_early_tp_base", original)
    source = textwrap.dedent(inspect.getsource(original)) if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_SOURCE_SHA256:
        raise RuntimeError("early TP draft candidate requires the exact qualified sample_tokens source")
    decorator = "@torch.inference_mode()\n"
    if not source.startswith(decorator) or source.count(CONDITION) != 1:
        raise RuntimeError("qualified early-PP source seam did not match")
    body = source[len(decorator) :].replace(CONDITION, REPLACEMENT)
    # Close over admission instead of adding mutable names to runner globals.
    factory_source = "def factory(_early_tp_allowed):\n" + textwrap.indent(body, "    ") + "    return sample_tokens\n"
    unwrapped = inspect.unwrap(original)
    namespace = dict(unwrapped.__globals__)
    exec(compile(factory_source, "<qfn-qualified-early-tp-draft>", "exec"), namespace)
    compiled = namespace["factory"](eligible)
    forward = FunctionType(
        compiled.__code__, unwrapped.__globals__, compiled.__name__, compiled.__defaults__, compiled.__closure__
    )
    forward.__annotations__ = compiled.__annotations__
    forward.__kwdefaults__ = compiled.__kwdefaults__
    forward.__qualname__ = original.__qualname__
    forward = unwrapped.__globals__["torch"].inference_mode()(forward)
    forward._qfn_early_tp_base = original
    return forward


def replacements():
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    return {TARGET: build_forward(NPUModelRunner.sample_tokens)}
