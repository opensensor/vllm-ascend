# SPDX-License-Identifier: Apache-2.0
"""Source-checked early draft with a separate output-observation stream."""

import hashlib
import inspect
import json
import textwrap
from types import FunctionType

from tools.qwen4exp.resident_candidates.qfn_acceptance_snapshot import RUNNER
from tools.qwen4exp.resident_candidates.qfn_early_tp_draft import CONDITION, QUALIFIED_SOURCE_SHA256
from tools.qwen4exp.resident_candidates.qfn_mamba_commit import RPC

REPLACEMENT = """        early_pp_padded_drafter = (
            (use_pp_spec_decode or _observer.start(
                self, sampler_output, scheduler_output, logits, hidden_states,
                spec_decode_metadata, batch_desc, use_padded_batch))
            and not self.use_async_scheduling
            and use_padded_batch
        )"""


def build_forward(original, observer, *, source=None):
    original = getattr(original, "_qfn_observer_base", original)
    source = textwrap.dedent(inspect.getsource(original)) if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_SOURCE_SHA256 or source.count(CONDITION) != 1:
        raise RuntimeError("output observer requires the exact qualified sample_tokens source")
    decorator = "@torch.inference_mode()\n"
    if not source.startswith(decorator):
        raise RuntimeError("missing qualified inference decorator")
    body = source[len(decorator) :].replace(CONDITION, REPLACEMENT)
    factory = "def factory(_observer):\n" + textwrap.indent(body, "    ") + "    return sample_tokens\n"
    unwrapped = inspect.unwrap(original)
    namespace = dict(unwrapped.__globals__)
    exec(compile(factory, "<qfn-output-observer>", "exec"), namespace)
    compiled = namespace["factory"](observer)
    forward = FunctionType(
        compiled.__code__, unwrapped.__globals__, compiled.__name__, compiled.__defaults__, compiled.__closure__
    )
    forward.__annotations__, forward.__kwdefaults__ = compiled.__annotations__, compiled.__kwdefaults__
    forward.__qualname__ = original.__qualname__
    forward = unwrapped.__globals__["torch"].inference_mode()(forward)
    forward._qfn_observer_base = original
    return forward


def make_replacements(base, observer, runner_class):
    fallback = base[RUNNER + "._bookkeeping_sync"]
    previous_control = base[RPC]

    def bookkeeping(runner, *args, **kwargs):
        if observer.future is not None:
            if runner is not observer.runner:
                raise RuntimeError("output observation belongs to a different runner")
            return observer.finish()
        return fallback(runner, *args, **kwargs)

    bookkeeping._qfn_transition_base = fallback

    def control(worker, enabled, verify=False):
        if isinstance(enabled, str):
            try:
                payload = json.loads(enabled)
            except (ValueError, TypeError):
                return dict(rank=worker.rank, passed=False, error="invalid control JSON")
            if isinstance(payload, dict) and payload.get("output_observer") is True:
                try:
                    if payload != {"output_observer": True, "status": True}:
                        if set(payload) != {"output_observer", "enabled"} or type(payload["enabled"]) is not bool:
                            raise ValueError("output observer control requires boolean enabled")
                        if (
                            observer.future is not None
                            or observer.failed
                            or observer.runner.execute_model_state is not None
                        ):
                            raise RuntimeError("drain output observation before changing its controls")
                        observer.enabled = payload["enabled"]
                    return dict(rank=worker.rank, passed=True, **observer.status())
                except Exception as error:
                    return dict(rank=worker.rank, passed=False, error=str(error))
        return previous_control(worker, enabled, verify)

    return {
        RUNNER + ".sample_tokens": build_forward(runner_class.sample_tokens, observer),
        RUNNER + "._bookkeeping_sync": bookkeeping,
        RPC: control,
    }
