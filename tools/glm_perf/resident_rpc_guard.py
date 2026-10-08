# SPDX-License-Identifier: Apache-2.0
"""Contain diagnostic exceptions before the executor loses peer responses."""

from functools import wraps

GUARDED_METHODS = frozenset({"resident_status", "profile"})
WORKER_PREFIX = "vllm_ascend._310p.worker_310p:NPUWorker310."


def guard_rpc(function):
    @wraps(function)
    def guarded(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except Exception as error:
            # Every rank returns its error receipt. The client can collect all
            # acknowledgments before deciding whether a transition is safe.
            return self._resident_error(error)

    return guarded


def guard_replacements(changes):
    result = dict(changes)
    for method in GUARDED_METHODS:
        target = WORKER_PREFIX + method
        if target in result:
            result[target] = guard_rpc(result[target])
    return result


def check_profile_label(label, allowed):
    """Reject an unsupported workload locally, before broadcasting a start."""
    if not isinstance(label, str) or label not in allowed:
        raise ValueError("unsupported CANN profile workload: " + repr(label))
    return label
