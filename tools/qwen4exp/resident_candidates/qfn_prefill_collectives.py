# SPDX-License-Identifier: Apache-2.0
"""Explicit cold RPC factory; never run during a suspended keep-pause."""

from tools.sidecard.qfn_prefill_collective_probe import run_probe

TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"


def make_rpc(original_stop, *, probe=run_probe):
    def qfn_cpu_profile_stop(self, output=None):
        if output is not None:
            return original_stop(self, output)
        try:
            if any(
                getattr(self, name, None) is not None for name in ("_qfn_cpu_profile", "_qfn_native_profile_session")
            ):
                raise RuntimeError("finish profiling before the private collective benchmark")
            return dict(rank=self.rank, prefill_collectives=probe(self))
        except Exception as error:
            return dict(
                rank=getattr(self, "rank", None),
                prefill_collectives=dict(
                    passed=False, error=f"{type(error).__name__}: {error}", resources_may_be_retained=True
                ),
            )

    return qfn_cpu_profile_stop


def replacements():
    from vllm_ascend._310p.worker_310p import NPUWorker310

    return {TARGET: make_rpc(NPUWorker310.qfn_cpu_profile_stop)}
