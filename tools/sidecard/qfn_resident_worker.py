# SPDX-License-Identifier: Apache-2.0
"""Paused QFN reload RPCs: retain weights and custom transport allocations."""

import gc

import torch

from tools.glm_perf.resident_worker import ResidentWorkerExtension, clear_graph_state
from tools.qwen4exp.draft_real_weight_gate import QfnDraftRealWeightMixin
from tools.sidecard.npu_collective_graph import CollectiveGraphObserver
from tools.sidecard.qfn_collective_profile import QfnProfileWorkerExtension
from tools.sidecard.qfn_collective_worker import _communicator
from tools.sidecard.qfn_resident_lifecycle import (
    renew_collective_graphs,
    reset_compact_mamba_tables,
    reset_prefix_tier,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
)


class QfnResidentWorkerExtension(QfnDraftRealWeightMixin, QfnProfileWorkerExtension, ResidentWorkerExtension):
    def resident_apply(self, generation):
        try:
            if getattr(self, "_qfn_reload_failed", False):
                raise RuntimeError("previous graph replacement failed; retain paused ownership for inspection")
            if getattr(self.model_runner, "execute_model_state", None) is not None:
                raise RuntimeError("worker still has pending model execution")
            session = self._resident_session()
            if session.current is not None and session.current.generation == generation:
                return self.resident_status()
            if session.pending is None or session.pending[0].generation != generation:
                raise RuntimeError("generation has not been prepared")
            control = session.pending[0]
            if control.mode != "graph":
                raise ValueError("custom QFN resident server requires graph execution")
            old_digest = session.current.digest if session.current else None
            changed = bool(control.source) if old_digest is None else old_digest != control.digest
            if changed or control.recapture:
                wrappers = self._resident_wrappers()
                facade = _communicator().custom_sum
                eager_wrappers = ()
                if getattr(self.model_runner.speculative_config, "method", None) == "dspark":
                    drafter = self.model_runner.drafter
                    if not self.model_runner.speculative_config.enforce_eager or drafter.use_cuda_graph:
                        raise RuntimeError("DSpark resident reload requires an explicitly eager draft")
                    eager_wrappers = (drafter.model,)

                def clear(wrappers):
                    clear_graph_state(
                        wrappers, [get_graph_params(), get_draft_graph_params(), get_draft_graph_prefill_params()]
                    )

                # Failure from this point leaves the server paused; never call
                # close_custom_collective, unload weights or free peer imports.
                self._qfn_reload_failed = True
                self._qfn_last_retirement = renew_collective_graphs(
                    facade,
                    wrappers,
                    synchronize=torch.npu.synchronize,
                    clear_state=clear,
                    observer_factory=CollectiveGraphObserver,
                    eager_wrappers=eager_wrappers,
                )
                session.apply(generation)
                gc.collect()
                torch.npu.empty_cache()
                self._qfn_reload_failed = False
            else:
                session.apply(generation)
            self._resident_set_mode()
            return self.resident_status()
        except Exception as error:
            return self._resident_error(error)

    def resident_status(self):
        try:
            return self._qfn_resident_status()
        except Exception as error:
            return self._resident_error(error)

    def _qfn_resident_status(self):
        result = super().resident_status()
        collective = self.npu_collective_status()
        result["custom_collective"] = collective
        result["last_graph_retirement"] = getattr(self, "_qfn_last_retirement", None)
        result["native_failed"] |= bool(getattr(self, "_qfn_reload_failed", False)) or collective["state"] != "active"
        result["prefix_mamba"] = {
            str(group): {
                "resident_checkpoints": len(tier._resident),
                "archive_checkpoints": len(tier._device_archive_resident),
                "host_checkpoints": len(tier._host),
            }
            for group, tier in getattr(self.model_runner, "_prefix_mamba_tiers", {}).items()
        }
        return result

    @torch.inference_mode()
    def resident_reset(self):
        try:
            super().resident_reset()
            reset_compact_mamba_tables(self.model_runner)
            for tier in getattr(self.model_runner, "_prefix_mamba_tiers", {}).values():
                reset_prefix_tier(tier)
            torch.npu.synchronize()
            return self.resident_status()
        except Exception as error:
            return self._resident_error(error)
