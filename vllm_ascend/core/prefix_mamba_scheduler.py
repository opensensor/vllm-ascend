# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in bounded prefix checkpoints for synchronous Qwen serving on 310P."""

from collections.abc import Mapping

from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import get_group_id
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager

from vllm_ascend._310p.prefix_mamba_state import prefix_mamba_slot_count

# Reserve both live windows, a CoW source window and a checkpoint-write window.
PREFIX_MAMBA_WORKING_WINDOWS = 4


def _is_310p() -> bool:
    # Hardware imports are deferred until scheduler construction so the
    # checkpoint policy can also be tested without torch_npu installed.
    from vllm_ascend.utils import is_310p

    return is_310p()


def prefix_mamba_checkpoint_limit(max_requests: int, speculative_tokens: int) -> int:
    primary_slots = prefix_mamba_slot_count(max_requests, speculative_tokens) - 1
    limit = primary_slots - max_requests * PREFIX_MAMBA_WORKING_WINDOWS * (1 + speculative_tokens)
    if limit < 1:
        raise ValueError("Bounded Mamba checkpoints need more compact slots or fewer requests/speculative tokens")
    return limit


def bounded_prefix_mamba_blocks(
    block_pool, managers: Mapping[int, object], checkpoint_limit: int, tracked=None
) -> dict[int, list[int]]:
    """Evict hash entries before retiring bytes; active/CoW states remain owned.

    All decisions happen once in the scheduler, rather than independently on
    TP workers. Evicting a Mamba hash makes hybrid lookup fall back to an older
    complete prefix. A block still owned by a request must retain its bytes,
    even when its hash has been evicted. The returned authoritative snapshot
    lets workers reclaim uncached, unowned states without CPU transfers.
    """
    if checkpoint_limit < 1:
        raise ValueError("Mamba checkpoint limit must be positive")
    if tracked is None:
        # One-shot inspection may begin with an existing cache. The serving
        # scheduler starts with an empty pool and passes persistent tracking,
        # avoiding a scan of all attention KV blocks on every decode step.
        tracked = {
            group_id: {block.block_id: None for block in block_pool.blocks if block.block_hash is not None}
            for group_id in managers
        }
    cached_ids = {}
    owned_ids = {}
    for group_id, manager in managers.items():
        owned = {block.block_id for blocks in manager.req_to_blocks.values() for block in blocks if not block.is_null}
        observed = tracked[group_id]
        for block_id in owned:
            observed.setdefault(block_id, None)
        for block_id, previous_hashes in list(observed.items()):
            block = block_pool.blocks[block_id]
            hashes = set(block_pool.cached_block_hashes_by_block.get(block_id, ()))
            if block.block_hash is not None:
                hashes.add(block.block_hash)
            hashes = frozenset(key for key in hashes if get_group_id(key) == group_id)
            if not hashes and block_id not in owned:
                del observed[block_id]
            elif hashes != previous_hashes:
                # Registering/promoting a checkpoint moves it to the newest
                # end. A recycled ID with another group's hash is not retained.
                del observed[block_id]
                observed[block_id] = hashes
        cached_ids[group_id] = [block_id for block_id, hashes in observed.items() if hashes]
        owned_ids[group_id] = owned
    victims = {block_id for ids in cached_ids.values() for block_id in ids[: max(0, len(ids) - checkpoint_limit)]}
    if victims:
        block_pool.evict_blocks(victims)
    result = {}
    for group_id in managers:
        for block_id in victims & tracked[group_id].keys():
            if block_id in owned_ids[group_id]:
                tracked[group_id][block_id] = frozenset()
            else:
                del tracked[group_id][block_id]
        retained = set(cached_ids[group_id]) - victims
        result[group_id] = sorted(owned_ids[group_id] | retained)
    return result


class PrefixMambaBoundedScheduler(Scheduler):
    """Bound retained histories separately from the attention KV token budget.

    Select explicitly with --scheduler-cls. No change to the default scheduler,
    live server, graph layout, precision or allocator is made by importing it.
    """

    def __init__(self, vllm_config, *args, **kwargs):
        model_type = getattr(vllm_config.model_config.hf_text_config, "model_type", None)
        if not _is_310p() or model_type != "qwen4_exp_text":
            raise ValueError("Bounded prefix Mamba scheduler currently requires Qwen4Exp on Ascend 310P")
        if vllm_config.scheduler_config.async_scheduling or vllm_config.kv_transfer_config is not None:
            raise ValueError("Bounded prefix Mamba scheduler requires synchronous standalone serving")
        if not vllm_config.cache_config.enable_prefix_caching or vllm_config.cache_config.mamba_cache_mode != "align":
            raise ValueError("Bounded prefix Mamba scheduler requires align-mode prefix caching")
        speculative = vllm_config.speculative_config
        speculative_tokens = speculative.num_speculative_tokens if speculative is not None else 0
        self._prefix_mamba_checkpoint_limit = prefix_mamba_checkpoint_limit(
            vllm_config.scheduler_config.max_num_seqs, speculative_tokens
        )
        super().__init__(vllm_config, *args, **kwargs)
        self._prefix_mamba_managers = {
            manager.kv_cache_group_id: manager
            for manager in self.kv_cache_manager.coordinator.single_type_managers
            if isinstance(manager, MambaManager)
        }
        if not self._prefix_mamba_managers:
            raise ValueError("Bounded prefix Mamba scheduler found no Mamba cache groups")
        self._prefix_mamba_tracked = {group_id: {} for group_id in self._prefix_mamba_managers}
        logger.info(
            "Bounded prefix Mamba retention: %d cached checkpoints per group; "
            "request-owned and pending CoW states are preserved separately.",
            self._prefix_mamba_checkpoint_limit,
        )

    def schedule(self, throttle_prefills: bool = False):
        output = super().schedule(throttle_prefills=throttle_prefills)
        retained = bounded_prefix_mamba_blocks(
            self.kv_cache_manager.block_pool,
            self._prefix_mamba_managers,
            self._prefix_mamba_checkpoint_limit,
            self._prefix_mamba_tracked,
        )
        # CoW sources can already be unowned by the time the output is sent.
        # Keep them through this step; the next snapshot can retire them.
        copies = {block_id for copy in (output.kv_cache_block_copies or ()) for block_id in copy if block_id > 0}
        output.ascend_prefix_mamba_block_ids = {
            group_id: sorted(set(block_ids) | copies) for group_id, block_ids in retained.items()
        }
        return output
