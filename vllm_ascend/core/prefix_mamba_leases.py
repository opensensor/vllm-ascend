# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned block leases for ordered, in-flight Mamba steps."""


class PrefixMambaStepLeases:
    """Keep physical IDs and worker snapshots alive until their output retires.

    Pool references also force copy-on-write when a newer step would otherwise
    mutate a bank still used by an older step. Aborting a request does not retire
    its submitted work; only the matching model output releases its lease.
    """

    def __init__(self, block_pool, depth):
        if type(depth) is not int or depth not in (1, 2):
            raise ValueError("Mamba step leases support queue depth one or two")
        self.block_pool = block_pool
        self.depth = depth
        self.pending = {}
        self.failed = False

    def check_capacity(self):
        if self.failed or len(self.pending) >= self.depth:
            raise RuntimeError("Mamba step leases failed or all reserved windows are in flight")

    def acquire(self, output, groups):
        self.check_capacity()
        key = id(output)
        if key in self.pending:
            raise RuntimeError("scheduler output already owns a Mamba lease")
        ids = {group: frozenset(values) for group, values in groups.items()}
        unique = sorted({value for values in ids.values() for value in values})
        blocks = []
        for block_id in unique:
            if not 0 <= block_id < len(self.block_pool.blocks):
                raise ValueError("Mamba lease references an invalid physical block")
            block = self.block_pool.blocks[block_id]
            if block.is_null or block.block_id != block_id:
                raise ValueError("Mamba lease cannot pin a null or mismatched block")
            blocks.append(block)
        # Retain the identities before touching the pool. A partial failure is
        # quarantined; guessing which references were acquired is unsafe.
        self.pending[key] = (output, ids, tuple(blocks))
        self.failed = True
        self.block_pool.touch(blocks)
        self.failed = False

    def retained(self, group):
        return {block for _, groups, _ in self.pending.values() for block in groups.get(group, ())}

    def release(self, output):
        record = self.pending.get(id(output))
        if self.failed or record is None or record[0] is not output:
            raise RuntimeError("Mamba lease retirement requires its exact completed output")
        self.failed = True
        self.block_pool.free_blocks(record[2])
        del self.pending[id(output)]
        self.failed = False

    @property
    def drained(self):
        return not self.failed and not self.pending
