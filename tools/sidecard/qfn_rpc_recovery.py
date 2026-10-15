# SPDX-License-Identifier: Apache-2.0
"""Cold repair of measured response-queue debt after a failed collective RPC.

The controller must pause and drain inference, measure each rank's FIFO offset
with unique prepare generations, then submit only read-only drain probes. This
does not drain hardware, change graphs, or authorize dropping inference output.
"""

MAX_REPLY_DEBT = 16


def suppress_stale_replies(process, count, nonce, *, inference_drained):
    """Suppress exactly the measured number of future control acknowledgments.

    Each all-rank probe still consumes one old reply from each indebted rank. Once
    the old replies are consumed, normal enqueue behavior is restored automatically.
    The current recovery RPC's acknowledgment is included in the suppressed count.
    """
    if inference_drained is not True or process.use_async_scheduling:
        raise ValueError("reply repair requires drained synchronous workers")
    if type(count) is not int or not 0 <= count <= MAX_REPLY_DEBT or not isinstance(nonce, str) or not nonce:
        raise ValueError("invalid measured reply debt or recovery identity")
    previous = getattr(process, "_qfn_reply_recovery_receipt", None)
    if previous is not None:
        if previous["nonce"] == nonce:
            if previous["requested"] != count:
                raise ValueError("recovery identity cannot change debt")
            return previous
        if not previous["complete"]:
            raise RuntimeError("previous reply repair is still active")
    receipt = dict(nonce=nonce, requested=count, suppressed=0, complete=count == 0)
    process._qfn_reply_recovery_receipt = receipt
    if count == 0:
        return receipt
    existed = "enqueue_output" in vars(process)
    original = process.enqueue_output

    def drop_control_reply(output):
        receipt["suppressed"] += 1
        if receipt["suppressed"] == count:
            if existed:
                process.enqueue_output = original
            else:
                del process.enqueue_output
            receipt["complete"] = True

    process.enqueue_output = drop_control_reply
    return receipt
