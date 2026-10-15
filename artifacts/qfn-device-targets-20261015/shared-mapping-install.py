def expected_rows(tokens, history, starts, geometry):
    result = []
    for request, (start, end) in enumerate(zip(starts, starts[1:])):
        for position in range(start, end):
            local = position - start
            current = tokens[position]
            recent = tokens[position - 1] if local else history[request][1]
            older = tokens[position - 2] if local >= 2 else history[request][local]
            if recent == geometry.eos:
                older = geometry.eos
            first = current * geometry.multipliers[0] ^ recent * geometry.multipliers[1]
            result.extend(
                (first ^ (older * geometry.multipliers[2] if head >= 8 else 0)) % size + offset
                for head, (size, offset) in enumerate(zip(geometry.sizes, geometry.offsets))
            )
    return result


"""Attach qualified lookup owners while retaining every original transport."""


def operation(wrapper):
    import os
    from types import MethodType

    import torch

    worker = getattr(wrapper, "worker", wrapper)
    try:
        state = worker._qfn_shared_ple_probe
        assert state["receipt"]["passed"] and len(state["owners"]) == 48
        live = wrapper._qfn_native_ple_serving
        manager = live["manager"]
        assert not manager.busy and not manager.failed
        state["old_lookups"] = manager.lookups
        state["old_factory"] = live["namespace"]["Lookup"]
        state["shadow"] = dict(checked=0, failed=False, errors=[])
        for key, owner in state["owners"].items():
            owner._shared_original_fetch = owner.fetch

            def checked_fetch(owner, tokens, starts, history):
                output = owner._shared_original_fetch(tokens, starts, history)
                if state["shadow"]["checked"] >= 96:
                    return output
                try:
                    token_list, starts_list, history_list = (
                        tokens.cpu().tolist(),
                        starts.cpu().tolist(),
                        history.cpu().tolist(),
                    )
                    rows = expected_rows(token_list, history_list, starts_list, owner.geometry)
                    expected = b"".join(os.pread(manager.fd, 320, row * 320) for row in rows)
                    assert owner.rows.cpu().reshape(-1).tolist() == rows
                    assert output.cpu().contiguous().view(torch.uint8).numpy().tobytes() == expected
                    state["shadow"]["checked"] += 1
                except Exception as error:
                    state["shadow"]["failed"] = True
                    if len(state["shadow"]["errors"]) < 4:
                        state["shadow"]["errors"].append(str(error))
                    # All native geometries passed privately. If live checking
                    # fails, use the already-owned original transport when one
                    # exists and retain diagnostics for the drained controller.
                    fallback = state["old_lookups"].get((tokens.shape[0], owner.geometry.requests, tokens.device.index))
                    if fallback is not None:
                        return fallback.fetch(tokens, starts, history)
                return output

            owner.fetch = MethodType(checked_fetch, owner)

        def factory(requested_manager, tokens, requests, device):
            assert requested_manager is manager
            return state["namespace"]["Lookup"](state["pool"], tokens, requests, device)

        state["factory"] = factory
        live["namespace"]["Lookup"] = factory
        manager.lookups = dict(state["owners"])
        manager.retained.extend(state["pool"].retained)
        state["pool"].retained = manager.retained
        state["installed"] = True
        return dict(rank=worker.rank, passed=True, owners=len(manager.lookups))
    except Exception as error:
        return dict(rank=worker.rank, passed=False, error=repr(error))


class Candidate:
    def validate(self, engine):
        assert (
            engine._pause_complete() and not engine.batch_queue and not engine.scheduler.get_num_unfinished_requests()
        )
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=all(row["passed"] for row in self.rows), workers=self.rows)

    def apply(self, engine):
        return self.validate(engine)


def prepare(engine):
    return Candidate()
