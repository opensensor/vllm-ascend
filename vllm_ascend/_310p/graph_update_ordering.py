# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in completion boundaries for mutable FULL graph task updates.

Host task-parameter mutation requires the previous replay to have completed.
A device wait alone cannot order this host mutation. Ready events additionally
cover current input staging and the update stream before starting replay.
"""

from collections.abc import Callable

import torch


class GraphUpdateOrdering:
    def __init__(self, event_factory: Callable | None = None) -> None:
        make_event = event_factory or torch.npu.Event
        self.previous_replay = make_event()
        self.update_done = make_event()
        self.replay_ready = make_event()
        self.has_previous_replay = False

    def before_update(self, main_stream, update_stream) -> None:
        if self.has_previous_replay and not self.previous_replay.query():
            self.previous_replay.synchronize()
        update_stream.wait_stream(main_stream)

    def ready(self, main_stream, update_stream):
        self.update_done.record(update_stream)
        main_stream.wait_event(self.update_done)
        self.replay_ready.record(main_stream)
        return self.replay_ready

    def replay_submitted(self, main_stream) -> None:
        self.previous_replay.record(main_stream)
        self.has_previous_replay = True
