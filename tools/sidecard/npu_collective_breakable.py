# SPDX-License-Identifier: Apache-2.0
"""Opt-in cold native ownership for upstream breakable graph captures.

Loaded only for configured custom collectives. The capture preparation follows
the qualified upstream BreakableCUDAGraphWrapper; default plugin capture still
delegates upstream. Segment replay remains the original bound native methods.
No global capture factory or thread-local implementation is monkeypatched.
"""

from vllm.compilation import breakable_cudagraph as upstream


class _CollectiveBreakableCapture(upstream.BreakableCUDAGraphCapture):
    def __init__(self, pool=None):
        super().__init__(pool=pool)
        self.native_graphs = []

    def _begin_segment(self):
        assert not self._capturing
        graph = upstream.torch.cuda.CUDAGraph()
        # Upstream assigns _current_graph only after capture_begin. Retain it
        # first so a partially failed native begin cannot lose its owner.
        self.native_graphs.append(graph)
        self._current_graph = graph
        if self.pool is not None:
            graph.capture_begin(pool=self.pool)
        else:
            graph.capture_begin()
        self._capturing = True

    def reset(self):
        if self._capturing or self._current_graph is not None:
            raise RuntimeError("breakable graph still owns an active native capture")
        for graph in self.native_graphs:
            if getattr(graph, "capture_token", None) is not None:
                raise RuntimeError("breakable segment still owns an active capture token")
            graph.reset()
        # Do not drop segment/task owners unless EVERY native reset succeeded.
        for graph in self.native_graphs:
            for name in ("tasks", "provider_sizes", "capture_resources"):
                metadata = getattr(graph, name, None)
                if metadata is not None:
                    metadata.clear()
        self.segments.clear()
        self.replay = self._retired_replay

    @staticmethod
    def _retired_replay():
        raise RuntimeError("breakable collective graph has been retired")


def capture_with_collective(wrapper, entry, args, kwargs, observer):
    """Capture with leases before any native segment starts; retain on error."""
    upstream.validate_cudagraph_capturing_enabled()
    entry.input_addresses = wrapper._collect_tensor_addresses(args, kwargs)
    if wrapper.graph_pool is not None:
        upstream.set_graph_pool_id(wrapper.graph_pool)
    else:
        upstream.set_graph_pool_id(upstream.current_platform.graph_pool_handle())
    if upstream.gc.isenabled():
        upstream.gc.collect()
        upstream.torch.accelerator.empty_cache()
    upstream.get_offloader().sync_prev_onload()

    capture = _CollectiveBreakableCapture(pool=wrapper.graph_pool)
    with observer.capture_scope(capture):
        with capture:
            output = wrapper.runnable(*args, **kwargs)
            upstream.get_offloader().join_after_forward()
            output = upstream.weak_ref_tensors(output)
        # Bind a monolithic capture straight to native replay. For multiple
        # segments, upstream's loop already contains native bound methods plus
        # original eager callables; no observer is inserted in that hot path.
        if capture.num_graphs == 1 and capture.num_eager_breaks == 0:
            capture.replay = capture.segments[0]
        if observer.track_replays and capture.num_eager_breaks:
            raise RuntimeError("breakable eager segments require unobserved collective replay")

    if observer.track_replays:
        native_replay = capture.replay

        def tracked_replay():
            observer.before_replay(capture)
            try:
                native_replay()
            except BaseException as error:
                observer.replay_failed(capture, error)
                raise
            observer.replayed(capture)

        capture.replay = tracked_replay
    entry.capture = capture
    entry.output = upstream.weak_ref_tensors(output)
    upstream.logger.debug("Captured breakable collective graph for %s: %r", entry.batch_descriptor, capture)
    return output
