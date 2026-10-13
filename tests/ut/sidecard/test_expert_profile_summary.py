# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from tools.sidecard.expert_profile_summary import summarize


def example():
    run = dict(
        passed=True,
        full_vocabulary_reference_bitwise_equal=True,
        workloads=[dict(name="example", first_sequence=0, end_sequence=2, prompt_tokens=2)],
    )
    frames, workers = [], []
    for phase, rows in ((0, 2), (1, 1)):
        dispatches = []
        for offset, layer in enumerate(range(18, 33)):
            sequence = phase * 15 + offset
            events = []
            for rank in (0, 1):
                events.append(dict(rank=rank, header=[sequence, layer, rows]))
                workers.append(
                    dict(
                        rank=rank,
                        sequence=sequence,
                        layer=layer,
                        rows=rows,
                        stream_ms=dict(upload=0.1 + rank * 0.1, native=2 + rank, download=0.1 + rank * 0.1),
                    )
                )
            dispatches.append(
                dict(layer=layer, rows=rows, cold_routes={0: 4 * rows, 1: 4 * rows}, socket_events=events)
            )
        frames.append(
            dict(
                sequence=phase,
                position=phase * 2,
                rows=rows,
                total_ms=100,
                host_ms=dict(input=30, receive=50, submit=10, head=10),
                dispatches=dispatches,
                device_stream_ms=[dict(name="backbone", ms=60), dict(name="dispatch", ms=60)],
            )
        )
    return run, frames, workers


def test_host_waterfall_accounts_for_total_but_parallel_device_spans_do_not_add():
    result = summarize(*example())["workloads"][0]["phases"]["decode"]
    assert sum(result["host_mean_ms"].values()) == result["mean_forward_ms"] == 100
    assert sum(result["host_share_percent"].values()) == 100
    assert sum(result["cuda_stream_mean_ms"].values()) == 120
    assert result["npu_sum_of_layer_rank_maxima_mean_ms"]["all_stages"] == pytest.approx(51)
    assert result["npu_sum_of_layer_rank_maxima_mean_ms"]["native"] == 45
    assert result["mean_cold_route_fraction"] == 1
    assert result["mean_chip_calls_per_step"] == 30


def test_dma_summary_reports_host_worker_spans_and_unknown_route_counts():
    run, frames, workers = example()
    run["transport"] = "direct-dma"
    for frame in frames:
        for dispatch in frame["dispatches"]:
            dispatch["cold_routes"] = None
    for worker in workers:
        stages = worker.pop("stream_ms")
        worker.update(
            upload_ms=stages["upload"],
            unpack_and_validation_ms=0.3,
            native_ms=stages["native"],
            pack_and_download_ms=stages["download"],
        )
    report = summarize(run, frames, workers)
    assert report["worker_timing_scope"] == "host wall including completion waits"
    result = report["workloads"][0]["phases"]["decode"]
    assert result["mean_cold_route_fraction"] is None
    assert result["npu_sum_of_layer_rank_maxima_mean_ms"]["validation"] == pytest.approx(4.5)
    assert result["npu_sum_of_layer_rank_maxima_mean_ms"]["all_stages"] == pytest.approx(55.5)


@pytest.mark.parametrize("error", ["duplicates", "missing_worker", "incomplete_layers", "host_accounting", "parity"])
def test_profile_summary_fails_closed_on_incomplete_or_ambiguous_evidence(error):
    run, frames, workers = copy.deepcopy(example())
    if error == "duplicates":
        workers.append(workers[0])
    elif error == "missing_worker":
        workers.pop()
    elif error == "incomplete_layers":
        frames[0]["dispatches"].pop()
    elif error == "host_accounting":
        frames[0]["total_ms"] += 1
    else:
        run["full_vocabulary_reference_bitwise_equal"] = False
    with pytest.raises(ValueError):
        summarize(run, frames, workers)
