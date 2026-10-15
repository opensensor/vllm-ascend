# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.qwen4exp import decode_preparation_graph as module
from tools.qwen4exp.iteration_transition import accept_step, prepare_next_step


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_fixed_geometry_preserves_upstream_index_dtypes(n):
    geometry = module.fixed_host_geometry(n)
    scheduled = np.full(n, 3, dtype=np.int32)
    expected_cumulative = np.cumsum(scheduled)
    assert geometry["cu_tokens"].dtype == expected_cumulative.dtype == np.int64
    np.testing.assert_array_equal(geometry["cu_tokens"], expected_cumulative)
    logits = np.repeat(geometry["cu_tokens"] - scheduled, scheduled) + geometry["query_pos"]
    assert logits.dtype == np.int64
    np.testing.assert_array_equal(logits, np.arange(n * 3, dtype=np.int64))


@pytest.mark.parametrize("native_failure", [False, True])
def test_verification_restores_only_completed_metadata_on_parity_failure(monkeypatch, native_failure):
    owner = module.DecodePreparationGraphs.__new__(module.DecodePreparationGraphs)
    owner.failed = False
    owner.enabled = owner.verify = True
    owner.checked = owner.replays = 0
    owner.verification_error = None
    packet = object()
    buffer = torch.tensor([17], dtype=torch.int32)
    spec = SimpleNamespace(**{key: torch.tensor([2]) for key in module.SPEC_FIELDS})

    def replay():
        buffer.fill_(99)
        if native_failure:
            raise RuntimeError("uncertain submission")

    owner.records = {
        1: dict(
            outputs={"computed": buffer, "draft_token_ids": spec.draft_token_ids},
            spec=spec,
            graph=SimpleNamespace(replay=replay),
        )
    }
    owner.runner = SimpleNamespace(input_batch=SimpleNamespace(num_reqs=1, req_ids=["a"]))
    owner.transition = SimpleNamespace(next_pending=(("a",), {"next": packet}), records={1: {"next": packet}})
    owner.queue = SimpleNamespace(expected_device_inputs=None)
    owner.history = SimpleNamespace(continuation=("a",))
    result = (spec.logits_indices, spec, 3)
    owner.original = lambda *args: result
    monkeypatch.setattr(module, "admits_packet", lambda *args: True)
    if native_failure:
        with pytest.raises(RuntimeError, match="uncertain submission"):
            owner.prepare(None, None)
        assert owner.failed
    else:
        assert owner.prepare(None, None) is result
        assert buffer.item() == 17
        assert not owner.enabled and not owner.failed
        assert owner.verification_error and owner.checked == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("accepted", [1, 2, 3])
@pytest.mark.parametrize("computed", [10, 126, 127, 128, 2047])
def test_packet_materialization_tracks_rejection_and_boundaries(n, accepted, computed):
    inputs = torch.zeros((5, 8), dtype=torch.int32)
    inputs[1, :n] = computed
    inputs[2, :n], inputs[3, :n] = 3, 2
    samples = torch.full((n, 3), -1, dtype=torch.int64)
    samples[:, :accepted] = torch.arange(accepted) + 100
    drafts = torch.arange(n * 2).reshape(n, 2) + 200
    receipt = accept_step(samples, inputs, block_size=128)
    packet = prepare_next_step(samples, drafts, inputs, receipt, max_model_len=4096)
    outputs = dict(
        computed=torch.empty(n, dtype=torch.int32),
        positions=torch.empty(n * 3, dtype=torch.int64),
        seq_lens=torch.full((7,), 999, dtype=torch.int32),
        draft_token_ids=torch.empty(n * 2, dtype=torch.int32),
        shape_constant=torch.empty(5, dtype=torch.int32),
    )
    addresses = {k: v.data_ptr() for k, v in outputs.items()}
    module.materialize_packet(packet, outputs, {"shape_constant": torch.arange(5, dtype=torch.int32)})
    assert outputs["computed"].tolist() == [computed + accepted] * n
    assert outputs["positions"].tolist() == list(range(computed + accepted, computed + accepted + 3)) * n
    assert outputs["seq_lens"].tolist() == [computed + accepted + 3] * n + [0] * (7 - n)
    assert torch.equal(outputs["draft_token_ids"], drafts.reshape(-1))
    assert outputs["shape_constant"].tolist() == list(range(5))
    assert addresses == {k: v.data_ptr() for k, v in outputs.items()}


def fixture():
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["a"],
        req_prompt_embeds={},
        num_computed_tokens_cpu=np.array([10]),
        sampling_metadata=SimpleNamespace(output_token_ids=[]),
    )
    runner = SimpleNamespace(
        input_batch=batch,
        use_async_spec_decode=False,
        is_multimodal_model=False,
        num_prompt_logprobs=0,
        use_dcp=False,
        uses_mrope=False,
        lora_config=None,
        enable_prompt_embeds=False,
        max_model_len=128,
        requests={"a": SimpleNamespace(num_tokens=11)},
    )
    transition = SimpleNamespace(enabled=True, failed=False, verify=False, next_pending=(("a",), {}))
    history = SimpleNamespace(enabled=True, failed=False, verify=False, pending=("a",))
    schedule = SimpleNamespace(
        scheduled_new_reqs=[],
        finished_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a"], resumed_req_ids=set()),
        num_scheduled_tokens={"a": 3},
        scheduled_spec_decode_tokens={"a": [0, 0]},
        total_num_scheduled_tokens=3,
        has_structured_output_requests=False,
    )
    owner = SimpleNamespace(
        runner=runner,
        transition=transition,
        history=history,
        queue=SimpleNamespace(expected_device_inputs=("a",)),
        enabled=True,
        records={1: {}},
    )
    return owner, schedule, np.array([3])


def test_unchanged_packet_is_admitted():
    assert module.admits_packet(*fixture())


def test_multimodal_capable_model_accepts_text_only_decode():
    owner, schedule, counts = fixture()
    owner.runner.is_multimodal_model = True
    assert module.admits_packet(owner, schedule, counts)
    owner.runner.requests["a"].mm_features = [object()]
    assert not module.admits_packet(owner, schedule, counts)


@pytest.mark.parametrize(
    "change",
    [
        "disabled",
        "async",
        "new",
        "finished",
        "order",
        "history",
        "packet",
        "budget",
        "counts",
        "prefill",
        "penalty",
        "embeds",
        "verify",
    ],
)
def test_unsupported_state_falls_back_before_modifying_device_buffers(change):
    owner, schedule, counts = fixture()
    if change == "disabled":
        owner.enabled = False
    elif change == "async":
        owner.runner.use_async_spec_decode = True
    elif change == "new":
        schedule.scheduled_new_reqs = [object()]
    elif change == "finished":
        schedule.finished_req_ids.add("old")
    elif change == "order":
        schedule.scheduled_cached_reqs.req_ids = ["b"]
    elif change == "history":
        owner.history.pending = None
    elif change == "packet":
        owner.queue.expected_device_inputs = None
    elif change == "budget":
        owner.runner.max_model_len = 12
    elif change == "counts":
        counts[0] = 2
    elif change == "prefill":
        owner.runner.requests["a"].num_tokens = 100
    elif change == "penalty":
        owner.runner.input_batch.sampling_metadata.output_token_ids = [[1]]
    elif change == "embeds":
        owner.runner.input_batch.req_prompt_embeds = {"a": object()}
    elif change == "verify":
        owner.transition.verify = True
    assert not module.admits_packet(owner, schedule, counts)


def test_unqualified_runner_source_rejected():
    with pytest.raises(RuntimeError, match="exact qualified"):
        module.compile_preparation(lambda: None, None, source="def _prepare_inputs(): pass\n")


def test_failed_submission_cannot_fall_back(monkeypatch):
    owner = module.DecodePreparationGraphs.__new__(module.DecodePreparationGraphs)
    owner.failed = True
    owner.original = lambda *args: pytest.fail("must not replay after an uncertain submission")
    with pytest.raises(RuntimeError, match="retain graph owners"):
        owner.prepare(None, None)
