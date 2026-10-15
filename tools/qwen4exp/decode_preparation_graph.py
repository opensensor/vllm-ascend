# SPDX-License-Identifier: Apache-2.0
"""Consume the device next-step packet without rebuilding fixed decode metadata.

The host still owns allocation, compact state remapping and checkpoint restore.
Only exact, unchanged MTP2 rows use this owner; request changes use the original
runner. Existing acceptance synchronization remains until checkpoint reset
ownership also moves to the device.
"""

import hashlib
import inspect
import textwrap
from types import FunctionType, MethodType

import numpy as np
import torch

from tools.qwen4exp.queued_schedule import can_use_device_drafts

MTP_WIDTH = 3
MAX_REQUESTS = 4
QUALIFIED_SOURCE_SHA256 = "3925a317d984fbe22146416068fe4d3729df152cdbdee261e2990ca26761dea7"
SPEC_FIELDS = (
    "draft_token_ids",
    "cu_num_draft_tokens",
    "cu_num_sampled_tokens",
    "target_logits_indices",
    "bonus_logits_indices",
    "logits_indices",
)


def fixed_host_geometry(n):
    """Match the runner's INT64 cumulative scheduling/index contract exactly."""
    return dict(
        req_indices=np.repeat(np.arange(n, dtype=np.int64), MTP_WIDTH),
        query_pos=np.tile(np.arange(MTP_WIDTH, dtype=np.int64), n),
        cu_tokens=np.arange(1, n + 1, dtype=np.int64) * MTP_WIDTH,
        drafts=np.full(n, MTP_WIDTH - 1, dtype=np.int32),
    )


def materialize_packet(packet, outputs, constants):
    """Device tensor operations only; page ownership is deliberately untouched."""
    outputs["computed"].copy_(packet.computed)
    outputs["positions"].copy_(packet.positions.reshape(-1))
    outputs["seq_lens"].zero_()
    outputs["seq_lens"][: packet.computed.numel()].copy_(packet.seq_lens)
    outputs["draft_token_ids"].copy_(packet.input_ids[:, 1:].reshape(-1))
    for key, value in constants.items():
        outputs[key].copy_(value)


def compile_preparation(original, owner, source=None, *, checkpoint=False):
    """Keep the reviewed runner's cache/lifecycle code; replace narrow seams."""
    source = textwrap.dedent(inspect.getsource(original)) if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != QUALIFIED_SOURCE_SHA256:
        raise RuntimeError("decode preparation requires the exact qualified runner source")

    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError("qualified decode preparation seam differs")
        source = source.replace(old, new)

    replace_once(
        "req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)",
        'req_indices = _owner.host[num_reqs]["req_indices"]',
    )
    begin = source.index("    if not scheduler_output.scheduled_spec_decode_tokens:")
    end = source.index("    with_prefill =", begin)
    source = source[:begin] + "    attn_state = self.attn_state = AscendAttentionState.SpecDecoding\n\n" + source[end:]
    replace_once(
        "cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)",
        'cu_num_tokens = _owner.host[num_reqs]["cu_tokens"]\n'
        '    self.query_pos.np[:total_num_scheduled_tokens] = _owner.host[num_reqs]["query_pos"]',
    )
    replace_once(
        """        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )""",
        "        pass  # Device packet supplies computed positions.",
    )
    begin = source.index("    token_indices = positions_np +")
    end = source.index("    if self.enable_prompt_embeds:", begin)
    source = source[:begin] + source[end:]
    previous = "    self._compute_prev_positions(num_reqs)"
    if source.count(previous) != 2:
        raise RuntimeError("qualified previous-position calls differ")
    second = source.index(previous, source.index(previous) + len(previous))
    source = source[:second] + source[second:].replace(previous, "    pass  # Previous positions already prepared.", 1)
    for statement in (
        "self.query_start_loc.copy_to_gpu()",
        "self.gdn_query_start_loc.copy_to_gpu()",
        "self.req_indices.copy_to_gpu(total_num_scheduled_tokens)",
        "self.query_pos.copy_to_gpu(total_num_scheduled_tokens)",
        "self.num_scheduled_tokens.copy_to_gpu(num_reqs)",
        "self.num_decode_draft_tokens.copy_to_gpu()",
    ):
        replace_once(statement, "pass  # Materialized by the device preparation graph.")
    begin = source.index("    self.positions[:total_num_scheduled_tokens].copy_(")
    end = source.index("    if (\n        self._needs_seq_lens_cpu_sync", begin)
    source = source[:begin] + source[end:]
    begin = source.index("    num_tokens = [self.requests[r].num_tokens")
    end = source.index("    self.req_indices.np", begin)
    source = source[:begin] + "    self.num_discarded_requests = 0\n\n" + source[end:]
    begin = source.index("        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)")
    end = source.index("        spec_decode_metadata = self._calc_spec_decode_metadata(", begin)
    source = (
        source[:begin]
        + ('        num_draft_tokens = num_decode_draft_tokens = _owner.host[num_reqs]["drafts"]\n\n')
        + source[end:]
    )
    replace_once(
        """        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens,
            cu_num_tokens,
        )""",
        '        spec_decode_metadata = _owner.records[num_reqs]["spec"]',
    )
    if checkpoint:
        begin = source.index("    if self.num_accepted_tokens_event is not None:")
        end = source.index("    need_async_num_computed_update =", begin)
        source = (
            source[:begin]
            + "    if not _owner.checkpoint_preparation_v1.prepare_counts(num_reqs):\n"
            + textwrap.indent(source[begin:end], "    ")
            + source[end:]
        )
    # Host cache/lifecycle mirrors remain; the optional checkpoint owner
    # supplies device-normalized acceptance instead of the count round trip.
    factory = "def factory(_owner):\n" + textwrap.indent(source, "    ") + "    return _prepare_inputs\n"
    namespace = dict(original.__globals__)
    exec(compile(factory, "<qfn-device-preparation>", "exec"), namespace)
    compiled = namespace["factory"](owner)
    return FunctionType(
        compiled.__code__, original.__globals__, compiled.__name__, compiled.__defaults__, compiled.__closure__
    )


def admits_packet(owner, schedule, counts):
    runner = owner.runner
    n = runner.input_batch.num_reqs
    ids = tuple(runner.input_batch.req_ids)
    return (
        owner.enabled
        and n in owner.records
        and not runner.use_async_spec_decode
        and not runner.input_batch.req_prompt_embeds
        and owner.queue.expected_device_inputs == ids
        and can_use_device_drafts(runner, schedule, owner.transition, owner.history)
        and np.asarray(counts).shape == (n,)
        and np.all(counts == MTP_WIDTH)
        and np.all(runner.input_batch.num_computed_tokens_cpu[:n] + MTP_WIDTH <= runner.max_model_len)
        and all(
            not getattr(runner.requests[key], "mm_features", ())
            and runner.requests[key].num_tokens <= runner.input_batch.num_computed_tokens_cpu[i] + MTP_WIDTH
            for i, key in enumerate(ids)
        )
    )


class DecodePreparationGraphs:
    def __init__(self, commit):
        self.runner = commit.runner
        self.transition = commit.device_transition_v1
        self.history = commit.ple_history_v2
        self.queue = commit.queued_schedule_v1
        self.enabled = self.verify = False
        self.failed = True
        self.replays = self.checked = self.fallbacks = 0
        self.verification_error = None
        self.records, self.host = {}, {}
        self.original = self.runner._prepare_inputs
        self.fast = compile_preparation(self.original.__func__, self)
        runner = self.runner
        # All shapes and storage addresses stay identical to the model captures.
        stream = torch.npu.Stream(device=runner.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for n in range(1, MAX_REQUESTS + 1):
                count = n * MTP_WIDTH
                self.host[n] = fixed_host_geometry(n)
                req_indices, query_pos, cu_tokens = (
                    self.host[n][key] for key in ("req_indices", "query_pos", "cu_tokens")
                )
                spec = runner._calc_spec_decode_metadata(np.full(n, MTP_WIDTH - 1, dtype=np.int32), cu_tokens)
                outputs = dict(
                    computed=runner.num_computed_tokens[:n],
                    positions=runner.positions[:count],
                    seq_lens=runner.seq_lens,
                    draft_token_ids=spec.draft_token_ids,
                    req_indices=runner.req_indices.gpu[:count],
                    query_pos=runner.query_pos.gpu[:count],
                    num_scheduled=runner.num_scheduled_tokens.gpu[:n],
                    query_start=runner.query_start_loc.gpu,
                    gdn_start=runner.gdn_query_start_loc.gpu,
                    num_drafts=runner.num_decode_draft_tokens.gpu,
                )
                constants = {
                    key: torch.empty_like(outputs[key])
                    for key in ("req_indices", "query_pos", "num_scheduled", "query_start", "gdn_start", "num_drafts")
                }
                constants["req_indices"].copy_(torch.from_numpy(req_indices))
                constants["query_pos"].copy_(torch.from_numpy(query_pos))
                constants["num_scheduled"].fill_(MTP_WIDTH)
                constants["query_start"].fill_(-1)
                constants["gdn_start"].fill_(count)
                boundaries = torch.arange(n + 1, dtype=constants["query_start"].dtype, device=runner.device) * MTP_WIDTH
                constants["query_start"][: n + 1].copy_(boundaries)
                constants["gdn_start"][: n + 1].copy_(boundaries)
                constants["num_drafts"].fill_(-1)
                constants["num_drafts"][:n].fill_(MTP_WIDTH - 1)
                record = dict(spec=spec, outputs=outputs, constants=constants, graph=torch.npu.NPUGraph())
                self.records[n] = record  # Retain before native capture.
                packet = self.transition.records[n]["next"]
                for _ in range(3):
                    materialize_packet(packet, outputs, constants)
                torch.npu.synchronize()
                with torch.npu.graph(record["graph"], stream=stream):
                    materialize_packet(packet, outputs, constants)
                torch.npu.synchronize()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.failed = False

    def prepare(self, schedule, counts):
        if self.failed:
            raise RuntimeError("device preparation failed; retain graph owners")
        if not admits_packet(self, schedule, counts):
            self.fallbacks += 1
            return self.original(schedule, counts)
        n = self.runner.input_batch.num_reqs
        packet = self.transition.next_pending[1]["next"]
        if packet is not self.transition.records[n]["next"]:
            raise RuntimeError("device preparation packet owner differs")
        self.failed = True
        record = self.records[n]
        expected = None
        if self.verify:
            result = self.original(schedule, counts)
            expected = {name: tensor.clone() for name, tensor in record["outputs"].items() if name != "draft_token_ids"}
            expected["draft_token_ids"] = result[1].draft_token_ids.clone()
        else:
            result = self.fast(self.runner, schedule, counts)
        if self.queue.expected_device_inputs is not None or self.history.continuation != tuple(
            self.runner.input_batch.req_ids
        ):
            raise RuntimeError("device preparation requires consumed target/history packet")
        # The existing input consumer ordered the next_done event on this stream.
        record["graph"].replay()
        if expected is not None:
            try:
                for name, value in expected.items():
                    torch.testing.assert_close(record["outputs"][name].cpu(), value.cpu(), rtol=0, atol=0)
                for field in SPEC_FIELDS[1:]:
                    torch.testing.assert_close(
                        getattr(record["spec"], field).cpu(), getattr(result[1], field).cpu(), rtol=0, atol=0
                    )
            except AssertionError as error:
                # Only pure metadata was written, and CPU comparisons observe
                # completed work. Restore the independently computed original
                # buffers before any model runs. Native submission failures
                # deliberately do not enter this recovery path.
                self.verification_error = str(error)
                self.enabled = False
                for name, value in expected.items():
                    record["outputs"][name].copy_(value)
                self.failed = False
                return result
            self.checked += 1
        self.replays += 1
        self.failed = False
        return result

    def install(self):
        owner = self

        def prepare(runner, schedule, counts):
            if runner is not owner.runner:
                raise RuntimeError("device preparation belongs to another runner")
            return owner.prepare(schedule, counts)

        self.runner._prepare_inputs = MethodType(prepare, self.runner)

    def status(self):
        return dict(
            enabled=self.enabled,
            verify=self.verify,
            failed=self.failed,
            replays=self.replays,
            checked=self.checked,
            fallbacks=self.fallbacks,
            verification_error=self.verification_error,
            capture_requests=sorted(self.records),
            host_cache_management_retained=True,
        )
