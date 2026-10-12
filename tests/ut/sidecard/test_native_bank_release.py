# SPDX-License-Identifier: Apache-2.0
import weakref
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.native_bank_release import BANK_FIELDS, EXPERT_FIELDS, release_cuda_only_bank
from tools.sidecard.native_expert_probe import NATIVE_LAYOUT, SCALE_LAYOUT
from tools.sidecard.one_card_partition import PRECISION_PARTITION, one_card_partition


class Bank(list):
    pass


def native_bank():
    bank = Bank(SimpleNamespace() for _ in range(288))
    bank.local_expert_offset, bank.num_local_experts = 144, 72
    bank.grouped_ready, bank.offload_to_cpu = True, False
    bank.native_weight_layout, bank.native_weight_scale_layout = NATIVE_LAYOUT, SCALE_LAYOUT
    bank.gate_up_packed_bank = torch.zeros((72, 64, 32), dtype=torch.int8)
    bank.gate_packed_bank = bank.gate_up_packed_bank[:, :32]
    bank.up_packed_bank = bank.gate_up_packed_bank[:, 32:]
    bank.down_packed_bank = torch.zeros((72, 64, 16), dtype=torch.int8)
    bank.gate_up_scale_bank = torch.ones((72, 2, 1), dtype=torch.float16)
    bank.gate_scale_bank = bank.gate_up_scale_bank[:, :1]
    bank.up_scale_bank = bank.gate_up_scale_bank[:, 1:]
    bank.down_scale_bank = torch.ones((72, 2, 1), dtype=torch.float16)
    for index, expert in enumerate(bank[144:216]):
        for field in EXPERT_FIELDS:
            setattr(expert, field, getattr(bank, field + "_bank")[index])
    return bank


def test_release_drops_all_aliases_and_counts_each_storage_once():
    bank = native_bank()
    names = ("gate_up_packed_bank", "down_packed_bank", "gate_up_scale_bank", "down_scale_bank")
    references = [weakref.ref(getattr(bank, name)) for name in names]
    expected = sum(getattr(bank, name).numel() * getattr(bank, name).element_size() for name in names)
    partition = one_card_partition(PRECISION_PARTITION)
    qualification = {"decode": 3, "prefill": 1}
    assert release_cuda_only_bank(bank, partition, 8, qualification) == expected
    assert all(getattr(bank, name) is None for name in BANK_FIELDS)
    assert all(getattr(expert, name) is None for expert in bank[144:216] for name in EXPERT_FIELDS)
    assert all(reference() is None for reference in references)
    assert bank.grouped_ready is False
    assert release_cuda_only_bank(bank, partition, 8, qualification) == expected


@pytest.mark.parametrize("qualification", ({"decode": 2, "prefill": 1}, {"decode": 3, "prefill": 0}))
def test_release_requires_both_real_input_qualification_stages(qualification):
    bank = native_bank()
    with pytest.raises(ValueError, match="placement checks"):
        release_cuda_only_bank(bank, one_card_partition(PRECISION_PARTITION), 8, qualification)
    assert bank.grouped_ready and bank.gate_up_packed_bank is not None


def test_active_w4_bank_can_never_be_released():
    bank = native_bank()
    with pytest.raises(ValueError, match="actively routed"):
        release_cuda_only_bank(bank, one_card_partition(PRECISION_PARTITION), 3, {"decode": 3, "prefill": 1})
    assert bank.grouped_ready and bank.gate_up_packed_bank is not None


def test_incomplete_bank_fails_before_any_storage_is_dropped():
    bank = native_bank()
    bank.up_packed_bank = None
    with pytest.raises(ValueError, match="incomplete"):
        release_cuda_only_bank(bank, one_card_partition(PRECISION_PARTITION), 8, {"decode": 3, "prefill": 1})
    assert bank.grouped_ready and bank.gate_up_packed_bank is not None
