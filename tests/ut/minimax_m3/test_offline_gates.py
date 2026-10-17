# SPDX-License-Identifier: Apache-2.0
"""Real geometry, zero-copy EP views and committed source snapshot boundaries."""

import io
import subprocess
import tarfile

import pytest
import torch

from tests.ut.minimax_m3.test_runtime import attention_oracle, make_bank
from tools.minimax_m3.real_moe_probe import ep_view
from tools.minimax_m3.source_bundle import archive_tree
from vllm_ascend._310p.ops.minimax_m3 import paged_sparse_attention


def test_ep_views_share_storage_and_preserve_global_expert_identity():
    bank = make_bank()
    bank.routed_scale = 1.5
    for rank in range(2):
        view = ep_view(bank, rank)
        assert view.routed_scale == 1.5
        assert view.local_expert_offset == rank * 2
        assert view.num_local_experts == 2
        assert view.w1_codes.untyped_storage().data_ptr() == bank.w1_codes.untyped_storage().data_ptr()
        torch.testing.assert_close(view.w1_codes, bank.w1_codes[rank * 2 : rank * 2 + 2])


@pytest.mark.parametrize("prefix,tokens", [(129, 2), (8190, 3)])
def test_tp2_production_attention_geometry_matches_independent_oracle(prefix, tokens):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        generator = torch.Generator().manual_seed(6144)
        length, block, dim, heads = prefix + tokens, 128, 128, 2
        pages = (length + block - 1) // block
        q = torch.randn(tokens, 32, dim, generator=generator).half()
        k = torch.randn(length, heads, dim, generator=generator).half()
        v = torch.randn(length, heads, dim, generator=generator).half()
        iq = torch.randn(tokens, heads, dim, generator=generator).half()
        ik = torch.randn(length, dim, generator=generator).half()
        main_pages = torch.randperm(pages, generator=generator)
        index_pages = main_pages.flip(0)
        positions = torch.arange(length)
        main_slots = main_pages[positions // block] * block + positions % block
        index_slots = index_pages[positions // block] * block + positions % block
        kc = torch.full((pages, block, heads, dim), torch.nan, dtype=torch.float16)
        vc = torch.full_like(kc, torch.nan)
        ic = torch.full((pages, block, dim), torch.nan, dtype=torch.float16)
        kc.flatten(0, 1)[main_slots[:prefix]] = k[:prefix]
        vc.flatten(0, 1)[main_slots[:prefix]] = v[:prefix]
        ic.flatten(0, 1)[index_slots[:prefix]] = ik[:prefix]
        output = paged_sparse_attention(
            q,
            k[prefix:],
            v[prefix:],
            iq,
            ik[prefix:],
            kc,
            vc,
            ic,
            query_starts=(0, tokens),
            seq_lens=torch.tensor([length]),
            block_table=main_pages[None],
            index_block_table=index_pages[None],
            slot_mapping=main_slots[prefix:],
            index_slot_mapping=index_slots[prefix:],
            max_seq_len=length,
            topk_blocks=16,
            scale=dim**-0.5,
        )
        reference = attention_oracle(q, k, v, iq, ik, prefix=prefix, block_size=block, topk=16, local=1, init=0)
        torch.testing.assert_close(output, reference, atol=0.001, rtol=0.001)
    finally:
        torch.set_num_threads(previous)


def test_archive_uses_committed_sources_and_excludes_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "model.py").write_text("committed\n")
    (repo / "artifacts").mkdir()
    (repo / "artifacts/large.dat").write_text("excluded\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qsm",
            "fixture",
        ],
        check=True,
    )
    (repo / "model.py").write_text("uncommitted\n")
    archive = tmp_path / "source.tar.gz"
    result = archive_tree(repo, "HEAD", archive, "runtime/")
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes()), mode="r:gz") as handle:
        assert handle.extractfile("runtime/model.py").read() == b"committed\n"
        assert not any("artifacts" in name for name in handle.getnames())
    assert result["excluded_uncommitted_changes"] is True
    with pytest.raises(ValueError, match="already exists"):
        archive_tree(repo, "HEAD", archive, "runtime/")
