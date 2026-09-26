# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for the 310P Mamba align fallback.

The fallback is only active on 310P and depends on runtime NPU/vLLM state.
Keep these checks import-free so they can run in lightweight DT environments
while still guarding the important upstream semantic contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATCH_MAMBA_UTILS = ROOT / "vllm_ascend" / "patch" / "worker" / "patch_mamba_utils.py"
MODEL_RUNNER = ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
QWEN4EXP_MODEL = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "model.py"


def _func(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {path}")


def _src(node: ast.AST) -> str:
    return ast.unparse(node)


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"method {class_name}.{method_name} not found in {path}")


def test_310p_postprocess_fallback_preserves_upstream_metadata_semantics() -> None:
    src = _src(_func(PATCH_MAMBA_UTILS, "_postprocess_mamba_align_gpu_cpu_fallback"))

    assert "num_accepted_tokens_gpu" in src
    assert "num_accepted_tokens_cpu_tensor[:num_reqs].copy_(num_accepted_tokens_gpu[:num_reqs])" in src
    assert "num_accepted_tokens = num_accepted_tokens_cpu_tensor" in src
    assert "num_accepted_tokens = input_batch.num_accepted_tokens_cpu" not in src
    assert "num_tokens_running_state = num_computed_tokens[i] + num_scheduled_tokens[i] - num_draft_tokens[i]" in src
    assert "new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens[i] - 1" in src
    assert "aligned_new_computed_tokens = new_num_computed_tokens // block_size * block_size" in src
    assert "if aligned_new_computed_tokens < num_tokens_running_state:" in src
    assert "if src_block_idx == dest_block_idx:" in src
    assert "num_accepted_tokens_cpu_tensor[i] = 1" in src


def test_310p_postprocess_fallback_mirrors_state_copy_without_triton() -> None:
    src = _src(_func(PATCH_MAMBA_UTILS, "_postprocess_mamba_align_gpu_cpu_fallback"))

    assert "run_fused_postprocess" not in src
    assert "postprocess_mamba_fused_kernel" not in src
    assert "accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state" in src
    assert "if accept_token_bias == 0:" in src
    assert "continue" in src
    assert "for mamba_group_id in ctx.mamba_group_ids:" in src
    assert "get_mamba_postprocess_block_ids(input_batch, mamba_group_id, i)" in src
    assert "copy_spec = state_copy_func(state, block_ids, src_block_idx, accept_token_bias + 1)" in src
    assert "_tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)" in src
    assert "dst_state.copy_(src_state.clone())" in src


def test_310p_fallback_selects_copy_funcs_by_mamba_layer_type() -> None:
    """GDN and PLE have different state-copy tuples in the upstream API."""
    for function_name in (
        "_collect_mamba_copy_meta_torch",
        "_collect_mamba_copy_meta_with_layers",
        "_postprocess_mamba_align_gpu_cpu_fallback",
    ):
        src = _src(_func(PATCH_MAMBA_UTILS, function_name))
        assert "mamba_utils._get_mamba_spec_for_layer" in src
        assert "state_copy_funcs = mamba_state_copy_funcs[mamba_spec.mamba_type]" in src
        assert "zip(kv_caches, state_copy_funcs)" in src

    groups_src = _src(_func(PATCH_MAMBA_UTILS, "_get_mamba_groups"))
    assert "group_spec = group_spec.first_spec" in groups_src
    assert "mamba_groups.setdefault(spec, set()).add(group_id)" in groups_src
    assert "inner_specs[0]" not in groups_src


def test_runner_passes_per_type_copy_funcs_to_align_paths() -> None:
    for method_name in ("_update_states_after_model_execute", "execute_model"):
        src = _src(_method(MODEL_RUNNER, "NPUModelRunner", method_name))
        assert "self._get_mamba_state_copy_funcs()" in src
        assert "self.model.get_mamba_state_copy_func()" not in src

    wrapper_src = _src(_method(QWEN4EXP_MODEL, "AscendQwen4ExpForConditionalGeneration", "get_mamba_state_copy_funcs"))
    assert "AscendQwen4ExpForCausalLM.get_mamba_state_copy_funcs(mamba_types)" in wrapper_src


def test_align_postprocess_reads_the_staged_compact_table() -> None:
    fallback_src = _src(_func(PATCH_MAMBA_UTILS, "_postprocess_mamba_align_gpu_cpu_fallback"))
    assert "get_mamba_postprocess_block_ids(input_batch, mamba_group_id, i)" in fallback_src

    runner_310 = ROOT / "vllm_ascend" / "_310p" / "model_runner_310p.py"
    remap_src = _src(_method(runner_310, "NPUModelRunner310", "_remap_compact_mamba_block_tables"))
    assert "mapped_tables[group_idx] = mapped" in remap_src
    assert "self.input_batch._prefix_mamba_postprocess_tables = mapped_tables" in remap_src


def test_no_prefix_compaction_covers_later_columns_and_cpu_state_copies() -> None:
    runner_310 = ROOT / "vllm_ascend" / "_310p" / "model_runner_310p.py"
    remap_src = _src(_method(runner_310, "NPUModelRunner310", "_remap_compact_mamba_block_tables"))
    stage_src = _src(_method(runner_310, "NPUModelRunner310", "_stage_prefix_mamba_request_ids"))
    execute_src = _src(_method(runner_310, "NPUModelRunner310", "execute_model"))

    assert "compact_columns.remainder_(self.num_compact_mamba_blocks)" in remap_src
    assert "mapped_tables[group_idx] = np.broadcast_to" in remap_src
    assert "column % self.num_compact_mamba_blocks" in stage_src
    assert "if self.supports_compact_mamba_state:" in execute_src
