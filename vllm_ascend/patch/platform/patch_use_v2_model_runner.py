import vllm.envs as envs
from vllm.config.vllm import VllmConfig
from vllm.logger import logger

from vllm_ascend._310p.qwen4exp_mtp import is_qwen4exp_mtp_config
from vllm_ascend.utils import is_310p
from vllm_ascend.worker.v2.pp_utils import resolve_spec_pp_support

_original_validate_v2_model_runner = VllmConfig._validate_v2_model_runner
_original_get_unsupported_features = VllmConfig._get_v2_model_runner_unsupported_features

_ASCEND_V1_SUPPORTED_FEATURES = frozenset(
    {
        "dspark speculative decoding",
        "dflash2 drafts",
    }
)


def _needs_310p_qwen4exp_mtp_v1(self) -> bool:
    """310P MRv2 has no speculative input packing or rejection sampler yet."""
    return is_310p() and is_qwen4exp_mtp_config(self.model_config, self.speculative_config)


def _patched_use_v2_model_runner(self) -> bool:
    """Use the Ascend runner selection, including 310P MTP fallback.

    The upstream use_v2_model_runner gate-keeps the v2 runner with
    per-model architecture whitelists, Triton availability checks, and
    feature-support inspections. On Ascend the v2 runner is controlled
    by the VLLM_USE_V2_MODEL_RUNNER environment variable. The 310P Qwen4Exp
    MTP path uses MRv1 until MRv2 can pack and verify speculative tokens.
    """
    if _needs_310p_qwen4exp_mtp_v1(self):
        logger.warning_once("Qwen4Exp MTP on 310P uses model runner v1 for speculative decoding.")
        return False
    use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2 is not None:
        return use_v2
    return False


def _patched_get_unsupported_features(self) -> list[str]:
    unsupported = _original_get_unsupported_features(self)
    support = resolve_spec_pp_support(self)
    unsupported_feature = support.unsupported_feature if support is not None else None
    if unsupported_feature is not None and unsupported_feature in unsupported:
        unsupported.remove(unsupported_feature)
    return unsupported


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)
VllmConfig._get_v2_model_runner_unsupported_features = _patched_get_unsupported_features


def _patched_validate_v2_model_runner(self) -> None:
    if is_310p():
        return
    _original_validate_v2_model_runner(self)


VllmConfig._validate_v2_model_runner = _patched_validate_v2_model_runner

# vLLM main exposes this helper; v0.28.0 does not. Prefer hasattr over
# vllm_version_is(): CI installs from a commit SHA can report __version__="dev"
# and would otherwise apply the main-only patch on a release-lane checkout.
if hasattr(VllmConfig, "_get_v1_model_runner_unsupported_features"):
    _original_get_v1_model_runner_unsupported_features = VllmConfig._get_v1_model_runner_unsupported_features

    def _patched_get_v1_model_runner_unsupported_features(self) -> list[str]:
        unsupported = _original_get_v1_model_runner_unsupported_features(self)
        return [feature for feature in unsupported if feature not in _ASCEND_V1_SUPPORTED_FEATURES]

    VllmConfig._get_v1_model_runner_unsupported_features = _patched_get_v1_model_runner_unsupported_features
