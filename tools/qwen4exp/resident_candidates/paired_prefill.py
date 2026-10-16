# SPDX-License-Identifier: Apache-2.0
"""Lossless paired-limb prefill trial over the resident W4 banks."""

from tools.qwen4exp.resident_candidates import grouped_prefill

RESOURCE_NAME = "qwen_prefill_v27"


def replacements(native_resources):
    # Keep the admitted shape/size policy and the existing decode fallback.
    # The manifest binds projection to the paired entry; no W8 bank is loaded.
    return grouped_prefill.replacements({grouped_prefill.RESOURCE_NAME: native_resources[RESOURCE_NAME]})
