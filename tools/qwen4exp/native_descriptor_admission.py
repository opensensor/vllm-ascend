# SPDX-License-Identifier: Apache-2.0
"""Read-only admission for append-only native descriptor dictionaries."""


def descriptors_available(resource, keys, *, capacity, is_capturing):
    """Admit all missing keys together without allocating or evicting entries.

    Existing descriptors remain usable at capacity and during graph capture.
    The resident runner serializes calls, so no other request can consume a
    descriptor slot between this metadata check and the admitted operation.
    """
    configs = getattr(resource, "_configs", None)
    if not isinstance(configs, dict):
        return False
    missing = {key for key in keys if key not in configs}
    if not missing:
        return True
    if len(configs) + len(missing) > capacity:
        return False
    return not is_capturing()
