"""Source selection helpers for pinned repository archives."""

from __future__ import annotations

import fnmatch

LINUX_DOCUMENTATION_DIRS = (
    "Documentation/admin-guide",
    "Documentation/core-api",
    "Documentation/scheduler",
    "Documentation/trace",
    "Documentation/bpf",
)


def matches(path: str, source: dict) -> bool:
    included = any(fnmatch.fnmatch(path, pattern) for pattern in source["include_globs"])
    excluded = any(fnmatch.fnmatch(path, pattern) for pattern in source["exclude_globs"])
    return included and not excluded


__all__ = ["LINUX_DOCUMENTATION_DIRS", "matches"]
