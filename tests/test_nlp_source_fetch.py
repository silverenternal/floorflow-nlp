from __future__ import annotations

from floorflow.nlp_flow.source_fetch import LINUX_DOCUMENTATION_DIRS, matches


def test_source_globs_include_and_exclude() -> None:
    source = {
        "include_globs": ["Doc/**/*.rst", "Lib/**/*.py"],
        "exclude_globs": ["Lib/test/**"],
    }

    assert matches("Doc/library/asyncio.rst", source)
    assert matches("Lib/asyncio/tasks.py", source)
    assert not matches("Lib/test/test_asyncio.py", source)
    assert not matches("Objects/object.c", source)


def test_linux_sparse_directories_are_documentation_only() -> None:
    assert LINUX_DOCUMENTATION_DIRS
    assert all(path.startswith("Documentation/") for path in LINUX_DOCUMENTATION_DIRS)
