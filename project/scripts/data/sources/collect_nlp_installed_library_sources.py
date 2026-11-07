#!/usr/bin/env python3
"""Import Python source from locked, installed open-source libraries."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PACKAGES = {
    "torch": {
        "license": "BSD-3-Clause",
        "source_url": "https://github.com/pytorch/pytorch",
    },
    "transformers": {
        "license": "Apache-2.0",
        "source_url": "https://github.com/huggingface/transformers",
    },
    "numpy": {
        "license": "BSD-3-Clause",
        "source_url": "https://github.com/numpy/numpy",
    },
    "scipy": {
        "license": "BSD-3-Clause",
        "source_url": "https://github.com/scipy/scipy",
    },
    "sklearn": {
        "license": "BSD-3-Clause",
        "source_url": "https://github.com/scikit-learn/scikit-learn",
        "metadata_name": "scikit-learn",
    },
    "pandas": {
        "license": "BSD-3-Clause",
        "source_url": "https://github.com/pandas-dev/pandas",
    },
    "sympy": {
        "license": "BSD-3-Clause",
        "source_url": "https://github.com/sympy/sympy",
    },
    "networkx": {
        "license": "BSD-3-Clause",
        "source_url": "https://github.com/networkx/networkx",
    },
}
MAX_FILE_BYTES = 2_000_000


def main() -> None:
    output_root = ROOT / ".data/nlp/mixed_v1"
    records = []
    for package, metadata in PACKAGES.items():
        spec = importlib.util.find_spec(package)
        if spec is None or spec.submodule_search_locations is None:
            raise RuntimeError(f"installed package not found: {package}")
        source_root = Path(next(iter(spec.submodule_search_locations))).resolve()
        version = importlib.metadata.version(
            str(metadata.get("metadata_name", package))
        )
        source_id = f"installed_{package}"
        target_root = output_root / "code" / "python" / source_id
        for source in sorted(source_root.rglob("*.py")):
            if "__pycache__" in source.parts or source.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                source.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            relative = source.relative_to(source_root)
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            content = target.read_bytes()
            records.append(
                {
                    "source_id": source_id,
                    "repository": metadata["source_url"],
                    "revision": f"installed-{package}-{version}",
                    "license": metadata["license"],
                    "usage": "training",
                    "language": "python",
                    "family": "code",
                    "path": str(target),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    report = {
        "version": "nlp-installed-library-source-v1",
        "files": records,
        "total_files": len(records),
        "total_bytes": sum(int(row["bytes"]) for row in records),
    }
    (output_root / "local_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"files": len(records), "bytes": report["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
