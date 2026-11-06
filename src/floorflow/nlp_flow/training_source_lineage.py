"""Verify training-only source bytes and declaration-fragment provenance."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.real_corpus_action_data import iter_declaration_fragments
from floorflow.nlp_flow.real_corpus_policy_data import action_from_record
from floorflow.nlp_flow.tree_action_decoder import decode_actions, render_tree_structure


def audit_training_sources(
    parents: Sequence[Mapping[str, Any]],
    fragments: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    source_root: Path,
    fragment_config: Mapping[str, object],
    grammars: Mapping[str, ASTTransitionGrammar],
) -> dict[str, object]:
    """Verify train split only, retaining failures without including source text."""
    permitted = source_root.resolve(strict=True)
    by_id: dict[str, Mapping[str, Any]] = {}
    extracted: dict[str, dict[str, Mapping[str, Any]]] = {}
    failures: list[dict[str, str]] = []
    for parent in parents:
        if parent.get("split") != "train":
            raise ValueError("only train split source content may enter this audit")
        case_id = str(parent["case_id"])
        if case_id in by_id:
            raise ValueError("duplicate training parent")
        by_id[case_id] = parent
        try:
            path = (root / str(parent["source_path"])).resolve(strict=True)
            if not path.is_relative_to(permitted) or not path.is_file():
                raise ValueError("source path is outside the allowed data root")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != parent["source_sha256"]:
                raise ValueError("source file hash mismatch")
            if data.decode("utf-8") != parent["source"]:
                raise ValueError("source text does not match pinned file")
            generated = iter_declaration_fragments([parent], fragment_config)
            extracted[case_id] = {str(row["case_id"]): row for row in generated}
        except (OSError, KeyError, TypeError, ValueError, UnicodeError) as error:
            failures.append({"case_id": case_id, "stage": "parent", "error": str(error)})
    seen: set[str] = set()
    verified = 0
    for fragment in fragments:
        if fragment.get("split") != "train":
            raise ValueError("only train split fragment content may enter this audit")
        case_id = str(fragment["case_id"])
        if case_id in seen:
            raise ValueError("duplicate training fragment")
        seen.add(case_id)
        parent_id, marker, _tail = case_id.rpartition("-fragment-")
        try:
            if not marker or parent_id not in by_id:
                raise ValueError("training fragment parent is missing")
            if parent_id not in extracted:
                raise ValueError("training fragment parent was not verified")
            source_row = extracted[parent_id].get(case_id)
            if source_row is None:
                raise ValueError("fragment not reproduced from pinned parent")
            if source_row["language"] != fragment["language"]:
                raise ValueError("fragment language mismatch")
            grammar = grammars[str(fragment["language"])]
            actions = [action_from_record(record) for record in fragment["actions"]]
            source = render_tree_structure(decode_actions(grammar, actions))
            actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if source != source_row["source"] or actual != fragment["target_source_sha256"]:
                raise ValueError("fragment replay or fingerprint mismatch")
            verified += 1
        except (KeyError, TypeError, ValueError, UnicodeError) as error:
            failures.append({"case_id": case_id, "stage": "fragment", "error": str(error)})
    return {
        "status": "pass" if parents and fragments and not failures else "partial_or_failed",
        "split": "train_only",
        "parent_rows": len(parents),
        "verified_parents": len(extracted),
        "fragment_rows": len(fragments),
        "verified_fragments": verified,
        "failures": failures,
        "evidence_boundary": (
            "Training source bytes and fragment replay only; validation/test content, clone families, "
            "benchmark overlap and external source authenticity are not verified."
        ),
    }
