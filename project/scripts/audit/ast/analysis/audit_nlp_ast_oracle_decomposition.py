#!/usr/bin/env python3
"""Exercise offline oracle controls on explicitly synthetic CPU fixtures only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.ast.diagnostics.recovery.ast_oracle_controls import render_oracle_projection
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import decode_actions, trace_teacher_actions

ROOT = Path(__file__).resolve().parents[5]


def fixture_structure(source: str) -> dict:
    """Parse a declared synthetic source without consulting any dataset."""
    grammar = load_ast_grammar(ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json", "python")
    payload = json.loads(serialize_ast_graph(parse_ast_graph(source, "python")))
    return decode_actions(grammar, trace_teacher_actions(grammar, payload))


def run_fixture() -> dict:
    """Verify all four diagnostic combinations and preserve their oracle labels."""
    teacher = fixture_structure("fixture_teacher = 1\n")
    candidate = fixture_structure("fixture_candidate = 2\n")
    teacher_leaves = [node["index"] for node in teacher["nodes"] if not node["children"]]
    candidate_leaves = [node["index"] for node in candidate["nodes"] if not node["children"]]
    forward = dict(zip(teacher_leaves, candidate_leaves, strict=True))
    backward = {candidate_id: teacher_id for teacher_id, candidate_id in forward.items()}
    cases = {
        "teacher_topology_teacher_lexical": (teacher, teacher, {key: key for key in teacher_leaves}),
        "teacher_topology_candidate_lexical": (teacher, candidate, forward),
        "candidate_topology_teacher_lexical": (candidate, teacher, backward),
        "candidate_topology_candidate_lexical": (candidate, candidate, {key: key for key in candidate_leaves}),
    }
    expected = {
        "teacher_topology_teacher_lexical": "fixture_teacher = 1\n",
        "teacher_topology_candidate_lexical": "fixture_candidate = 2\n",
        "candidate_topology_teacher_lexical": "fixture_teacher = 1\n",
        "candidate_topology_candidate_lexical": "fixture_candidate = 2\n",
    }
    outputs = {}
    for name, (topology, lexical, alignment) in cases.items():
        source = render_oracle_projection(topology, lexical, alignment)
        parse_ast_graph(source, "python")
        outputs[name] = {"matches_fixture": source == expected[name], "parse_valid": True}
    return {
        "version": "nlp-ast-oracle-fixture-v11",
        "status": "pass" if all(row["matches_fixture"] for row in outputs.values()) else "fail",
        "split": "synthetic_fixture_only",
        "uses_target_oracle": True,
        "arms": outputs,
        "evidence_boundary": "Fixture wiring only; no model quality, upper bound or validation claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_ast_oracle_fixture_v11.json")
    args = parser.parse_args()
    report = run_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
