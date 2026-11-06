import ast
import inspect
import json
from pathlib import Path

import pytest

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.ast_oracle_controls import oracle_projection, render_oracle_projection
from floorflow.nlp_flow.prefix_tree_runtime import generate_prefix_tree
from floorflow.nlp_flow.real_corpus_tree_runtime import generate_real_task
from floorflow.nlp_flow.tree_action_decoder import decode_actions, render_tree_structure, trace_teacher_actions

ROOT = Path(__file__).resolve().parents[1]


def _structure(source: str) -> dict:
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    graph = json.loads(serialize_ast_graph(parse_ast_graph(source, "python")))
    return decode_actions(grammar, trace_teacher_actions(grammar, graph))


def _leaves(structure: dict) -> list[int]:
    return [node["index"] for node in structure["nodes"] if not node["children"]]


def test_oracle_separates_lexical_values_from_topology_without_mutation() -> None:
    teacher = _structure("x = 1\n")
    candidate = _structure("y = 2\n")
    alignment = dict(zip(_leaves(teacher), _leaves(candidate), strict=True))
    before_teacher = json.dumps(teacher, sort_keys=True)
    before_candidate = json.dumps(candidate, sort_keys=True)
    projected = oracle_projection(teacher, candidate, alignment)

    assert render_tree_structure(projected) == "y = 2\n"
    assert [node["children"] for node in projected["nodes"]] == [node["children"] for node in teacher["nodes"]]
    assert render_oracle_projection(candidate, teacher, {value: key for key, value in alignment.items()}) == "x = 1\n"
    assert json.dumps(teacher, sort_keys=True) == before_teacher
    assert json.dumps(candidate, sort_keys=True) == before_candidate


def test_oracle_rejects_missing_or_incompatible_alignment() -> None:
    teacher = _structure("x = 1\n")
    candidate = _structure("y = 2\n")
    alignment = dict(zip(_leaves(teacher), _leaves(candidate), strict=True))
    with pytest.raises(ValueError, match="cover both trees"):
        oracle_projection(teacher, candidate, {})
    values = list(alignment.values())
    keys = list(alignment)
    swapped = dict(alignment)
    swapped[keys[0]], swapped[keys[-1]] = values[-1], values[0]
    with pytest.raises(ValueError, match="incompatible"):
        oracle_projection(teacher, candidate, swapped)


from copy import deepcopy


def test_oracle_keeps_selected_topology_when_leaf_order_differs() -> None:
    teacher = _structure("x=y\n")
    candidate = deepcopy(teacher)
    assignment = next(node for node in candidate["nodes"] if node["kind"] == "assignment")
    assignment["children"] = list(reversed(assignment["children"]))
    identity = {index: index for index in _leaves(teacher)}
    assert render_oracle_projection(teacher, teacher, identity) == "x=y\n"
    assert render_oracle_projection(candidate, teacher, identity) == "y=x\n"


def test_normal_generation_entrypoints_have_no_oracle_target_parameters() -> None:
    for generator in (generate_prefix_tree, generate_real_task):
        parameters = inspect.signature(generator).parameters
        assert not any("target" in key or "oracle" in key or "alignment" in key for key in parameters)
        module = ast.parse(inspect.getsource(inspect.getmodule(generator)))
        imports = [node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom)]
        assert "floorflow.nlp_flow.ast_oracle_controls" not in imports
