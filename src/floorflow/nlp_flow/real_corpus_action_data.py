"""Build prompt/action supervision from the frozen parser-valid AST corpus."""

from __future__ import annotations

import ast as python_ast
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from floorflow.nlp_flow.ast_codec import ASTNode, parse_ast_graph, parser_provenance
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.tree_action_decoder import (
    AttachEdgeAction,
    CloseScopeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TerminateAction,
    TreeAction,
    decode_actions,
    render_tree_structure,
    trace_teacher_actions,
)


@dataclass(frozen=True)
class RealCorpusActionRow:
    case_id: str
    language: str
    split: str
    prompt: str
    prompt_source: str
    target_source_sha256: str
    parser_grammar_sha256: str
    action_count: int
    actions: tuple[dict[str, object], ...]
    prompt_metadata: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "language": self.language,
            "split": self.split,
            "prompt": self.prompt,
            "prompt_source": self.prompt_source,
            "target_source_sha256": self.target_source_sha256,
            "parser_grammar_sha256": self.parser_grammar_sha256,
            "action_count": self.action_count,
            "actions": list(self.actions),
            "prompt_metadata": self.prompt_metadata,
        }


def action_record(action: TreeAction) -> dict[str, object]:
    payload = asdict(action)
    payload["type"] = {
        AttachEdgeAction: "attach_edge",
        OpenNodeAction: "open_node",
        OpenScopeAction: "open_scope",
        DeclareLeafAction: "declare_leaf",
        CloseScopeAction: "close_scope",
        TerminateAction: "terminate",
    }[type(action)]
    return payload


def _walk_payload(node: Mapping[str, object]):
    yield node
    for child in node["children"]:
        yield from _walk_payload(child["node"])


def _node_count(node: ASTNode) -> int:
    return 1 + sum(_node_count(child) for _field, child in node.children)


def _comment_text(source: str, payload: Mapping[str, object]) -> str:
    fragments = []
    code_lines = {
        line.strip()
        for line in source.splitlines()
        if line.strip()
        and not line.lstrip().startswith(("#", "//", "/*", "*", "*/"))
    }
    for node in _walk_payload(payload["root"]):
        node_type = str(node["type"]).lower()
        if "comment" not in node_type:
            continue
        lexical = node.get("lexical")
        if not isinstance(lexical, str):
            continue
        cleaned = lexical
        for marker in ("#", "//", "/*", "*/"):
            cleaned = cleaned.replace(marker, " ")
        cleaned = re.sub(r"^\s*\*+\s?", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if (
            cleaned
            and cleaned not in code_lines
            and not cleaned.startswith(("include", "define", "pragma"))
        ):
            fragments.append(cleaned)
    return " ".join(fragments)


def _docstring_text(
    source: str,
    payload: Mapping[str, object],
    language: str,
) -> str:
    if language != "python":
        return ""
    encoded = source.encode("utf-8")

    def visit(node: Mapping[str, object]) -> str:
        if node["type"] in {"function_definition", "class_definition"}:
            for child in node["children"]:
                if child["field"] != "body":
                    continue
                block = child["node"]
                for statement in block["children"]:
                    expression = statement["node"]
                    if expression["type"] != "expression_statement":
                        continue
                    for item in expression["children"]:
                        string_node = item["node"]
                        if string_node["type"] != "string":
                            continue
                        literal = encoded[
                            int(string_node["start"]) : int(string_node["end"])
                        ].decode("utf-8")
                        try:
                            value = python_ast.literal_eval(literal)
                        except (SyntaxError, ValueError):
                            continue
                        if isinstance(value, str):
                            return " ".join(value.split())
                    break
        for child in node["children"]:
            value = visit(child["node"])
            if value:
                return value
        return ""

    return visit(payload["root"])


def _symbol_summary(payload: Mapping[str, object], limit: int) -> tuple[str, list[str], list[str]]:
    names = []
    kinds = []
    for symbol in payload["symbols"]:
        name = str(symbol["name"])
        kind = str(symbol["kind"])
        if name not in names:
            names.append(name)
            kinds.append(kind)
        if len(names) >= limit:
            break
    declarations = ", ".join(
        f"{kind} {name}" for name, kind in zip(names, kinds, strict=True)
    )
    return declarations, names, kinds


def _semantic_tokens(
    payload: Mapping[str, object],
    *,
    max_identifiers: int,
    max_operators: int,
    max_literals: int,
    max_constructs: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    identifiers = set()
    operators = set()
    literals = set()
    constructs = set()
    structural = {
        "function_definition",
        "function_item",
        "class_definition",
        "class_specifier",
        "struct_specifier",
        "impl_item",
        "trait_item",
        "return_statement",
        "if_statement",
        "if_expression",
        "for_statement",
        "for_expression",
        "while_statement",
        "while_expression",
        "match_expression",
        "binary_operator",
        "binary_expression",
        "call_expression",
        "assignment",
        "assignment_expression",
    }
    operator_symbols = {
        "+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">=",
        "&&", "||", "&", "|", "^", "<<", ">>", "=", "+=", "-=",
        "*=", "/=",
    }
    for node in _walk_payload(payload["root"]):
        node_type = str(node["type"])
        lexical = node.get("lexical")
        if node_type in structural:
            constructs.add(node_type)
        if not isinstance(lexical, str):
            continue
        if "identifier" in node_type and len(identifiers) < max_identifiers:
            identifiers.add(lexical)
        elif lexical in operator_symbols and len(operators) < max_operators:
            operators.add(lexical)
        elif any(
            token in node_type
            for token in ("integer", "float", "number", "string_literal")
        ) and len(literals) < max_literals:
            literals.add(lexical)
    return (
        sorted(identifiers),
        sorted(operators),
        sorted(literals),
        sorted(constructs)[:max_constructs],
    )


def build_prompt(
    language: str,
    source: str,
    payload: Mapping[str, object],
    *,
    max_prompt_chars: int,
    max_symbol_names: int,
    max_identifiers: int,
    max_operators: int,
    max_literals: int,
    max_constructs: int,
    max_comment_fragments: int,
    max_comment_chars: int,
    max_docstring_chars: int,
) -> tuple[str, str, dict[str, object]]:
    declarations, names, kinds = _symbol_summary(payload, max_symbol_names)
    identifiers, operators, literals, constructs = _semantic_tokens(
        payload,
        max_identifiers=max_identifiers,
        max_operators=max_operators,
        max_literals=max_literals,
        max_constructs=max_constructs,
    )
    comments = _comment_text(source, payload)
    docstring = _docstring_text(source, payload, language)[
        :max_docstring_chars
    ]
    fragments = [
        fragment
        for fragment in re.split(r"(?<=[.!?])\s+", comments)
        if fragment
    ][:max_comment_fragments]
    description = " ".join(
        value for value in (docstring, " ".join(fragments)[:max_comment_chars])
        if value
    ).strip()
    language_name = {"python": "Python", "rust": "Rust", "cpp": "C++"}[language]
    parts = [f"Write one complete parser-valid {language_name} module."]
    if description:
        parts.append(f"Natural-language intent: {description}")
    if declarations:
        parts.append(f"Required declarations: {declarations}.")
    if identifiers:
        parts.append(f"Identifiers: {' | '.join(identifiers)}.")
    if operators:
        parts.append(f"Operators: {' '.join(operators)}.")
    if literals:
        parts.append(f"Literal values: {' | '.join(literals)}.")
    if constructs:
        parts.append(f"Required structure: {', '.join(constructs)}.")
    parts.append("Preserve every required identifier and return only the source program.")
    prompt = " ".join(parts)[:max_prompt_chars]
    prompt_source = (
        "docstring_comments_symbols"
        if docstring and fragments
        else "docstring_symbols"
        if docstring
        else "comments_symbols"
        if description
        else "symbols"
        if declarations
        else "metadata"
    )
    return prompt, prompt_source, {
        "symbol_names": names,
        "symbol_kinds": kinds,
        "comment_fragments": len(fragments),
        "description": description,
        "identifiers": identifiers,
        "operators": operators,
        "literals": literals,
        "constructs": constructs,
    }


def prompt_answer_leaks(
    *,
    prompt: str,
    source: str,
    ast_payload: Mapping[str, object],
    minimum_line_chars: int,
    allowed_tokens: Sequence[str] = (),
    allowed_description_lines: Sequence[str] = (),
) -> dict[str, object]:
    source_lines = [
        line.strip()
        for line in source.splitlines()
        if len(line.strip()) >= minimum_line_chars
    ]
    allowed = set(allowed_tokens)
    allowed_descriptions = set(allowed_description_lines)
    matching_lines = [line for line in source_lines if line in prompt]
    semantic_token_lines = [line for line in matching_lines if line in allowed]
    description_lines = [
        line for line in matching_lines
        if line in allowed_descriptions and line not in allowed
    ]
    leaked_lines = [
        line for line in matching_lines
        if line not in allowed and line not in allowed_descriptions
    ]
    serialized_ast = json.dumps(ast_payload, sort_keys=True)
    checks = {
        "target_source_present": source.strip() in prompt if source.strip() else False,
        "target_ast_present": serialized_ast in prompt,
        "complete_source_lines_present": bool(leaked_lines),
        "leaked_source_lines": leaked_lines,
        "allowed_semantic_token_lines": semantic_token_lines,
        "allowed_description_lines": description_lines,
    }
    checks["pass"] = not any(
        checks[name]
        for name in (
            "target_source_present",
            "target_ast_present",
            "complete_source_lines_present",
        )
    )
    return checks


def iter_declaration_fragments(
    rows: Sequence[Mapping[str, object]],
    fragment_config: Mapping[str, object],
) -> list[dict[str, object]]:
    declaration_types = {
        language: set(values)
        for language, values in fragment_config["declaration_types"].items()
    }
    maximum = int(fragment_config["max_fragments_per_row"])
    max_chars = int(fragment_config["max_fragment_chars"])
    max_nodes = int(fragment_config["max_fragment_nodes"])
    output = []
    seen = set()
    for row in rows:
        language = str(row["language"])
        source = str(row["source"])
        graph = parse_ast_graph(source, language)
        encoded = source.encode("utf-8")
        selected = [
            child
            for _field, child in graph.root.children
            if child.node_type in declaration_types.get(language, set())
            and child.end_byte - child.start_byte <= max_chars
        ][:maximum]
        for index, node in enumerate(selected):
            fragment = encoded[node.start_byte : node.end_byte].decode("utf-8")
            fragment = fragment.rstrip() + "\n"
            digest = hashlib.sha256(fragment.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            try:
                fragment_graph = parse_ast_graph(fragment, language)
            except (TypeError, ValueError, UnicodeError):
                continue
            if _node_count(fragment_graph.root) > max_nodes:
                continue
            output.append(
                {
                    "case_id": f"{row['case_id']}-fragment-{index:02d}-{digest[:8]}",
                    "language": language,
                    "split": row["split"],
                    "source": fragment,
                    "source_sha256": digest,
                    "ast": serialize_ast_graph(fragment_graph),
                    "fragment_kind": node.node_type,
                    "parent_case_id": row["case_id"],
                }
            )
    return output


def build_real_corpus_action_dataset(
    rows: Sequence[Mapping[str, object]],
    grammars: Mapping[str, ASTTransitionGrammar],
    prompt_config: Mapping[str, object],
    fragment_config: Mapping[str, object] | None = None,
) -> tuple[list[RealCorpusActionRow], dict[str, object]]:
    if fragment_config is not None and fragment_config.get("enabled", False):
        rows = iter_declaration_fragments(rows, fragment_config)
    output = []
    failures = []
    leakage = []
    action_counts: Counter[str] = Counter()
    replay_exact = 0
    canonical_exact = 0
    for row in rows:
        language = str(row["language"])
        source = str(row["source"])
        payload = json.loads(str(row["ast"]))
        prompt, prompt_source, prompt_metadata = build_prompt(
            language,
            source,
            payload,
            max_prompt_chars=int(prompt_config["max_prompt_chars"]),
            max_symbol_names=int(prompt_config["max_symbol_names"]),
            max_identifiers=int(prompt_config["max_identifiers"]),
            max_operators=int(prompt_config["max_operators"]),
            max_literals=int(prompt_config["max_literals"]),
            max_constructs=int(prompt_config["max_constructs"]),
            max_comment_fragments=int(prompt_config["max_comment_fragments"]),
            max_comment_chars=int(prompt_config["max_comment_chars"]),
            max_docstring_chars=int(prompt_config["max_docstring_chars"]),
        )
        leak = prompt_answer_leaks(
            prompt=prompt,
            source=source,
            ast_payload=payload,
            minimum_line_chars=int(
                prompt_config["minimum_source_line_chars_for_leak_check"]
            ),
            allowed_tokens=(
                prompt_metadata["identifiers"]
                + prompt_metadata["operators"]
                + prompt_metadata["literals"]
            ),
            allowed_description_lines=[
                line.strip()
                for line in source.splitlines()
                if line.strip()
                and line.strip() in prompt_metadata["description"]
            ],
        )
        if not leak["pass"]:
            leakage.append({"case_id": row["case_id"], **leak})
        try:
            actions = trace_teacher_actions(grammars[language], payload)
            structure = decode_actions(grammars[language], actions)
            rendered = render_tree_structure(structure)
            replay_exact += int(rendered == source)
            reparsed = parse_ast_graph(rendered, language)
            canonical_exact += int(
                json.loads(serialize_ast_graph(reparsed)) == payload
            )
        except (KeyError, TypeError, ValueError, UnicodeError) as error:
            failures.append({"case_id": row["case_id"], "error": str(error)})
            continue
        output.append(
            RealCorpusActionRow(
                case_id=str(row["case_id"]),
                language=language,
                split=str(row["split"]),
                prompt=prompt,
                prompt_source=prompt_source,
                target_source_sha256=str(row["source_sha256"]),
                parser_grammar_sha256=str(
                    parser_provenance(language)["grammar_sha256"]
                ),
                action_count=len(actions),
                actions=tuple(action_record(action) for action in actions),
                prompt_metadata=prompt_metadata,
            )
        )
        action_counts[language] += len(actions)
    report = {
        "version": "nlp-tree-action-corpus-audit-v1",
        "status": (
            "pass"
            if output
            and not failures
            and not leakage
            and replay_exact == len(output)
            and canonical_exact == len(output)
            else "fail"
        ),
        "rows": len(output),
        "by_language": dict(Counter(row.language for row in output)),
        "by_split": dict(Counter(row.split for row in output)),
        "fragment_granularity": (
            "top_level_declaration"
            if fragment_config is not None and fragment_config.get("enabled", False)
            else "whole_module"
        ),
        "action_counts": dict(action_counts),
        "exact_source_replay_rate": replay_exact / max(1, len(output)),
        "canonical_ast_roundtrip_rate": canonical_exact / max(1, len(output)),
        "prompt_leakage": leakage,
        "failures": failures,
        "target_ast_input": False,
        "target_source_input": False,
        "repair_or_retry": False,
        "evidence_boundary": (
            "Teacher-action corpus and prompt non-leakage audit only; no "
            "target-free generation or capability claim."
        ),
    }
    return output, report


def write_real_corpus_action_dataset(
    rows: Sequence[RealCorpusActionRow],
    *,
    output: Path,
) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row.as_dict(), sort_keys=True) + "\n"
        for row in rows
    )
    output.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "RealCorpusActionRow",
    "action_record",
    "build_prompt",
    "build_real_corpus_action_dataset",
    "iter_declaration_fragments",
    "prompt_answer_leaks",
    "write_real_corpus_action_dataset",
]
