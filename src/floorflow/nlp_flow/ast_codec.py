"""Canonical, reversible Python, Rust, and C++ AST graphs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import tree_sitter_cpp
import tree_sitter_python
import tree_sitter_rust
from tree_sitter import Language, Node, Parser

SERIALIZATION_VERSION = "nlp-ast-graph-v1"
SUPPORTED_LANGUAGES = ("python", "rust", "cpp")

_MODULES = {
    "python": tree_sitter_python,
    "rust": tree_sitter_rust,
    "cpp": tree_sitter_cpp,
}

_ROOTS = {
    "python": "module",
    "rust": "source_file",
    "cpp": "translation_unit",
}

_SCOPE_TYPES = {
    "python": frozenset({"module", "function_definition", "class_definition", "lambda"}),
    "rust": frozenset(
        {
            "source_file",
            "function_item",
            "impl_item",
            "mod_item",
            "trait_item",
            "closure_expression",
        }
    ),
    "cpp": frozenset(
        {
            "translation_unit",
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
            "lambda_expression",
        }
    ),
}

_SYMBOL_NODE_KINDS = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "mod_item": "module",
        "type_item": "type",
        "const_item": "constant",
        "static_item": "static",
    },
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "enum_specifier": "enum",
        "namespace_definition": "namespace",
        "type_definition": "type",
    },
}

_IDENTIFIER_TYPES = frozenset(
    {
        "identifier",
        "field_identifier",
        "type_identifier",
        "namespace_identifier",
        "primitive_type",
    }
)


@dataclass(frozen=True)
class ASTNode:
    """A source-ordered AST with deterministic scope and symbol metadata."""

    node_type: str
    start_byte: int
    end_byte: int
    named: bool
    children: tuple[tuple[str, ASTNode], ...]
    lexical_value: str | None = None
    gap_before: str = ""
    scope_id: int = 0
    symbol_name: str | None = None
    symbol_kind: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class ASTScope:
    scope_id: int
    node_type: str
    start_byte: int
    end_byte: int
    parent_scope_id: int | None
    path: tuple[int, ...]


@dataclass(frozen=True)
class ASTSymbol:
    name: str
    kind: str
    node_type: str
    scope_id: int
    scope_path: tuple[str, ...]
    start_byte: int
    end_byte: int


@dataclass(frozen=True)
class ASTGraph:
    language: str
    root: ASTNode
    scopes: tuple[ASTScope, ...]
    symbols: tuple[ASTSymbol, ...]
    source_sha256: str


@dataclass(frozen=True)
class ParserFingerprint:
    language: str
    package: str
    package_version: str
    abi_version: int
    semantic_version: tuple[int, int, int] | None
    native_sha256: str
    grammar_sha256: str


def language_for(name: str) -> Language:
    if name not in _MODULES:
        raise ValueError(f"unsupported AST language: {name}")
    return Language(_MODULES[name].language())



def parser_fingerprint(name: str) -> ParserFingerprint:
    """Return parser provenance through the dedicated parser helper."""

    from floorflow.nlp_flow.ast_codec_parser import (
        parser_fingerprint as implementation,
    )

    return implementation(name)


def parser_provenance(name: str) -> dict[str, object]:
    """Return frozen parser metadata through the dedicated helper."""

    from floorflow.nlp_flow.ast_codec_parser import (
        parser_provenance as implementation,
    )

    return implementation(name)


def _source_slice(source: bytes, node: Node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="strict")


def _first_identifier(node: Node, source: bytes) -> str | None:
    if node.type in _IDENTIFIER_TYPES:
        value = _source_slice(source, node).strip()
        return value or None
    for child in node.children:
        value = _first_identifier(child, source)
        if value is not None:
            return value
    return None


def _definition_name(node: Node, source: bytes) -> str | None:
    for index, child in enumerate(node.children):
        if node.field_name_for_child(index) in {"name", "declarator"}:
            return _first_identifier(child, source)
    for index, child in enumerate(node.children):
        if node.field_name_for_child(index) == "declarator":
            value = _first_identifier(child, source)
            if value is not None:
                return value
    return _first_identifier(node, source)


def _convert(
    node: Node,
    source: bytes,
    *,
    language: str,
    scope_counter: list[int],
    covered_until: list[int],
    current_scope: int | None = None,
    is_root: bool = False,
) -> ASTNode:
    lexical_value: str | None = None
    gap_before = ""
    if not node.children:
        lexical_value = _source_slice(source, node)
        gap_start = covered_until[0]
        gap_before = source[gap_start : node.start_byte].decode("utf-8", errors="strict")
        covered_until[0] = max(covered_until[0], node.end_byte)

    if node.type in _SCOPE_TYPES[language] or is_root:
        scope_id = scope_counter[0]
        scope_counter[0] += 1
    else:
        if current_scope is None:
            raise ValueError("non-root AST conversion requires a current scope")
        scope_id = current_scope

    symbol_name: str | None = None
    symbol_kind: str | None = None
    definition_kind = _SYMBOL_NODE_KINDS[language].get(node.type)
    if definition_kind is not None:
        symbol_name = _definition_name(node, source)
        symbol_kind = definition_kind if symbol_name else None
    children: list[tuple[str, ASTNode]] = []
    for index, child in enumerate(node.children):
        field = node.field_name_for_child(index) or "child"
        children.append(
            (
                field,
                _convert(
                    child,
                    source,
                    language=language,
                    scope_counter=scope_counter,
                    covered_until=covered_until,
                    current_scope=scope_id,
                ),
            )
        )

    if is_root and covered_until[0] < len(source):
        remaining = source[covered_until[0] :].decode("utf-8", errors="strict")
        if not children:
            lexical_value = (lexical_value or "") + remaining
        else:
            children.append(
                (
                    "trivia",
                    ASTNode(
                        node_type="trivia",
                        start_byte=covered_until[0],
                        end_byte=len(source),
                        named=False,
                        children=(),
                        lexical_value=remaining,
                        scope_id=scope_id,
                        language=None,
                    ),
                )
            )

    return ASTNode(
        node_type=node.type,
        start_byte=int(node.start_byte),
        end_byte=int(node.end_byte),
        named=bool(node.is_named),
        children=tuple(children),
        lexical_value=lexical_value,
        gap_before=gap_before,
        scope_id=scope_id,
        symbol_name=symbol_name,
        symbol_kind=symbol_kind,
        language=language if is_root else None,
    )


def parse_ast(source: str, language: str) -> ASTNode:
    """Parse source into a canonical AST or reject any parser error node."""

    if language not in _MODULES:
        raise ValueError(f"unsupported AST language: {language}")
    source_bytes = source.encode("utf-8")
    parser = Parser(language_for(language))
    tree = parser.parse(source_bytes)
    if tree.root_node.has_error:
        raise ValueError(f"{language} source contains a parse error")
    if tree.root_node.type != _ROOTS[language]:
        raise ValueError(f"{language} parser returned {tree.root_node.type}, expected {_ROOTS[language]}")
    return _convert(
        tree.root_node,
        source_bytes,
        language=language,
        scope_counter=[0],
        covered_until=[0],
        is_root=True,
    )


def _graph_metadata(root: ASTNode) -> tuple[tuple[ASTScope, ...], tuple[ASTSymbol, ...]]:
    scopes_by_id: dict[int, ASTScope] = {}
    symbols: list[ASTSymbol] = []

    def visit(
        node: ASTNode,
        path: tuple[int, ...],
        ancestors: tuple[int, ...],
    ) -> None:
        if node.scope_id not in scopes_by_id:
            parent_scope_id = ancestors[-1] if ancestors else None
            scopes_by_id[node.scope_id] = ASTScope(
                scope_id=node.scope_id,
                node_type=node.node_type,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                parent_scope_id=parent_scope_id,
                path=path,
            )
            scope_ancestors = ancestors + (node.scope_id,)
        else:
            scope_ancestors = ancestors
        if node.symbol_name is not None and node.symbol_kind is not None:
            owner_chain = []
            current_scope: int | None = node.scope_id
            while current_scope is not None:
                owner = scopes_by_id[current_scope]
                owner_chain.append(owner.node_type)
                current_scope = owner.parent_scope_id
            symbols.append(
                ASTSymbol(
                    name=node.symbol_name,
                    kind=node.symbol_kind,
                    node_type=node.node_type,
                    scope_id=node.scope_id,
                    scope_path=tuple(reversed(owner_chain)),
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                )
            )
        for index, (_field, child) in enumerate(node.children):
            visit(child, path + (index,), scope_ancestors)

    visit(root, (), ())
    ordered_scopes = tuple(scopes_by_id[index] for index in sorted(scopes_by_id))
    return ordered_scopes, tuple(symbols)


def parse_ast_graph(source: str, language: str) -> ASTGraph:
    """Parse source into an explicit tree, scope graph, and symbol graph."""

    root = parse_ast(source, language)
    scopes, symbols = _graph_metadata(root)
    return ASTGraph(
        language=language,
        root=root,
        scopes=scopes,
        symbols=symbols,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )



def serialize_ast(node: ASTNode) -> str:
    """Serialize a deterministic AST through the strict codec implementation."""

    from floorflow.nlp_flow.ast_codec_io import serialize_ast as implementation

    return implementation(node)


def serialize_ast_graph(graph: ASTGraph) -> str:
    """Serialize a complete AST graph through the strict codec implementation."""

    from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph as implementation

    return implementation(graph)


def deserialize_ast(serialized: str) -> ASTNode:
    """Deserialize and validate one canonical AST."""

    from floorflow.nlp_flow.ast_codec_io import deserialize_ast as implementation

    return implementation(serialized)


def deserialize_ast_graph(serialized: str) -> ASTGraph:
    """Deserialize and validate a canonical AST graph."""

    from floorflow.nlp_flow.ast_codec_io import deserialize_ast_graph as implementation

    return implementation(serialized)


def _walk(node: ASTNode, path: tuple[int, ...] = ()) -> Iterator[tuple[tuple[int, ...], ASTNode]]:
    yield path, node
    for index, (_field, child) in enumerate(node.children):
        yield from _walk(child, path + (index,))


def _leaves(node: ASTNode) -> Iterator[ASTNode]:
    if not node.children:
        yield node
        return
    for _field, child in node.children:
        yield from _leaves(child)


def render_source(node: ASTNode) -> str:
    """Reconstruct the exact source bytes represented by the AST payload."""

    parts: list[str] = []
    for leaf in _leaves(node):
        if leaf.lexical_value is None:
            continue
        parts.append(leaf.gap_before)
        parts.append(leaf.lexical_value)
    return "".join(parts)


def _validate_node_or_raise(node: ASTNode, *, source: str | None = None) -> None:
    if node.language not in _MODULES:
        raise ValueError("AST root must declare a supported language")
    scope_max = -1
    previous_end = -1
    nodes_by_path = dict(_walk(node))

    for path, current in _walk(node):
        if not current.node_type:
            raise ValueError(f"node {path} has an empty type")
        if current.start_byte < 0 or current.end_byte < current.start_byte:
            raise ValueError(f"node {path} has invalid byte span")
        if current.scope_id < 0:
            raise ValueError(f"node {path} has a negative scope id")
        if not path:
            if current.start_byte != 0:
                raise ValueError("AST root must start at byte zero")
        else:
            parent = nodes_by_path[path[:-1]]
            if current.start_byte < parent.start_byte or current.end_byte > parent.end_byte:
                raise ValueError(f"node {path} is outside its parent byte span")
        previous_end = max(previous_end, current.end_byte)
        scope_max = max(scope_max, current.scope_id)

        if current.children:
            if current.lexical_value is not None:
                raise ValueError(f"internal node {path} cannot carry a lexical value")
        else:
            if current.lexical_value is None:
                raise ValueError(f"leaf node {path} must carry a lexical value")
            if len(current.lexical_value.encode("utf-8")) != current.end_byte - current.start_byte:
                raise ValueError(f"leaf node {path} lexical length disagrees with its span")
        if (current.symbol_name is None) != (current.symbol_kind is None):
            raise ValueError(f"node {path} must define both symbol name and kind or neither")
        if current.symbol_name is not None and not current.symbol_name:
            raise ValueError(f"node {path} has an empty symbol")

    if node.end_byte < previous_end:
        raise ValueError("AST root end precedes one of its descendants")
    observed_scopes = {current.scope_id for _path, current in _walk(node)}
    if observed_scopes != set(range(scope_max + 1)):
        raise ValueError("AST scope ids must be dense and zero-based")

    rendered = render_source(node)
    if source is not None and rendered != source:
        raise ValueError("AST does not render to the supplied source")
    reparsed = parse_ast(rendered, node.language)
    if reparsed != node:
        raise ValueError("AST is not a lossless canonical parse of its rendered source")


def _validate_ast_or_raise(graph: ASTNode | ASTGraph, *, source: str | None = None) -> None:
    if isinstance(graph, ASTGraph):
        if graph.root.language != graph.language:
            raise ValueError("AST graph language disagrees with its root")
        _validate_node_or_raise(graph.root, source=source)
        rendered = render_source(graph.root)
        source_sha256 = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        if source_sha256 != graph.source_sha256:
            raise ValueError("AST graph source hash mismatch")
        expected_scopes, expected_symbols = _graph_metadata(graph.root)
        if graph.scopes != expected_scopes:
            raise ValueError("AST graph scope relations are not canonical")
        if graph.symbols != expected_symbols:
            raise ValueError("AST graph symbol relations are not canonical")
        return
    _validate_node_or_raise(graph, source=source)


def validate_ast(graph: ASTNode | ASTGraph, *, source: str | None = None) -> bool:
    """Return whether a node or graph satisfies every reversible AST invariant."""

    try:
        _validate_ast_or_raise(graph, source=source)
    except (IndexError, KeyError, RecursionError, TypeError, UnicodeError, ValueError):
        return False
    return True


def is_parse_valid(node: ASTNode | ASTGraph) -> bool:
    if isinstance(node, ASTGraph):
        return validate_ast(node) and is_parse_valid(node.root)
    if node.language not in _MODULES:
        return False
    try:
        reparsed = parse_ast(render_source(node), node.language)
    except (TypeError, ValueError, UnicodeError):
        return False
    return reparsed == node



def tree_edit_distance(left: ASTNode | ASTGraph, right: ASTNode | ASTGraph) -> int:
    """Return ordered tree edit distance through the metrics implementation."""

    from floorflow.nlp_flow.ast_codec_metrics import tree_edit_distance as implementation

    return implementation(left, right)


def normalized_tree_edit_distance(
    left: ASTNode | ASTGraph,
    right: ASTNode | ASTGraph,
) -> float:
    """Return normalized tree edit distance through the metrics implementation."""

    from floorflow.nlp_flow.ast_codec_metrics import (
        normalized_tree_edit_distance as implementation,
    )

    return implementation(left, right)


def ast_similarity(
    predicted: ASTNode | ASTGraph,
    reference: ASTNode | ASTGraph,
) -> dict[str, float]:
    """Return the frozen Stage-A structural metrics."""

    from floorflow.nlp_flow.ast_codec_metrics import ast_similarity as implementation

    return implementation(predicted, reference)


def ast_metrics(
    predicted: ASTNode | ASTGraph,
    reference: ASTNode | ASTGraph,
) -> dict[str, float | bool]:
    """Return all frozen AST metrics."""

    from floorflow.nlp_flow.ast_codec_metrics import ast_metrics as implementation

    return implementation(predicted, reference)


__all__ = [
    "SERIALIZATION_VERSION",
    "SUPPORTED_LANGUAGES",
    "ASTGraph",
    "ASTNode",
    "ASTScope",
    "ASTSymbol",
    "ParserFingerprint",
    "ast_metrics",
    "ast_similarity",
    "deserialize_ast",
    "deserialize_ast_graph",
    "is_parse_valid",
    "language_for",
    "normalized_tree_edit_distance",
    "parse_ast",
    "parse_ast_graph",
    "parser_fingerprint",
    "parser_provenance",
    "render_source",
    "serialize_ast",
    "serialize_ast_graph",
    "tree_edit_distance",
    "validate_ast",
]
