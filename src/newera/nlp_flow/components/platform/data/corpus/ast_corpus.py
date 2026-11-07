"""Bounded parser-valid AST/source corpus construction."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from newera.nlp_flow.components.ast.representation.codec.ast_codec import (
    ASTGraph,
    ASTNode,
    parse_ast_graph,
    parser_provenance,
    render_source,
)
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.platform.data.splits.ast_holdout import construct_name_digest


@dataclass(frozen=True)
class ASTCorpusRow:
    case_id: str
    language: str
    source_id: str
    source_path: str
    source_sha256: str
    parser: Mapping[str, object]
    license: str
    license_url: str
    repository: str
    revision: str
    source_url: str
    split: str
    node_count: int
    edge_count: int
    max_depth: int
    scope_count: int
    symbol_count: int
    constructs: tuple[str, ...]
    source: str
    ast_json: str

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "language": self.language,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "parser": dict(self.parser),
            "license": self.license,
            "license_url": self.license_url,
            "repository": self.repository,
            "revision": self.revision,
            "source_url": self.source_url,
            "split": self.split,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "max_depth": self.max_depth,
            "scope_count": self.scope_count,
            "symbol_count": self.symbol_count,
            "constructs": list(self.constructs),
            "source": self.source,
            "ast": self.ast_json,
        }


def _tree_shape(root: ASTNode) -> tuple[int, int, int]:
    nodes = 0
    edges = 0
    maximum = 0

    def visit(node: ASTNode, depth: int) -> None:
        nonlocal nodes, edges, maximum
        nodes += 1
        edges += len(node.children)
        maximum = max(maximum, depth)
        for _field, child in node.children:
            visit(child, depth + 1)

    visit(root, 1)
    return nodes, edges, maximum


def _constructs(graph: ASTGraph, source: str) -> tuple[str, ...]:
    names = {node.node_type for _path, node in _walk(graph.root)}
    groups = {
        "function": {"function_definition", "function_item"},
        "class": {"class_definition", "class_specifier", "struct_specifier"},
        "impl_trait": {"impl_item", "trait_item"},
        "control_flow": {
            "if_statement",
            "for_statement",
            "while_statement",
            "match_expression",
            "for_expression",
        },
        "macro": {"macro_invocation", "macro_definition"},
        "template": {"template_declaration"},
        "namespace": {"namespace_definition"},
        "lambda_closure": {"lambda", "closure_expression", "lambda_expression"},
    }
    labels = {
        name
        for name, members in groups.items()
        if names.intersection(members)
    }
    labels.update(
        f"macro_name:{value}"
        for value in re.findall(
            r"(?m)^\s*#\s*define\s+([A-Za-z_]\w*)",
            source,
        )
    )
    labels.update(f"attribute_name:{value}" for value in _attribute_names(graph, source))
    return tuple(sorted(labels))


def _attribute_names(graph: ASTGraph, source: str) -> set[str]:
    """Extract every Rust attribute and derive target from parsed attribute nodes."""
    encoded = source.encode("utf-8")
    names: set[str] = set()
    for _path, node in _walk(graph.root):
        if node.node_type not in {"attribute_item", "inner_attribute_item"}:
            continue
        snippet = encoded[node.start_byte : node.end_byte].decode("utf-8")
        attribute = re.match(r"#!?\[\s*([A-Za-z_]\w*)", snippet)
        if attribute is not None:
            names.add(attribute.group(1))
        derive = re.search(r"\bderive\s*\((.*?)\)", snippet, flags=re.DOTALL)
        if derive is None:
            continue
        for argument in derive.group(1).split(","):
            path = re.findall(r"[A-Za-z_]\w*", argument)
            if path:
                names.add(path[-1])
    return names


def _walk(node: ASTNode) -> Iterable[tuple[tuple[int, ...], ASTNode]]:
    yield (), node
    for index, (_field, child) in enumerate(node.children):
        for path, value in _walk(child):
            yield (index, *path), value


def _split_for_digest(
    digest: str,
    constructs: tuple[str, ...],
    split: Mapping[str, object],
) -> str:
    heldout = {str(value) for value in split.get("heldout_constructs", ())}
    if heldout.intersection(constructs):
        return "test"
    for construct in constructs:
        for prefix in split.get("heldout_name_prefixes", ()):
            if not construct.startswith(str(prefix)):
                continue
            if construct_name_digest(construct) >= int(split["heldout_name_modulus"]):
                return "test"
    bucket = int(digest[:8], 16) % 100
    if bucket < int(split["train_modulus"]):
        return "train"
    if bucket < int(split["validation_modulus"]):
        return "validation"
    return "test"


def _eligible_path(
    path: Path,
    root: Path,
    config: Mapping[str, object],
    extensions: tuple[str, ...],
) -> bool:
    if path.suffix not in extensions:
        return False
    relative = path.relative_to(root)
    excluded = {str(value).lower() for value in config["exclude_path_parts"]}
    return not any(part.lower() in excluded for part in relative.parts[:-1])


def iter_train_ast_graphs(
    config: Mapping[str, object],
    *,
    root: Path,
) -> Iterable[tuple[str, Path, str, ASTGraph]]:
    """Yield parser-valid grammar sources assigned to the train split only."""
    registry = json.loads(
        (root / str(config["source_registry"])).read_text(encoding="utf-8")
    )
    source_meta = {str(row["id"]): row for row in registry["sources"]}
    corpus_root = root / str(config["root"])
    scan = config.get("grammar_scan", {})
    max_bytes = int(scan.get("max_source_bytes", 2_000_000))
    max_files = int(scan.get("max_files_per_language", 20_000))
    seen: set[str] = set()
    for language, sources in config["sources"].items():
        scanned = 0
        for source in sources:
            source_id = str(source["id"])
            metadata = source_meta.get(source_id)
            if metadata is None or metadata["usage"] != "training":
                raise ValueError(f"AST source is not training-approved: {source_id}")
            base = corpus_root / str(source["directory"])
            extensions = tuple(str(value) for value in source["extensions"])
            candidates = sorted(
                path
                for path in base.rglob("*")
                if path.is_file()
                and _eligible_path(path, corpus_root, config, extensions)
            )
            for path in candidates:
                if scanned >= max_files:
                    break
                scanned += 1
                try:
                    data = path.read_bytes()
                    if len(data) > max_bytes:
                        continue
                    text = data.decode("utf-8")
                    graph = parse_ast_graph(text, language)
                except (OSError, UnicodeError, ValueError):
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                constructs = _constructs(graph, text)
                if _split_for_digest(digest, constructs, config["split"]) != "train":
                    continue
                yield language, path, digest, graph


def build_ast_corpus(
    config: Mapping[str, object],
    *,
    root: Path,
) -> tuple[list[ASTCorpusRow], dict[str, object]]:
    registry = json.loads(
        (root / str(config["source_registry"])).read_text(encoding="utf-8")
    )
    source_meta = {
        str(row["id"]): row
        for row in registry["sources"]
    }
    corpus_root = root / str(config["root"])
    selection = config["selection"]
    rows = []
    rejected: dict[str, int] = {}
    seen_sources: set[str] = set()
    for language, sources in config["sources"].items():
        accepted = 0
        scanned = 0
        for source in sources:
            source_id = str(source["id"])
            if source_id not in source_meta:
                raise ValueError(f"unregistered AST source: {source_id}")
            metadata = source_meta[source_id]
            if metadata["usage"] != "training":
                raise ValueError(f"AST source is not training-approved: {source_id}")
            base = corpus_root / str(source["directory"])
            extensions = tuple(str(value) for value in source["extensions"])
            candidates = sorted(
                path
                for path in base.rglob("*")
                if path.is_file()
                and _eligible_path(path, corpus_root, config, extensions)
            )
            random.Random(
                int(selection.get("seed", 0))
                + sum(ord(char) for char in language)
            ).shuffle(candidates)
            for path in candidates:
                if accepted >= int(selection["rows_per_language"]):
                    break
                if scanned >= int(selection["max_files_scanned_per_language"]):
                    break
                scanned += 1
                try:
                    data = path.read_bytes()
                except OSError:
                    rejected["read_error"] = rejected.get("read_error", 0) + 1
                    continue
                if not (
                    int(selection["min_source_bytes"])
                    <= len(data)
                    <= int(selection["max_source_bytes"])
                ):
                    rejected["source_size"] = rejected.get("source_size", 0) + 1
                    continue
                try:
                    source = data.decode("utf-8")
                    graph = parse_ast_graph(source, language)
                except (UnicodeError, ValueError):
                    rejected["parse_or_utf8"] = rejected.get("parse_or_utf8", 0) + 1
                    continue
                if render_source(graph.root) != source:
                    rejected["lossy_render"] = rejected.get("lossy_render", 0) + 1
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen_sources:
                    rejected["exact_duplicate"] = rejected.get("exact_duplicate", 0) + 1
                    continue
                node_count, edge_count, depth = _tree_shape(graph.root)
                if not (
                    int(selection["min_ast_nodes"])
                    <= node_count
                    <= int(selection["max_ast_nodes"])
                    and depth <= int(selection["max_depth"])
                ):
                    rejected["ast_size"] = rejected.get("ast_size", 0) + 1
                    continue
                seen_sources.add(digest)
                constructs = _constructs(graph, source)
                rows.append(
                    ASTCorpusRow(
                        case_id=f"ast-{language}-{digest[:16]}",
                        language=language,
                        source_id=source_id,
                        source_path=str(path.relative_to(root)),
                        source_sha256=digest,
                        parser=parser_provenance(language),
                        license=str(metadata["license"]),
                        license_url=str(metadata["license_url"]),
                        repository=str(metadata["repository"]),
                        revision=str(metadata["revision"]),
                        source_url=str(metadata["source_url"]),
                        split=_split_for_digest(
                            digest,
                            constructs,
                            config["split"],
                        ),
                        node_count=node_count,
                        edge_count=edge_count,
                        max_depth=depth,
                        scope_count=len(graph.scopes),
                        symbol_count=len(graph.symbols),
                        constructs=constructs,
                        source=source,
                        ast_json=serialize_ast_graph(graph),
                    )
                )
                accepted += 1
    summary = {
        "rows": len(rows),
        "by_language": {
            language: sum(row.language == language for row in rows)
            for language in config["sources"]
        },
        "by_split": {
            split: sum(row.split == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "rejected": rejected,
    }
    return rows, summary


def write_ast_corpus(
    rows: list[ASTCorpusRow],
    *,
    output: Path,
    manifest: Path,
    summary: Mapping[str, object],
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row.as_dict(), ensure_ascii=False, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    report = {
        **summary,
        "output": str(output),
        "manifest": str(manifest),
        "corpus_sha256": digest,
        "source_integrity": all(
            row.source_sha256 and row.license and row.revision
            for row in rows
        ),
        "split_integrity": len(
            {row.source_sha256 for row in rows}
        ) == len(rows),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = [
    "ASTCorpusRow",
    "build_ast_corpus",
    "write_ast_corpus",
]
