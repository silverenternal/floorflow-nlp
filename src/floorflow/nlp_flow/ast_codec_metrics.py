"""Structural and lexical metrics for canonical AST graphs."""

from __future__ import annotations

from collections import Counter

from floorflow.nlp_flow.ast_codec import ASTGraph, ASTNode, _walk, is_parse_valid


def _node_signatures(node: ASTNode, path: tuple[int, ...] = ()) -> Counter:
    signature = (path, node.node_type, node.named)
    counts = Counter({signature: 1})
    for index, (_field, child) in enumerate(node.children):
        counts.update(_node_signatures(child, path + (index,)))
    return counts


def _lexical_signatures(node: ASTNode, path: tuple[int, ...] = ()) -> Counter:
    counts = (
        Counter({(path, node.lexical_value): 1})
        if node.lexical_value is not None
        else Counter()
    )
    for index, (_field, child) in enumerate(node.children):
        counts.update(_lexical_signatures(child, path + (index,)))
    return counts


def _edge_signatures(node: ASTNode) -> Counter:
    counts = Counter()
    for field, child in node.children:
        counts[(node.node_type, field, child.node_type)] += 1
        counts.update(_edge_signatures(child))
    return counts


def _scope_signatures(node: ASTNode) -> Counter:
    nodes_by_scope: dict[int, ASTNode] = {}
    for _path, current in _walk(node):
        nodes_by_scope.setdefault(current.scope_id, current)
    counts = Counter()
    for _path, current in _walk(node):
        if current.symbol_name is None:
            continue
        owner = nodes_by_scope[current.scope_id]
        counts[(owner.node_type, current.symbol_kind, current.symbol_name)] += 1
    return counts


def _counter_f1(left: Counter, right: Counter) -> float:
    true_positive = sum((left & right).values())
    precision = true_positive / max(1, sum(left.values()))
    recall = true_positive / max(1, sum(right.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tree_records(node: ASTNode) -> tuple[list[tuple[str, ...]], list[int]]:
    labels: list[tuple[str, ...]] = []
    leftmost: list[int] = []

    def visit(current: ASTNode) -> int:
        child_indices = [visit(child) for _field, child in current.children]
        index = len(labels) + 1
        labels.append(
            (
                current.node_type,
                current.lexical_value or "",
                current.symbol_kind or "",
                current.symbol_name or "",
            )
        )
        leftmost.append(leftmost[child_indices[0] - 1] if child_indices else index)
        return index

    visit(node)
    return labels, leftmost


def tree_edit_distance(left: ASTNode | ASTGraph, right: ASTNode | ASTGraph) -> int:
    """Zhang-Shasha ordered tree edit distance with equal unit edit costs."""

    left_root = left.root if isinstance(left, ASTGraph) else left
    right_root = right.root if isinstance(right, ASTGraph) else right
    left_labels, left_lmd = _tree_records(left_root)
    right_labels, right_lmd = _tree_records(right_root)
    n, m = len(left_labels), len(right_labels)
    forest = [[0] * (m + 1) for _ in range(n + 1)]
    tree = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        forest[i][0] = forest[i - 1][0] + 1
        for j in range(1, m + 1):
            forest[0][j] = forest[0][j - 1] + 1
            for di in range(left_lmd[i - 1], i + 1):
                for dj in range(right_lmd[j - 1], j + 1):
                    if (
                        left_lmd[di - 1] == left_lmd[i - 1]
                        and right_lmd[dj - 1] == right_lmd[j - 1]
                    ):
                        rename = 0 if left_labels[di - 1] == right_labels[dj - 1] else 1
                        forest[di][dj] = min(
                            forest[di - 1][dj] + 1,
                            forest[di][dj - 1] + 1,
                            forest[di - 1][dj - 1] + rename,
                        )
                        tree[di][dj] = forest[di][dj]
                    else:
                        forest[di][dj] = min(
                            forest[di - 1][dj] + 1,
                            forest[di][dj - 1] + 1,
                            forest[left_lmd[di - 1] - 1][right_lmd[dj - 1] - 1]
                            + tree[di][dj],
                        )
    return int(tree[n][m])


def normalized_tree_edit_distance(
    left: ASTNode | ASTGraph,
    right: ASTNode | ASTGraph,
) -> float:
    left_root = left.root if isinstance(left, ASTGraph) else left
    right_root = right.root if isinstance(right, ASTGraph) else right
    left_size = sum(1 for _ in _walk(left_root))
    right_size = sum(1 for _ in _walk(right_root))
    if left_size == 0 and right_size == 0:
        return 0.0
    return tree_edit_distance(left, right) / max(left_size, right_size)


def ast_similarity(
    predicted: ASTNode | ASTGraph,
    reference: ASTNode | ASTGraph,
) -> dict[str, float]:
    """Return the frozen structural metrics used by the Stage-A gate."""

    predicted_root = predicted.root if isinstance(predicted, ASTGraph) else predicted
    reference_root = reference.root if isinstance(reference, ASTGraph) else reference
    scope_relation_f1 = _counter_f1(
        _scope_signatures(predicted_root),
        _scope_signatures(reference_root),
    )
    return {
        "node_type_f1": _counter_f1(
            _node_signatures(predicted_root),
            _node_signatures(reference_root),
        ),
        "typed_edge_f1": _counter_f1(
            _edge_signatures(predicted_root),
            _edge_signatures(reference_root),
        ),
        "scope_relation_f1": scope_relation_f1,
        "scope_accuracy": scope_relation_f1,
        "parse_validity": float(is_parse_valid(predicted)),
        "normalized_tree_edit_distance": normalized_tree_edit_distance(
            predicted,
            reference,
        ),
    }


def ast_metrics(
    predicted: ASTNode | ASTGraph,
    reference: ASTNode | ASTGraph,
) -> dict[str, float | bool]:
    """Return all frozen structural metrics without any semantic-quality claim."""

    predicted_root = predicted.root if isinstance(predicted, ASTGraph) else predicted
    reference_root = reference.root if isinstance(reference, ASTGraph) else reference
    metrics = ast_similarity(predicted, reference)
    metrics.update(
        {
            "lexical_f1": _counter_f1(
                _lexical_signatures(predicted_root),
                _lexical_signatures(reference_root),
            ),
            "parse_validity": bool(is_parse_valid(predicted)),
        }
    )
    return metrics


__all__ = [
    "ast_metrics",
    "ast_similarity",
    "normalized_tree_edit_distance",
    "tree_edit_distance",
]
