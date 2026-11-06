"""Strict upstream-quality gate for generated AST source endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from floorflow.nlp_flow.ast_codec import ASTNode, ast_metrics, parse_ast


@dataclass(frozen=True)
class ASTUpstreamThresholds:
    minimum_node_type_f1: float = 0.95
    minimum_typed_edge_f1: float = 0.95
    minimum_scope_relation_f1: float = 0.90
    minimum_lexical_f1: float = 0.95
    maximum_normalized_tree_edit_distance: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "minimum_node_type_f1",
            "minimum_typed_edge_f1",
            "minimum_scope_relation_f1",
            "minimum_lexical_f1",
            "maximum_normalized_tree_edit_distance",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class ASTHandoffEvidence:
    source_kind: str
    status: Literal["pass", "fail"]
    parse_validity: bool
    checks: Mapping[str, bool]
    metrics: Mapping[str, float | bool] | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail"}:
            raise ValueError("AST handoff status must be pass or fail")
        if self.status == "pass" and (
            not self.parse_validity or not all(self.checks.values())
        ):
            raise ValueError("passing AST evidence requires all gate checks")

    @property
    def ready(self) -> bool:
        return (
            self.source_kind in {"token_endpoint", "runtime_structural"}
            and self.status == "pass"
            and self.parse_validity
            and bool(self.checks)
            and all(self.checks.values())
        )

    @property
    def training_ready(self) -> bool:
        return self.ready and self.source_kind == "token_endpoint"

    @classmethod
    def runtime_structural(
        cls,
        *,
        parse_validity: bool,
        single_root: bool,
        nonempty: bool,
        within_node_budget: bool,
        confidence: float,
        minimum_confidence: float | None = None,
    ) -> ASTHandoffEvidence:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("runtime AST confidence must be in [0, 1]")
        if minimum_confidence is not None and not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum AST confidence must be in [0, 1]")
        checks = {
            "parse_validity": parse_validity,
            "single_root": single_root,
            "nonempty": nonempty,
            "within_node_budget": within_node_budget,
            "confidence": (
                confidence >= minimum_confidence
                if minimum_confidence is not None
                else True
            ),
        }
        return cls(
            source_kind="runtime_structural",
            status="pass" if all(checks.values()) else "fail",
            parse_validity=parse_validity,
            checks=checks,
            metrics={"confidence": confidence},
            failure=(
                None
                if all(checks.values())
                else "runtime AST structural gate failed"
            ),
        )

    @classmethod
    def from_report(
        cls,
        report: Mapping[str, object],
    ) -> ASTHandoffEvidence:
        checks = report.get("checks", {})
        metrics = report.get("metrics")
        if not isinstance(checks, Mapping) or not all(
            isinstance(name, str) and isinstance(value, bool)
            for name, value in checks.items()
        ):
            raise TypeError("AST gate checks must be a boolean mapping")
        if metrics is not None and not isinstance(metrics, Mapping):
            raise TypeError("AST gate metrics must be a mapping or null")
        return cls(
            source_kind=str(report["source_kind"]),
            status=str(report["status"]),
            parse_validity=bool(report["parse_validity"]),
            checks=dict(checks),
            metrics=(
                None
                if metrics is None
                else {
                    str(name): (
                        value
                        if isinstance(value, bool)
                        else float(value)
                    )
                    for name, value in metrics.items()
                }
            ),
            failure=(
                None
                if report.get("failure") is None
                else str(report["failure"])
            ),
        )

    def require_stage_b_ready(
        self,
        *,
        require_reference_metrics: bool = False,
    ) -> None:
        ready = (
            self.training_ready
            if require_reference_metrics
            else self.ready
        )
        if not ready:
            required = (
                "verified token-endpoint AST evidence"
                if require_reference_metrics
                else "verified AST evidence"
            )
            raise ValueError(
                f"Stage B requires {required}: "
                f"status={self.status}, source={self.source_kind}, "
                f"failure={self.failure}"
            )


def evaluate_ast_upstream_source(
    source: str,
    reference: ASTNode,
    language: Literal["python", "rust", "cpp"],
    *,
    source_kind: str,
    thresholds: ASTUpstreamThresholds | None = None,
) -> dict[str, object]:
    """Reject grammar-template retrieval as a substitute for token AST synthesis."""

    active = thresholds or ASTUpstreamThresholds()
    if source_kind != "token_endpoint":
        return {
            "status": "fail",
            "parse_validity": False,
            "source_kind": source_kind,
            "failure": "grammar templates are not accepted as upstream token AST synthesis",
            "metrics": None,
        }
    try:
        generated = parse_ast(source, language)
    except (TypeError, ValueError, UnicodeError) as error:
        return {
            "status": "fail",
            "parse_validity": False,
            "source_kind": source_kind,
            "failure": str(error),
            "metrics": None,
        }
    metrics = ast_metrics(generated, reference)
    checks = {
        "parse_validity": metrics["parse_validity"],
        "node_type_f1": (
            metrics["node_type_f1"] >= active.minimum_node_type_f1
        ),
        "typed_edge_f1": (
            metrics["typed_edge_f1"] >= active.minimum_typed_edge_f1
        ),
        "scope_relation_f1": (
            metrics["scope_relation_f1"]
            >= active.minimum_scope_relation_f1
        ),
        "lexical_f1": metrics["lexical_f1"] >= active.minimum_lexical_f1,
        "tree_edit_distance": (
            metrics["normalized_tree_edit_distance"]
            <= active.maximum_normalized_tree_edit_distance
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "parse_validity": metrics["parse_validity"],
        "source_kind": source_kind,
        "checks": checks,
        "metrics": metrics,
        "failure": None if all(checks.values()) else "quality threshold failed",
    }


__all__ = [
    "ASTHandoffEvidence",
    "ASTUpstreamThresholds",
    "evaluate_ast_upstream_source",
]
