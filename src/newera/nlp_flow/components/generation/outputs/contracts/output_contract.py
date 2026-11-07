"""Rust-generated, domain-specific output validity requirements."""

from __future__ import annotations

from dataclasses import dataclass

from newera.nlp_flow.components.platform.contracts.protocol._generated_protocol import OUTPUT_CONTRACT_ROWS
from newera.nlp_flow.components.generation.trees.typed.typed_tree import TypedTreeValidity


@dataclass(frozen=True)
class OutputContract:
    profile: str
    structural: bool
    schema: bool
    executable: bool

    @property
    def required_dimensions(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in ("structural", "schema", "executable")
            if getattr(self, name)
        )


OUTPUT_CONTRACTS = {
    profile: OutputContract(profile, structural, schema, executable)
    for profile, structural, schema, executable in OUTPUT_CONTRACT_ROWS
}


@dataclass(frozen=True)
class OutputContractDecision:
    contract: OutputContract
    valid: bool
    missing: tuple[str, ...]

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError(
                f"{self.contract.profile} output is missing validity: "
                f"{list(self.missing)}"
            )


def get_output_contract(profile: str) -> OutputContract:
    try:
        return OUTPUT_CONTRACTS[profile]
    except KeyError as error:
        raise ValueError(f"unknown output contract: {profile}") from error


def validate_output_contract(
    profile: str,
    validity: TypedTreeValidity,
) -> OutputContractDecision:
    contract = get_output_contract(profile)
    missing = tuple(
        dimension
        for dimension in contract.required_dimensions
        if not getattr(validity, dimension)
    )
    return OutputContractDecision(
        contract=contract,
        valid=not missing,
        missing=missing,
    )


__all__ = [
    "OUTPUT_CONTRACTS",
    "OutputContract",
    "OutputContractDecision",
    "get_output_contract",
    "validate_output_contract",
]
