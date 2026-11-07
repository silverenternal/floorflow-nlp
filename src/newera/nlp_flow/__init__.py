"""Natural-language cross-domain validation contracts."""

from newera.nlp_flow.components.platform.contracts.protocol.protocol import (
    HIERARCHY_LEVELS,
    LanguageFlowProtocol,
    hierarchy_from_payload,
    validate_protocol_payload,
)
from newera.nlp_flow.components.generation.outputs.contracts.structured_generation import (
    StructuredGenerationResult,
    TypedTreeGateway,
)

__all__ = [
    "HIERARCHY_LEVELS",
    "LanguageFlowProtocol",
    "StructuredGenerationResult",
    "TypedTreeGateway",
    "hierarchy_from_payload",
    "validate_protocol_payload",
]
