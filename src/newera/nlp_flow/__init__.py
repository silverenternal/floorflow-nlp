"""Natural-language cross-domain validation contracts.

Public contracts are loaded on access so importing an unrelated ``nlp_flow``
submodule does not initialize parsers and optional generation backends.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "HIERARCHY_LEVELS": "newera.nlp_flow.components.platform.contracts.protocol.protocol",
    "LanguageFlowProtocol": "newera.nlp_flow.components.platform.contracts.protocol.protocol",
    "hierarchy_from_payload": "newera.nlp_flow.components.platform.contracts.protocol.protocol",
    "validate_protocol_payload": "newera.nlp_flow.components.platform.contracts.protocol.protocol",
    "StructuredGenerationResult": "newera.nlp_flow.components.generation.outputs.contracts.structured_generation",
    "TypedTreeGateway": "newera.nlp_flow.components.generation.outputs.contracts.structured_generation",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))

__all__ = [
    "HIERARCHY_LEVELS",
    "LanguageFlowProtocol",
    "StructuredGenerationResult",
    "TypedTreeGateway",
    "hierarchy_from_payload",
    "validate_protocol_payload",
]
