import json
from pathlib import Path

import pytest

from floorflow.nlp_flow.ast_codec import (
    deserialize_ast_graph,
    parse_ast_graph,
    parser_fingerprint,
    render_source,
    serialize_ast_graph,
    validate_ast,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/nlp_canonical_ir_v11.json").read_text())
SAMPLES = {
    "python": '# π\n@decorator(value="λ")\ndef f(x: str) -> str:\n    # keep this gap\n    return x  \n',
    "rust": '// λ\n#[inline]\nfn f(x: &str) -> &str { /* gap */ x }\n',
    "cpp": '#define SCALE(x) ((x) * 2)\n// λ\nconst char* f() { return R"tag(raw λ)tag"; } // tail\n',
}


@pytest.mark.parametrize("language", ("python", "rust", "cpp"))
def test_pinned_multilingual_canonical_ir_replays_exact_bytes(language: str) -> None:
    source = SAMPLES[language]
    graph = parse_ast_graph(source, language)
    restored = deserialize_ast_graph(serialize_ast_graph(graph))
    binding = CONFIG["parser_packages"][language]
    runtime = parser_fingerprint(language)
    assert graph == restored
    assert render_source(restored.root) == source
    assert validate_ast(restored, source=source)
    assert graph.root.node_type == binding["root"]
    assert runtime.package.replace("_", "-") == binding["package"]
    assert runtime.package_version == binding["version"]
    assert runtime.abi_version == binding["abi_version"]
    assert runtime.grammar_sha256 == binding["grammar_sha256"]
    assert runtime.native_sha256 == binding["native_sha256"]


def test_canonical_config_declares_nonsemantic_and_unsupported_boundaries() -> None:
    for language in ("python", "rust", "cpp"):
        assert CONFIG["language_boundaries"][language]["does_not_claim"]
    assert "macro expansion" in CONFIG["language_boundaries"]["rust"]["does_not_claim"]
    assert CONFIG["unsupported_policy"]["parse_error"].startswith("Reject the whole row")
    assert CONFIG["graph_fields"]


@pytest.mark.parametrize("language", ("python", "rust", "cpp"))
def test_malformed_source_is_visible_as_rejection(language: str) -> None:
    malformed = {"python": "def f(:\n", "rust": "fn f( {", "cpp": "int f( {"}[language]
    with pytest.raises(ValueError, match="parse error"):
        parse_ast_graph(malformed, language)
