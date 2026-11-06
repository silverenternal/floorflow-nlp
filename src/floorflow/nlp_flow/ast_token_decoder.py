"""Causal token-endpoint decoder conditioned on generated AST states."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.nlp_flow.ast_cfm_data import ASTCodecVocabulary


class ASTTokenEndpointDecoder(nn.Module):
    """Serialize generated AST node states into syntax-ordered source tokens."""

    def __init__(
        self,
        vocabulary: ASTCodecVocabulary,
        hidden_size: int,
        *,
        max_tokens: int,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        for value, name in (
            (hidden_size, "hidden_size"),
            (max_tokens, "max_tokens"),
            (layers, "layers"),
            (heads, "heads"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.vocabulary = vocabulary
        self.max_tokens = int(max_tokens)
        self.token_embedding = nn.Embedding(len(vocabulary.source_tokens), hidden_size)
        self.position_embedding = nn.Embedding(max_tokens, hidden_size)
        self.phase_embedding = nn.Embedding(3, hidden_size)
        self.register_buffer(
            "lexeme_copy_matrix",
            self._copy_matrix(
                vocabulary.lexemes,
                "lex:",
            ),
            persistent=False,
        )
        self.register_buffer(
            "gap_copy_matrix",
            self._copy_matrix(
                vocabulary.gaps,
                "gap:",
            ),
            persistent=False,
        )
        layer = nn.TransformerDecoderLayer(
            hidden_size,
            heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.RMSNorm(hidden_size)
        self.output = nn.Linear(hidden_size, len(vocabulary.source_tokens))

    def _copy_matrix(
        self,
        values: tuple[str, ...],
        prefix: str,
    ) -> torch.Tensor:
        matrix = torch.zeros(
            len(values),
            len(self.vocabulary.source_tokens),
        )
        token_ids = {
            token: index
            for index, token in enumerate(self.vocabulary.source_tokens)
        }
        for row, value in enumerate(values):
            source_id = token_ids.get(f"{prefix}{value}")
            if source_id is not None:
                matrix[row, source_id] = 1.0
        return matrix

    def forward(
        self,
        node_states: torch.Tensor,
        node_mask: torch.Tensor,
        input_ids: torch.Tensor,
        source_token_node_ids: torch.Tensor | None = None,
        source_token_phase_ids: torch.Tensor | None = None,
        node_lexical_logits: torch.Tensor | None = None,
        node_gap_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_states.ndim != 3 or input_ids.ndim != 2:
            raise ValueError("node states and token ids must be batched")
        batch, length = input_ids.shape
        if node_states.shape[0] != batch or node_mask.shape != node_states.shape[:2]:
            raise ValueError("node states, node mask, and token batch must align")
        if length > self.max_tokens:
            raise ValueError("token endpoint exceeds decoder capacity")
        positions = torch.arange(length, device=input_ids.device)
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None]
        )
        if source_token_node_ids is not None or source_token_phase_ids is not None:
            if (
                source_token_node_ids is None
                or source_token_phase_ids is None
                or source_token_node_ids.shape != input_ids.shape
                or source_token_phase_ids.shape != input_ids.shape
            ):
                raise ValueError("source token node and phase ids must align")
            node_index = source_token_node_ids.to(input_ids.device).clamp(
                0,
                node_states.shape[1] - 1,
            )
            gathered = torch.gather(
                node_states,
                dim=1,
                index=node_index[..., None].expand(
                    -1,
                    -1,
                    node_states.shape[-1],
                ),
            )
            phase = source_token_phase_ids.to(input_ids.device).clamp(0, 2)
            hidden = hidden + gathered + self.phase_embedding(phase)
        causal = torch.ones(
            length,
            length,
            dtype=torch.bool,
            device=input_ids.device,
        ).tril()
        decoded = self.decoder(
            hidden,
            node_states,
            tgt_mask=causal,
            tgt_key_padding_mask=input_ids.eq(0),
            memory_key_padding_mask=~node_mask,
        )
        logits = self.output(self.norm(decoded))
        if (
            source_token_node_ids is not None
            and source_token_phase_ids is not None
            and (node_lexical_logits is not None or node_gap_logits is not None)
        ):
            node_index = source_token_node_ids.to(input_ids.device).clamp(
                0,
                node_states.shape[1] - 1,
            )
            phase = source_token_phase_ids.to(input_ids.device).clamp(0, 2)
            bias = torch.zeros_like(logits)
            if node_lexical_logits is not None:
                gathered_lexical = torch.gather(
                    node_lexical_logits,
                    dim=1,
                    index=node_index[..., None].expand(
                        -1,
                        -1,
                        node_lexical_logits.shape[-1],
                    ),
                )
                lexical_bias = torch.matmul(
                    gathered_lexical,
                    self.lexeme_copy_matrix,
                )
                bias = bias + lexical_bias * phase.eq(2)[..., None]
            if node_gap_logits is not None:
                gathered_gap = torch.gather(
                    node_gap_logits,
                    dim=1,
                    index=node_index[..., None].expand(
                        -1,
                        -1,
                        node_gap_logits.shape[-1],
                    ),
                )
                gap_bias = torch.matmul(
                    gathered_gap,
                    self.gap_copy_matrix,
                )
                bias = bias + gap_bias * phase.eq(1)[..., None]
            logits = logits + bias
        return logits

    @torch.no_grad()
    def sample(
        self,
        node_states: torch.Tensor,
        node_mask: torch.Tensor,
        *,
        source_token_node_ids: torch.Tensor | None = None,
        source_token_phase_ids: torch.Tensor | None = None,
        node_lexical_logits: torch.Tensor | None = None,
        node_gap_logits: torch.Tensor | None = None,
        minimum_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bos = self.vocabulary.source_tokens.index("<bos>")
        eos = self.vocabulary.source_tokens.index("<eos>")
        pad = self.vocabulary.source_tokens.index("<pad>")
        generated = torch.full(
            (node_states.shape[0], 1),
            bos,
            dtype=torch.long,
            device=node_states.device,
        )
        finished = torch.zeros(node_states.shape[0], dtype=torch.bool, device=node_states.device)
        lexical_ids = torch.tensor(
            [
                index
                for index, token in enumerate(self.vocabulary.source_tokens)
                if token.startswith("lex:")
            ],
            dtype=torch.long,
            device=node_states.device,
        )
        content_ids = torch.tensor(
            [
                index
                for index, token in enumerate(self.vocabulary.source_tokens)
                if token.startswith("lex:")
                and token.removeprefix("lex:").strip()
            ],
            dtype=torch.long,
            device=node_states.device,
        )
        gap_ids = torch.tensor(
            [
                index
                for index, token in enumerate(self.vocabulary.source_tokens)
                if token.startswith("gap:")
            ],
            dtype=torch.long,
            device=node_states.device,
        )
        has_lexical = torch.zeros(
            node_states.shape[0],
            dtype=torch.bool,
            device=node_states.device,
        )
        expect_gap = torch.ones_like(has_lexical)
        minimum = (
            torch.full(
                (node_states.shape[0],),
                3,
                dtype=torch.long,
                device=node_states.device,
            )
            if minimum_tokens is None
            else minimum_tokens.to(node_states.device, torch.long)
        )
        if minimum.shape != (node_states.shape[0],):
            raise ValueError("minimum token budget must be [batch]")
        for _step in range(1, self.max_tokens):
            logits = self(
                node_states,
                node_mask,
                generated,
                (
                    None
                    if source_token_node_ids is None
                    else source_token_node_ids[:, : generated.shape[1]]
                ),
                (
                    None
                    if source_token_phase_ids is None
                    else source_token_phase_ids[:, : generated.shape[1]]
                ),
                node_lexical_logits,
                node_gap_logits,
            )[:, -1]
            token_ids = torch.arange(logits.shape[-1], device=logits.device)
            gap_allowed = torch.isin(token_ids, gap_ids)[None].expand_as(logits)
            lexical_allowed = torch.isin(
                token_ids,
                lexical_ids,
            )[None].expand_as(logits)
            content_allowed = torch.isin(
                token_ids,
                content_ids,
            )[None].expand_as(logits)
            eos_allowed = token_ids.eq(eos)[None].expand_as(logits)
            allowed = (
                (expect_gap[:, None] & gap_allowed)
                | (
                    (~expect_gap)[:, None]
                    & (
                        (
                            has_lexical[:, None]
                            & (lexical_allowed | eos_allowed)
                        )
                        | (~has_lexical[:, None] & content_allowed)
                    )
                )
            )
            length_ready = generated.shape[1] + 1 >= minimum
            allowed = allowed & (length_ready[:, None] | ~eos_allowed)
            logits = logits.masked_fill(~allowed, float("-inf"))
            next_id = logits.argmax(dim=-1)
            next_id = torch.where(finished, torch.full_like(next_id, pad), next_id)
            generated = torch.cat((generated, next_id[:, None]), dim=1)
            selected_lexical = next_id[:, None].eq(lexical_ids).any(dim=-1)
            has_lexical = has_lexical | next_id[:, None].eq(content_ids).any(dim=-1)
            expect_gap = torch.where(
                finished,
                expect_gap,
                selected_lexical,
            )
            finished = finished | next_id.eq(eos)
            if finished.all():
                break
        if generated.shape[1] < self.max_tokens:
            generated = torch.nn.functional.pad(
                generated,
                (0, self.max_tokens - generated.shape[1]),
                value=pad,
            )
        return generated


def decode_source_tokens(ids: torch.Tensor, vocabulary: ASTCodecVocabulary) -> str:
    if ids.ndim != 1:
        raise ValueError("source token ids must be one-dimensional")
    parts = []
    for token_id in ids.tolist():
        token = vocabulary.source_tokens[int(token_id)]
        if token == "<eos>":
            break
        if token in {"<pad>", "<bos>"}:
            continue
        kind, value = token.split(":", 1)
        if kind not in {"gap", "lex"}:
            raise ValueError(f"unknown source token kind: {kind}")
        parts.append(value)
    source = "".join(parts)
    if not source.strip():
        raise ValueError("generated token endpoint is empty")
    return source


def source_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("source token logits and targets do not align")
    active = (
        targets.ne(0)
        if mask is None
        else mask.to(logits.device, torch.bool)
    )
    if active.shape != targets.shape or not active.any():
        raise ValueError("source token supervision has no active targets")
    return torch.nn.functional.cross_entropy(logits[active], targets[active])


__all__ = [
    "ASTTokenEndpointDecoder",
    "decode_source_tokens",
    "source_token_loss",
]
