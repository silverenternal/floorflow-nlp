"""Local multi-turn chat generation for compact NLP checkpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

from floorflow.nlp_flow.instruction_data import encode_chat_prompt
from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig


def model_config_from_experiment(payload: dict) -> NLPModelConfig:
    model = payload["model"]
    candidate = payload["arm"] == "candidate"
    return NLPModelConfig(
        vocab_size=int(model["vocab_size"]),
        hidden_size=int(model.get("hidden_size", 160)),
        num_layers=int(model.get("num_layers", 3)),
        num_heads=int(model.get("num_heads", 4)),
        num_kv_heads=int(model.get("num_kv_heads", 2)),
        intermediate_size=int(model.get("intermediate_size", 400)),
        max_seq_len=int(model["max_context_tokens"]),
        channel_size=int(model.get("channel_size", 32)),
        parameter_cap=min(23_000_000, int(int(model["target_parameters"]) * 1.05)),
        enable_typed_channels=candidate,
        enable_typed_fusion=candidate,
        enable_two_stage_cfm=candidate,
        enable_json_action_head=candidate,
        enable_expert_factory=candidate,
        enable_emergence_objective=bool(
            model.get("enable_emergence_objective", False)
        ),
        emergence_basis_count=int(model.get("emergence_basis_count", 32)),
        emergence_active_basis=int(model.get("emergence_active_basis", 4)),
        emergence_reward_weight=float(
            model.get("emergence_reward_weight", 0.1)
        ),
        emergence_norm_envelope_ratio=float(
            model.get("emergence_norm_envelope_ratio", 1.5)
        ),
        enable_dynamic_stack=bool(model.get("enable_dynamic_stack", False)),
        dynamic_max_steps=int(model.get("dynamic_max_steps", 4)),
        dynamic_min_steps=int(model.get("dynamic_min_steps", 1)),
        dynamic_parallel_lanes=int(model.get("dynamic_parallel_lanes", 2)),
        dynamic_parallel_top_k=int(model.get("dynamic_parallel_top_k", 1)),
        dynamic_lane_rank=int(model.get("dynamic_lane_rank", 8)),
        enable_gap_ensemble=bool(model.get("enable_gap_ensemble", False)),
        gap_adapter_rank=int(model.get("gap_adapter_rank", 8)),
        gap_fusion_dim=int(model.get("gap_fusion_dim", 64)),
        gap_heads=int(model.get("gap_heads", 4)),
        gap_router_enabled=bool(model.get("gap_router_enabled", True)),
        gap_names=tuple(
            model.get(
                "gap_names",
                (
                    "json_validity",
                    "ast_structure",
                    "python_execution",
                    "rust_execution",
                    "cpp_execution",
                    "engineering_acceleration",
                    "model_architecture",
                    "language_retention",
                    "compositional_emergence",
                ),
            )
        ),
        enable_benchmark_proxy=bool(
            model.get("enable_benchmark_proxy", False)
        ),
        proxy_test_output_classes=int(
            model.get("proxy_test_output_classes", 64)
        ),
        proxy_max_constraints=int(
            model.get("proxy_max_constraints", 16)
        ),
        proxy_retrieval_dim=int(model.get("proxy_retrieval_dim", 32)),
        enable_turn_embeddings=bool(
            model.get("enable_turn_embeddings", False)
        ),
        turn_type_count=int(model.get("turn_type_count", 5)),
        enable_conversation_control=bool(
            model.get("enable_conversation_control", False)
        ),
        use_learned_stop_at_inference=bool(
            model.get("use_learned_stop_at_inference", False)
        ),
        enable_expert_bank=bool(model.get("enable_expert_bank", False)),
        expert_bank_mode=str(model.get("expert_bank_mode", "generated")),
        expert_count=int(model.get("expert_count", 4)),
        expert_top_k=int(model.get("expert_top_k", 2)),
        enable_neuron_module=bool(
            model.get("enable_neuron_module", False)
        ),
        neuron_local_size=int(model.get("neuron_local_size", 32)),
        enable_unified_competence=bool(
            model.get("enable_unified_competence", False)
        ),
        unified_cell_rank=int(model.get("unified_cell_rank", 8)),
        unified_top_k=int(model.get("unified_top_k", 3)),
        unified_planner_dim=int(model.get("unified_planner_dim", 64)),
        unified_generated_rank=int(model.get("unified_generated_rank", 16)),
        unified_generated_basis_count=int(
            model.get("unified_generated_basis_count", 4)
        ),
        unified_generated_expert_mode=str(
            model.get("unified_generated_expert_mode", "online")
        ),
        enable_runtime_competence=bool(
            model.get("enable_runtime_competence", False)
        ),
        runtime_competence_capacity=int(
            model.get("runtime_competence_capacity", 8)
        ),
        runtime_trigger_threshold=float(
            model.get("runtime_trigger_threshold", 0.5)
        ),
        runtime_promotion_trials=int(
            model.get("runtime_promotion_trials", 2)
        ),
        runtime_promotion_success_rate=float(
            model.get("runtime_promotion_success_rate", 0.75)
        ),
    )


def render_conversation(
    turns: list[tuple[str, str]],
    user_message: str,
) -> str:
    parts = []
    for user, assistant in turns:
        parts.append(f"User: {user.strip()}\nAssistant: {assistant.strip()}\n")
    parts.append(f"User: {user_message.strip()}\nAssistant:")
    return "".join(parts)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    stop_reason: str = "max_tokens"


@torch.no_grad()
def generate_reply(
    model: CompactNLPDecoder,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
    top_k: int | None = 40,
    repetition_penalty: float = 1.0,
    seed: int = 19,
    device: torch.device | str = "cpu",
) -> GenerationResult:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0.0:
        raise ValueError("temperature cannot be negative")
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise ValueError("tokenizer needs <bos> and <eos>")
    context_limit = model.config.max_seq_len - 1
    prompt_ids, prompt_turn_ids = encode_chat_prompt(
        prompt,
        tokenizer,
        max_tokens=context_limit,
    )
    sequence = [bos, *prompt_ids]
    turn_ids = [0, *prompt_turn_ids]
    prompt_tokens = len(sequence)
    generated: list[int] = []
    stop_reason = "max_tokens"
    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(seed)
    for _step in range(max_new_tokens):
        if len(sequence) >= context_limit:
            sequence = [bos, *sequence[-(context_limit - 1):]]
            turn_ids = [0, *turn_ids[-(context_limit - 1):]]
        input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
        model_kwargs = {}
        if model.turn_embedding is not None:
            model_kwargs["turn_ids"] = torch.tensor(
                [turn_ids],
                dtype=torch.long,
                device=device,
            )
        output = model(input_ids, **model_kwargs)
        logits = output.token_logits[0, -1].float()
        if (
            model.config.use_learned_stop_at_inference
            and output.conversation_logits is not None
            and generated
            and bool(
                output.conversation_logits["continue_logits"][0, -1] <= 0.0
            )
        ):
            stop_reason = "learned_stop"
            break
        for token in set(generated[-32:]):
            logits[token] = (
                logits[token] / repetition_penalty
                if logits[token] > 0
                else logits[token] * repetition_penalty
            )
        if temperature == 0.0:
            token = int(logits.argmax())
        else:
            scaled = logits / temperature
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(scaled, min(top_k, scaled.numel()))
                probabilities = torch.softmax(values, dim=-1)
                selected = int(
                    torch.multinomial(
                        probabilities,
                        1,
                        generator=generator,
                    )
                )
                token = int(indices[selected])
            else:
                token = int(
                    torch.multinomial(
                        torch.softmax(scaled, dim=-1),
                        1,
                        generator=generator,
                    )
                )
        if token == eos:
            stop_reason = "eos"
            break
        generated.append(token)
        sequence.append(token)
        turn_ids.append(1)
        partial = tokenizer.decode(generated)
        if "\nUser:" in partial:
            generated_text = partial.split("\nUser:", 1)[0].rstrip()
            stop_reason = "role_boundary"
            break
    if "generated_text" not in locals():
        generated_text = tokenizer.decode(generated).strip()
    elapsed = time.perf_counter() - started
    return GenerationResult(
        text=generated_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(generated),
        elapsed_seconds=elapsed,
        stop_reason=stop_reason,
    )


__all__ = [
    "GenerationResult",
    "generate_reply",
    "model_config_from_experiment",
    "render_conversation",
]
