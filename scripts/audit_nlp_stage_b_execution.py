#!/usr/bin/env python3
"""Train generated-AST Stage B on a bounded executable program-selection probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from floorflow.nlp_flow.ast_cfm import (
    ASTCodecVocabulary,
    ASTStageAFlow,
    ast_cfm_objective_losses,
    collate_ast_batches,
    encode_ast_graph,
)
from floorflow.nlp_flow.ast_codec import parse_ast
from floorflow.nlp_flow.ast_generation import predict_generated_topology_batch
from floorflow.nlp_flow.ast_token_decoder import decode_source_tokens
from floorflow.nlp_flow.cfm_training import CFMTrainingConfig, sample_flow_times
from floorflow.nlp_flow.model.prompt_encoding import SemanticPromptEncoder
from floorflow.nlp_flow.result_cfm import GeneratedASTResultCFM, ResultCFMConfig
from floorflow.nlp_flow.stage_b_execution import (
    PROGRAM_NAMES,
    NoResultFlowControl,
    ResultExecutionHead,
    execute_program,
)

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / ".data/nlp/tokenizer_v1/tokenizer.json"


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return "cpu"


def _program_score(predicted_ids: torch.Tensor) -> float:
    inputs = ((2, 3), (7, 5), (11, 4), (9, 2), (6, 10), (13, 7))
    correct = 0
    total = 0
    for program_id in range(len(PROGRAM_NAMES)):
        predicted = int(predicted_ids[program_id])
        for left, right in inputs:
            correct += int(
                execute_program(predicted, left, right)
                == execute_program(program_id, left, right)
            )
            total += 1
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--ast-steps", type=int, default=1500)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_result_cfm_execution_probe_v1.json",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")
    torch.manual_seed(311)
    sources = (
        "def add(a, b):\n    return a + b\n",
        "def subtract(a, b):\n    return a - b\n",
        "def multiply(a, b):\n    return a * b\n",
        "def maximum(a, b):\n    return max(a, b)\n",
    )
    graphs = [parse_ast(source, "python") for source in sources]
    vocabulary = ASTCodecVocabulary.from_graphs(graphs)
    templates = {source: index for index, source in enumerate(vocabulary.templates)}
    batches = [
        encode_ast_graph(
            graph,
            vocabulary,
            max_nodes=32,
            template_id=templates[source],
        )
        for graph, source in zip(graphs, sources, strict=True)
    ]
    target_batch = collate_ast_batches(batches)
    stage_a = ASTStageAFlow(
        vocabulary,
        hidden_size=24,
        num_heads=4,
        max_nodes=32,
    ).to(args.device).eval()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    prompts = [f"implement {operation} without branches" for operation in PROGRAM_NAMES]
    prompt_ids = [tokenizer.encode(prompt).ids[:64] for prompt in prompts]
    max_prompt = max(len(ids) for ids in prompt_ids)
    input_ids = torch.zeros(4, max_prompt, dtype=torch.long)
    prompt_mask = torch.zeros(4, max_prompt, dtype=torch.bool)
    for row, ids in enumerate(prompt_ids):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        prompt_mask[row, : len(ids)] = True
    input_ids = input_ids.to(args.device)
    prompt_mask = prompt_mask.to(args.device)
    prompt_encoder = SemanticPromptEncoder(
        tokenizer.get_vocab_size(),
        24,
        max_tokens=64,
        layers=1,
        heads=4,
        language_count=3,
    ).to(args.device)
    ast_optimizer = torch.optim.AdamW(
        [*stage_a.parameters(), *prompt_encoder.parameters()],
        lr=1.5e-3,
    )
    ast_sampler = CFMTrainingConfig(timestep_sampler="logit_normal")
    for _step in range(args.ast_steps):
        stage_a.train()
        prompt_encoder.train()
        ast_optimizer.zero_grad(set_to_none=True)
        contexts = prompt_encoder(
            input_ids,
            prompt_mask,
            torch.zeros(4, dtype=torch.long, device=args.device),
        ).context
        target_state = stage_a.embed_target(target_batch, args.device)
        noise = torch.randn_like(target_state)
        time = sample_flow_times(4, ast_sampler, device=args.device)
        state = (
            (1.0 - time[:, None, None]) * noise
            + time[:, None, None] * target_state
        )
        output = stage_a(
            state,
            time,
            contexts,
            target_batch,
            source_token_inputs=target_batch.source_token_ids[:, :-1],
            source_token_node_ids=target_batch.source_token_node_ids[:, :-1],
            source_token_phase_ids=target_batch.source_token_phase_ids[:, :-1],
        )
        categorical_output = stage_a(
            noise,
            torch.zeros_like(time),
            contexts,
            target_batch,
            source_token_inputs=target_batch.source_token_ids[:, :-1],
            source_token_node_ids=target_batch.source_token_node_ids[:, :-1],
            source_token_phase_ids=target_batch.source_token_phase_ids[:, :-1],
            source_token_teacher_state=target_state,
        )
        losses = ast_cfm_objective_losses(
            output,
            target_state,
            noise,
            target_batch,
            categorical_output=categorical_output,
        )
        losses.total.backward()
        ast_optimizer.step()
    stage_a.eval()
    prompt_encoder.eval()
    with torch.no_grad():
        contexts = prompt_encoder(
            input_ids,
            prompt_mask,
            torch.zeros(4, dtype=torch.long, device=args.device),
        ).context
    generated_asts = []
    ast_masks = []
    parent_rows = []
    level_rows = []
    blocked = None
    for index in range(len(PROGRAM_NAMES)):
        sampled = stage_a.sample_target_free(
            contexts[index : index + 1],
            steps=4,
            noise=torch.randn(
                1,
                32,
                24,
                device=args.device,
                generator=torch.Generator(device=args.device).manual_seed(
                    400 + index
                ),
            ),
        )
        try:
            decoded_source = decode_source_tokens(
                sampled.source_token_ids[0],
                vocabulary,
            )
            parse_ast(decoded_source, "python")
        except ValueError as error:
            blocked = str(error)
            break
        generated_asts.append(sampled.endpoint)
        ast_masks.append(sampled.predicted_node_mask())
        parent, level = predict_generated_topology_batch(sampled, ast_masks[-1])
        parent_rows.append(parent)
        level_rows.append(level)
        if not decoded_source:
            blocked = "generated AST decoded to empty source"
            break
    if blocked is not None:
        report = {
            "version": "nlp-result-cfm-execution-probe-v1",
            "status": "blocked",
            "device": args.device,
            "blocker": blocked,
            "node_mask_source": "predicted_node_count",
            "prompt_encoder": "semantic_token_transformer",
            "target_ast_used_at_inference": False,
            "teacher_ast_used_at_inference": False,
            "evidence_boundary": (
                "Stage-A target-free structure failed; Stage-B capability "
                "and execution evidence are unavailable."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    generated_ast = torch.cat(generated_asts)
    ast_mask = torch.cat(ast_masks)
    parent_indices = torch.cat(parent_rows)
    node_levels = torch.cat(level_rows)
    stage_b = GeneratedASTResultCFM(
        ResultCFMConfig(hidden_size=24, rank=8)
    ).to(args.device)
    execution_head = ResultExecutionHead(24).to(args.device)
    optimizer = torch.optim.AdamW(
        [*stage_b.parameters(), *execution_head.parameters()],
        lr=0.02,
    )
    targets = F.one_hot(
        torch.arange(4, device=args.device),
        num_classes=24,
    ).float()[:, None].expand(-1, 5, -1)
    noise = torch.randn_like(targets)
    for _step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        flow_loss, output = stage_b.training_loss(
            targets,
            contexts,
            generated_ast,
            ast_mask,
            parent_indices=parent_indices,
            node_levels=node_levels,
            noise=noise,
            time=torch.tensor(
                [0.2, 0.4, 0.6, 0.8],
                device=args.device,
            ),
        )
        action_loss = F.cross_entropy(
            execution_head(output.endpoint),
            torch.arange(4, device=args.device),
        )
        (flow_loss.total + action_loss).backward()
        optimizer.step()
    with torch.no_grad():
        sampled = stage_b.sample(
            contexts,
            generated_ast,
            ast_mask,
            parent_indices=parent_indices,
            node_levels=node_levels,
            result_length=5,
            steps=4,
            noise=noise,
        )
        stage_b_ids = execution_head(sampled.endpoint).argmax(dim=-1)
        control = NoResultFlowControl(24).to(args.device)
        # The control sees the same prompt for every operation.
        shared_prompt = contexts.mean(dim=0, keepdim=True).expand(4, -1)
        control_ids = control(shared_prompt).argmax(dim=-1)
    stage_b_score = _program_score(stage_b_ids)
    control_score = _program_score(control_ids)
    report = {
        "version": "nlp-result-cfm-execution-probe-v1",
        "status": "pass" if stage_b_score > control_score else "fail",
        "device": args.device,
        "cases": 4,
        "programs": list(PROGRAM_NAMES),
        "stage_b_predictions": stage_b_ids.tolist(),
        "no_result_flow_predictions": control_ids.tolist(),
        "stage_b_executable_success": stage_b_score,
        "no_result_flow_executable_success": control_score,
        "primary_condition": "Stage-A generated AST only",
        "node_mask_source": "predicted_node_count",
        "prompt_encoder": "semantic_token_transformer",
        "target_ast_used_at_inference": False,
        "teacher_ast_used_at_inference": False,
        "evidence_boundary": (
            "Four-program synthetic executable probe only; no public coding benchmark claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
