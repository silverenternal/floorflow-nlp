#!/usr/bin/env python3
"""Audit Stage-A AST objectives, target-free sampling, and NFE parseability."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from floorflow.nlp_flow.ast_cfm import (
    ASTCodecVocabulary,
    ASTStageAFlow,
    ast_cfm_objective_losses,
    collate_ast_batches,
    decode_generated_template,
    encode_ast_graph,
)
from floorflow.nlp_flow.ast_codec import ast_metrics, parse_ast, render_source
from floorflow.nlp_flow.ast_generation import decode_structured_ast
from floorflow.nlp_flow.ast_token_decoder import decode_source_tokens
from floorflow.nlp_flow.cfm_training import (
    CFMTrainingConfig,
    sample_flow_times,
)
from floorflow.nlp_flow.model.prompt_encoding import SemanticPromptEncoder

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_NFE = 4
NFE_VALUES = (1, 2, 4, 8)
TRAINING_STEPS = 1_500
TOKENIZER = ROOT / ".data/nlp/tokenizer_v1/tokenizer.json"

SAMPLES = (
    {
        "language": "python",
        "prompt": "implement Python integer addition function",
        "source": "def add(a, b):\n    return a + b\n",
    },
    {
        "language": "rust",
        "prompt": "implement Rust integer addition function",
        "source": "fn add(a: i32, b: i32) -> i32 { a + b }\n",
    },
    {
        "language": "cpp",
        "prompt": "implement C++ integer addition function",
        "source": "int add(int a, int b) { return a + b; }\n",
    },
)


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _build():
    graphs = [
        parse_ast(sample["source"], sample["language"])
        for sample in SAMPLES
    ]
    vocabulary = ASTCodecVocabulary.from_graphs(graphs)
    template_to_id = vocabulary.template_to_id
    batches = [
        encode_ast_graph(
            graph,
            vocabulary,
            max_nodes=32,
            template_id=template_to_id[render_source(graph)],
        )
        for graph in graphs
    ]
    return graphs, vocabulary, collate_ast_batches(batches)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return "cpu"


def _train(
    vocabulary: ASTCodecVocabulary,
    batch,
    *,
    training_steps: int,
    timestep_sampler: str,
    device: str,
    seed: int,
) -> tuple[ASTStageAFlow, SemanticPromptEncoder, torch.Tensor]:
    torch.manual_seed(seed)
    model = ASTStageAFlow(
        vocabulary,
        hidden_size=48,
        num_heads=4,
        max_nodes=32,
    ).to(device)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    prompt_ids = []
    prompt_masks = []
    for sample in SAMPLES:
        ids = tokenizer.encode(sample["prompt"]).ids[:128]
        prompt_ids.append(ids)
        prompt_masks.append([True] * len(ids))
    max_prompt = max(len(ids) for ids in prompt_ids)
    input_ids = torch.zeros(len(SAMPLES), max_prompt, dtype=torch.long)
    prompt_mask = torch.zeros(len(SAMPLES), max_prompt, dtype=torch.bool)
    for row, ids in enumerate(prompt_ids):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        prompt_mask[row, : len(ids)] = True
    input_ids = input_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    language_ids = torch.tensor([0, 1, 2], device=device)
    encoder = SemanticPromptEncoder(
        tokenizer.get_vocab_size(),
        model.hidden_size,
        max_tokens=128,
        layers=1,
        heads=4,
        language_count=3,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *encoder.parameters()],
        lr=1.5e-3,
        weight_decay=0.0,
    )
    generator = torch.Generator(device=device).manual_seed(seed + 83)
    sampler = CFMTrainingConfig(timestep_sampler=timestep_sampler)
    for _step in range(training_steps):
        model.train()
        encoder.train()
        optimizer.zero_grad(set_to_none=True)
        context = encoder(
            input_ids,
            prompt_mask,
            language_ids,
        ).context
        target = model.embed_target(batch, device)
        noise = torch.randn(
            target.shape,
            device=device,
            generator=generator,
        )
        time = sample_flow_times(
            target.shape[0],
            sampler,
            device=device,
            generator=generator,
        )
        state = (
            (1.0 - time[:, None, None]) * noise
            + time[:, None, None] * target
        )
        output = model(
            state,
            time,
            context,
            batch,
            source_token_inputs=batch.source_token_ids[:, :-1],
            source_token_node_ids=batch.source_token_node_ids[:, :-1],
            source_token_phase_ids=batch.source_token_phase_ids[:, :-1],
        )
        categorical_output = model(
            noise,
            torch.zeros_like(time),
            context,
            batch,
            source_token_inputs=batch.source_token_ids[:, :-1],
            source_token_node_ids=batch.source_token_node_ids[:, :-1],
            source_token_phase_ids=batch.source_token_phase_ids[:, :-1],
            source_token_teacher_state=target,
        )
        losses = ast_cfm_objective_losses(
            output,
            target,
            noise,
            batch,
            categorical_output=categorical_output,
        )
        losses.total.backward()
        optimizer.step()
    encoder.eval()
    with torch.no_grad():
        context = encoder(
            input_ids,
            prompt_mask,
            language_ids,
        ).context
    return model.eval(), encoder, context


def _objective_gradient_audit(
    model: ASTStageAFlow,
    batch,
    context: torch.Tensor,
    device: str,
    seed: int,
) -> dict[str, object]:
    model.zero_grad(set_to_none=True)
    target = model.embed_target(batch, device)
    noise = torch.randn(
        target.shape,
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed + 202),
    )
    time = torch.tensor([0.2, 0.5, 0.8], device=device)
    state = (1.0 - time[:, None, None]) * noise + time[:, None, None] * target
    output = model(
        time=time,
        state=state,
        context=context,
        batch=batch,
        source_token_inputs=batch.source_token_ids[:, :-1],
        source_token_node_ids=batch.source_token_node_ids[:, :-1],
        source_token_phase_ids=batch.source_token_phase_ids[:, :-1],
    )
    categorical_output = model(
        state=noise,
        time=torch.zeros_like(time),
        context=context,
        batch=batch,
        source_token_inputs=batch.source_token_ids[:, :-1],
        source_token_node_ids=batch.source_token_node_ids[:, :-1],
        source_token_phase_ids=batch.source_token_phase_ids[:, :-1],
        source_token_teacher_state=target,
    )
    losses = ast_cfm_objective_losses(
        output,
        target,
        noise,
        batch,
        categorical_output=categorical_output,
    )
    losses.total.backward()
    head_gradients = {
        "velocity": model.velocity_head.weight.grad is not None,
        "categorical_conditioning": (
            model.categorical_conditioner[-1].weight.grad is not None
        ),
        "parent_context_conditioning": (
            model.parent_context_conditioner[-1].weight.grad is not None
        ),
        "node_type": model.node_type_head.weight.grad is not None,
        "edge_type": model.edge_type_head.weight.grad is not None,
        "lexical": model.lexical_head.weight.grad is not None,
        "gap": model.gap_head.weight.grad is not None,
        "level": model.level_head.weight.grad is not None,
        "parent": model.parent_query.weight.grad is not None,
        "root": model.root_head.weight.grad is not None,
        "node_count": model.node_count_head[-1].weight.grad is not None,
        "source_length": model.source_length_head.weight.grad is not None,
        "template": model.template_head.weight.grad is not None,
        "parseability": model.parseability_head.weight.grad is not None,
        "source_tokens": model.token_endpoint.output.weight.grad is not None,
        "source_tokens_free": model.token_endpoint.output.weight.grad is not None,
    }
    finite = {
        name: bool(torch.isfinite(value))
        for name, value in losses.as_dict().items()
    }
    return {
        "status": "pass" if all(head_gradients.values()) and all(finite.values()) else "fail",
        "losses": {name: float(value.detach()) for name, value in losses.as_dict().items()},
        "finite": finite,
        "head_gradients": head_gradients,
    }


def _nfe_curve(
    model: ASTStageAFlow,
    context: torch.Tensor,
    batch,
    vocabulary: ASTCodecVocabulary,
    references,
    device: str,
    seed: int,
) -> dict[str, object]:
    rows = {}
    all_primary_token_parse = True
    for nfe in NFE_VALUES:
        nfe_rows = []
        for index, sample in enumerate(SAMPLES):
            noise = torch.randn(
                1,
                model.max_nodes,
                model.hidden_size,
                device=device,
                generator=torch.Generator(device=device).manual_seed(
                    seed + 300 + index
                ),
            )
            output = model.sample_target_free(
                context[index : index + 1],
                steps=nfe,
                noise=noise,
            )
            grammar_source = decode_generated_template(
                output,
                vocabulary,
                allow_template_shortcut=True,
            )
            try:
                generated = parse_ast(grammar_source, sample["language"])
                grammar_parseable = True
            except ValueError:
                generated = None
                grammar_parseable = False
            try:
                token_source = decode_source_tokens(
                    output.source_token_ids[0],
                    vocabulary,
                )
                if not token_source.strip():
                    raise ValueError("generated token endpoint is empty")
                parse_ast(token_source, sample["language"])
                token_parseable = True
                token_error = None
            except ValueError as error:
                token_source = ""
                token_parseable = False
                token_error = str(error)
            token_ids = (
                []
                if output.source_token_ids is None
                else output.source_token_ids[0].tolist()
            )
            predicted_source_length = float(
                (
                    torch.softmax(output.source_length_logits[0], dim=-1)
                    * torch.arange(
                        output.source_length_logits.shape[-1],
                        device=output.source_length_logits.device,
                    )
                ).sum()
            )
            try:
                structured = decode_structured_ast(
                    output,
                    vocabulary,
                    sample["language"],
                )
                structure_parseable = True
                structured_source = structured.source
            except ValueError as error:
                structure_parseable = False
                structured_source = ""
                structure_error = str(error)
            else:
                structure_error = None
            nfe_rows.append(
                {
                    "language": sample["language"],
                    "grammar_shell_parseable": grammar_parseable,
                    "grammar_shell_metrics": (
                        None
                        if generated is None
                        else ast_metrics(generated, references[index])
                    ),
                    "grammar_shell_sha256": hashlib.sha256(
                        grammar_source.encode("utf-8")
                    ).hexdigest(),
                    "token_path_parseable": token_parseable,
                    "token_path_error": token_error,
                    "token_path_source": token_source,
                    "token_path_ids": token_ids,
                    "context_sha256": _tensor_digest(
                        context[index : index + 1]
                    ),
                    "ast_endpoint_sha256": _tensor_digest(
                        output.endpoint
                    ),
                    "source_token_logits_sha256": (
                        None
                        if output.source_token_logits is None
                        else _tensor_digest(output.source_token_logits)
                    ),
                    "predicted_source_length": predicted_source_length,
                    "structure_path_parseable": structure_parseable,
                    "structure_path_error": structure_error,
                    "structure_path_source": structured_source,
                }
            )
            if nfe == PRIMARY_NFE and not token_parseable:
                all_primary_token_parse = False
        rows[str(nfe)] = {
            "grammar_shell_parse_rate": sum(
                row["grammar_shell_parseable"] for row in nfe_rows
            )
            / len(nfe_rows),
            "token_path_parse_rate": sum(
                row["token_path_parseable"] for row in nfe_rows
            )
            / len(nfe_rows),
            "samples": nfe_rows,
        }
    return {
        "status": "pass" if all_primary_token_parse else "fail",
        "primary_nfe": PRIMARY_NFE,
        "values": rows,
        "selection_rule": "All NFE values were fixed before sampling; no post-hoc NFE selection.",
        "token_path_boundary": (
            "Only predicted-topology token endpoints can pass; grammar "
            "templates are reported as diagnostics and cannot mask failure."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_cfm_objective_audit_v1.json",
    )
    parser.add_argument(
        "--nfe-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_cfm_nfe_curve_v1.json",
    )
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument(
        "--timestep-sampler",
        choices=("uniform", "logit_normal"),
        default="logit_normal",
    )
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    graphs, vocabulary, batch = _build()
    model, prompt_encoder, context = _train(
        vocabulary,
        batch,
        training_steps=args.training_steps,
        timestep_sampler=args.timestep_sampler,
        device=str(device),
        seed=args.seed,
    )
    objective = _objective_gradient_audit(
        model,
        batch,
        context,
        str(device),
        args.seed,
    )
    objective.update(
        {
            "version": "nlp-ast-cfm-objective-audit-v1",
            "target_free_sample_signature": {
                name: str(parameter)
                for name, parameter in inspect.signature(
                    model.sample_target_free
                ).parameters.items()
            },
            "sample_api_has_target_argument": any(
                "target" in name
                for name in inspect.signature(
                    model.sample_target_free
                ).parameters
            ),
            "training_timestep_sampler": args.timestep_sampler,
            "training_steps": args.training_steps,
            "training_device": str(device),
            "seed": args.seed,
            "prompt_encoder_parameters": sum(
                parameter.numel()
                for parameter in prompt_encoder.parameters()
            ),
            "prompt_encoder": "semantic_token_transformer",
            "categorical_training_pass": "fixed_t0_endpoint",
            "losses": objective["losses"],
            "evidence_boundary": (
                "Synthetic Stage-A objective and gradient contract only; "
                "no corpus-scale AST-generation quality claim."
            ),
        }
    )
    nfe = {
        "version": "nlp-ast-cfm-nfe-curve-v1",
        **_nfe_curve(
            model,
            context,
            batch,
            vocabulary,
            graphs,
            str(device),
            args.seed,
        ),
        "training_steps": args.training_steps,
        "training_timestep_sampler": args.timestep_sampler,
        "training_device": str(device),
        "seed": args.seed,
        "node_mask_source": "predicted_node_count",
        "structure_decoder": "predicted_parent_and_field_tree",
        "evidence_boundary": (
            "Three fixed parser-validated grammar shells and deterministic samples only; "
            "the grammar-shell path is not evidence for open-domain AST synthesis."
        ),
    }
    for path, report in (
        (args.objective_output, objective),
        (args.nfe_output, nfe),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(path)
        print(json.dumps(report, indent=2))
    if objective["status"] != "pass" or nfe["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
