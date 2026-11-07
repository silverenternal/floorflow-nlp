"""Training and evaluation procedures for the causal-prefix policy."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping

import torch

from newera.nlp_flow.engineering.components.free_schedule_spec.schedule_free import ScheduleFreeAdamW
from newera.nlp_flow.components.research.experiments.config.experiment_deadline import ExperimentDeadline
from newera.nlp_flow.components.generation.trees.hierarchical.hierarchical_tree_training import (
    RolloutCurriculum,
    hierarchical_tree_losses,
)
from newera.nlp_flow.components.generation.policies.prefix.prefix_policy_data import PrefixPolicyBatch, encode_prefix_rows
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_policy import PrefixTreePolicy
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_action_data import action_record
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import (
    REAL_ACTIONS,
)
from newera.nlp_flow.components.generation.trees.runtime.tree_completion_runtime import TreeCompletionBudget


def _move_batch(batch: PrefixPolicyBatch, device: torch.device) -> PrefixPolicyBatch:
    return PrefixPolicyBatch(
        **{
            name: value.to(device)
            for name, value in vars(batch).items()
            if isinstance(value, torch.Tensor)
        },
        history={
            name: value.to(device) for name, value in batch.history.items()
        },
        previous={
            name: value.to(device) for name, value in batch.previous.items()
        },
    )


def _slice_batch(batch: PrefixPolicyBatch, index: torch.Tensor) -> PrefixPolicyBatch:
    maximum = int(batch.step_mask[index].sum(dim=1).max())
    blueprint_values = {
        name: value[index]
        for name in ("blueprint_node", "blueprint_edge", "blueprint_parent", "blueprint_presence")
        if (value := getattr(batch, name)) is not None
    }
    values = {
        name: value[index]
        for name, value in vars(batch).items()
        if isinstance(value, torch.Tensor)
        and name not in {
            "prompts", "prompt_masks", "prompt_token_ids", "prompt_token_mask", "node_budget",
            "blueprint_node", "blueprint_edge", "blueprint_parent", "blueprint_presence",
        }
    }
    values = {
        name: value[:, :maximum] if value.ndim >= 2 else value
        for name, value in values.items()
    }
    return PrefixPolicyBatch(
        prompts=batch.prompts[index],
        prompt_masks=batch.prompt_masks[index],
        prompt_token_ids=batch.prompt_token_ids[index],
        prompt_token_mask=batch.prompt_token_mask[index],
        node_budget=batch.node_budget[index],
        history={
            name: value[index, :maximum]
            for name, value in batch.history.items()
        },
        previous={
            name: value[index, :maximum]
            for name, value in batch.previous.items()
        },
        **blueprint_values,
        **values,
    )


def _loss(
    output: Mapping[str, torch.Tensor],
    batch: PrefixPolicyBatch,
    *,
    aligned_mask: torch.Tensor | None = None,
    viability_target: torch.Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    return hierarchical_tree_losses(
        output,
        batch,
        aligned_mask=aligned_mask,
        viability_target=viability_target,
        weights=weights,
    )[0]


def _build_rollout_batch(
    model: PrefixTreePolicy,
    rows: list[Mapping[str, object]],
    grammars: Mapping[str, object],
    vocabulary: Mapping[str, object],
    *,
    device: torch.device,
    max_steps: int,
    local_transition_mask: bool = True,
    lexical_candidate_mask: bool = True,
    completion_budgets: Mapping[str, TreeCompletionBudget] | None = None,
    deadline: ExperimentDeadline | None = None,
) -> tuple[PrefixPolicyBatch, torch.Tensor, dict[str, int]]:
    from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_runtime import generate_prefix_tree

    rollout_rows = []
    outcomes = []
    for row in rows:
        if deadline is not None:
            deadline.check("train_only_rollout_batch")
        RolloutCurriculum.require_train_row(row)
        grammar = grammars[str(row["language"])]
        generated_source, generated_actions, _ = generate_prefix_tree(
            model,
            str(row["prompt"]),
            str(row["language"]),
            grammar,
            vocabulary,
            device=device,
            max_steps=max_steps,
            local_transition_mask=local_transition_mask,
            lexical_candidate_mask=lexical_candidate_mask,
            completion_budget=(
                completion_budgets[str(row["language"])]
                if completion_budgets is not None
                else None
            ),
            deadline=deadline,
        )
        if not generated_actions:
            raise ValueError("free-running tree policy produced no action for a train row")
        rollout_rows.append(
            {
                "case_id": row["case_id"],
                "language": row["language"],
                "prompt": row["prompt"],
                "split": "train",
                "actions": [action_record(action) for action in generated_actions],
            }
        )
        outcomes.append(int(generated_source is not None))
    budget_minimums = (
        {language: budget.minimum_root_nodes() for language, budget in completion_budgets.items()}
        if completion_budgets is not None
        else None
    )
    if model.budget_residual and budget_minimums is None:
        raise ValueError("residual budget rollouts require completion contracts")
    batch = encode_prefix_rows(
        rollout_rows,
        grammars,
        vocabulary,
        budget_minimums=budget_minimums,
    )
    labels = torch.tensor(outcomes, dtype=torch.float32, device=device)
    viability = labels[:, None].expand(batch.step_mask.shape).clone()
    return batch, viability, {
        "rollout_rows": len(outcomes),
        "successful_rollouts": sum(outcomes),
        "failed_rollouts": len(outcomes) - sum(outcomes),
    }


def train_prefix_tree_policy(
    train_batch: PrefixPolicyBatch,
    validation_batch: PrefixPolicyBatch,
    vocabulary: Mapping[str, object],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_rows: int,
    hidden_size: int,
    layers: int,
    heads: int,
    learning_rate: float = 2.5e-3,
    warmup_steps: int = 0,
    gradient_clip_norm: float = 1.0,
    loss_weights: Mapping[str, float] | None = None,
    train_rows: list[Mapping[str, object]] | None = None,
    grammars: Mapping[str, object] | None = None,
    rollout_curriculum: RolloutCurriculum | None = None,
    local_transition_mask: bool = True,
    lexical_candidate_mask: bool = True,
    completion_budgets: Mapping[str, TreeCompletionBudget] | None = None,
    blueprint_conditioning: bool = True,
    node_kind_rank: int | None = None,
    budget_residual: bool = False,
    initial_state_dict: Mapping[str, torch.Tensor] | None = None,
    progress_callback: Callable[[PrefixTreePolicy, int, float], None] | None = None,
    deadline: ExperimentDeadline | None = None,
) -> tuple[PrefixTreePolicy, dict[str, object]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_data = _move_batch(train_batch, device)
    validation_data = _move_batch(validation_batch, device)
    model = PrefixTreePolicy(
        vocabulary,
        hidden_size=hidden_size,
        layers=layers,
        heads=heads,
        max_steps=train_batch.action.shape[1],
        blueprint_conditioning=blueprint_conditioning,
        node_kind_rank=node_kind_rank,
        budget_residual=budget_residual,
    ).to(device)
    if initial_state_dict is not None:
        current = model.state_dict()
        for name, value in initial_state_dict.items():
            if name in current and current[name].shape == value.shape:
                current[name] = value.to(device=device, dtype=current[name].dtype)
        model.load_state_dict(current)
    optimizer = ScheduleFreeAdamW(
        model.parameters(),
        lr=learning_rate,
        warmup_steps=warmup_steps,
    )
    generator = torch.Generator().manual_seed(seed + 1)
    history = []
    python_rng = random.Random(seed + 2)
    curriculum = rollout_curriculum or RolloutCurriculum()
    if (train_rows is None) != (grammars is None):
        raise ValueError("train rows and grammars must be supplied together")
    if train_rows is not None and len(train_rows) != train_batch.action.shape[0]:
        raise ValueError("train row identities must align with the encoded batch")
    if budget_residual and (
        train_batch.node_budget_minimum is None
        or validation_batch.node_budget_minimum is None
    ):
        raise ValueError("residual budget training requires certified minima for both splits")
    rollout_counts = {"rollout_rows": 0, "successful_rollouts": 0, "failed_rollouts": 0}
    optimizer_steps = 0
    for _epoch in range(epochs):
        if deadline is not None:
            deadline.check("training_epoch")
        model.train()
        optimizer.train()
        order = torch.randperm(train_data.action.shape[0], generator=generator)
        total = 0.0
        steps = 0
        for start in range(0, len(order), batch_rows):
            if deadline is not None:
                deadline.check("optimizer_batch")
            index = order[start : start + batch_rows].to(device)
            batch = _slice_batch(train_data, index)
            viability_target = None
            teacher_probability = curriculum.teacher_probability(optimizer_steps)
            if train_rows is not None and python_rng.random() > teacher_probability:
                selected_rows = [train_rows[int(row_index)] for row_index in index.cpu().tolist()]
                rollout_batch, viability_target, counts = _build_rollout_batch(
                    model,
                    selected_rows,
                    grammars,
                    vocabulary,
                    device=device,
                    max_steps=train_batch.action.shape[1],
                    local_transition_mask=local_transition_mask,
                    lexical_candidate_mask=lexical_candidate_mask,
                    completion_budgets=completion_budgets,
                    deadline=deadline,
                )
                batch = _move_batch(rollout_batch, device)
                for key, value in counts.items():
                    rollout_counts[key] += value
            output = model(batch)
            loss = _loss(
                output,
                batch,
                viability_target=viability_target,
                weights=loss_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer_steps += 1
            if deadline is not None:
                deadline.check("optimizer_step")
            total += float(loss.detach()) * len(index)
            steps += len(index)
        history.append(total / max(1, steps))
        if progress_callback is not None:
            progress_callback(model, _epoch + 1, history[-1])
    model.eval()
    optimizer.eval()
    if deadline is not None:
        deadline.check("teacher_forced_validation")
    metrics = evaluate_prefix_policy(
        model,
        validation_data,
        batch_rows=batch_rows,
        deadline=deadline,
    )
    metrics["loss_history"] = history
    metrics["train_rows"] = train_batch.action.shape[0]
    metrics["validation_rows"] = validation_batch.action.shape[0]
    metrics["optimizer"] = {
        "name": "ScheduleFreeAdamW",
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "gradient_clip_norm": gradient_clip_norm,
    }
    metrics["loss_weights"] = dict(loss_weights or {})
    metrics["runtime_controls"] = {
        "local_transition_mask": local_transition_mask,
        "lexical_candidate_mask": lexical_candidate_mask,
        "terminalizable_completion_mask": completion_budgets is not None,
        "blueprint_conditioning": blueprint_conditioning,
        "node_kind_rank": node_kind_rank,
        "budget_residual": budget_residual,
    }
    metrics["training_regime"] = "teacher_forced_only"
    metrics["optimizer_steps"] = optimizer_steps
    metrics["examples_seen"] = int(train_batch.action.shape[0]) * int(epochs)
    metrics["action_steps_seen"] = (
        sum(len(row["actions"]) for row in train_rows) * int(epochs)
        if train_rows is not None
        else int(train_batch.step_mask.sum()) * int(epochs)
    )
    if train_rows is not None:
        metrics["training_regime"] = "teacher_forced_plus_train_only_free_running_rollout"
        metrics["rollout_training"] = {
            **rollout_counts,
            "optimizer_steps": optimizer_steps,
            "teacher_probability_first_step": curriculum.teacher_probability(0),
            "teacher_probability_last_step": curriculum.teacher_probability(max(0, optimizer_steps - 1)),
            "sampled_actions_used_as_training_inputs": True,
            "viability_negative_examples": rollout_counts["failed_rollouts"],
            "validation_or_test_rollout_inputs": False,
        }
    return model, metrics


@torch.no_grad()
def evaluate_prefix_policy(
    model: PrefixTreePolicy,
    batch: PrefixPolicyBatch,
    *,
    batch_rows: int,
    deadline: ExperimentDeadline | None = None,
) -> dict[str, object]:
    correct = {name: 0 for name in ("action", "node_kind", "edge_field", "parent", "leaf_class")}
    total = {name: 0 for name in correct}
    budget_correct = 0
    budget_total = 0
    blueprint_correct = {name: 0 for name in ("node", "edge", "parent", "presence")}
    blueprint_total = {name: 0 for name in blueprint_correct}
    for start in range(0, batch.action.shape[0], batch_rows):
        if deadline is not None:
            deadline.check("teacher_forced_validation_batch")
        index = torch.arange(
            start,
            min(start + batch_rows, batch.action.shape[0]),
            device=batch.action.device,
        )
        local = _slice_batch(batch, index)
        output = model(local)
        action = local.action
        checks = {
            "action": action,
            "node_kind": local.node_kind,
            "edge_field": local.edge_field,
            "parent": local.candidate_target,
            "leaf_class": local.leaf_class,
        }
        masks = {
            "action": local.step_mask,
            "node_kind": local.step_mask & action.eq(REAL_ACTIONS.index("open_node")),
            "edge_field": local.step_mask & action.eq(REAL_ACTIONS.index("attach_edge")),
            "parent": local.step_mask & action.eq(REAL_ACTIONS.index("attach_edge")),
            "leaf_class": local.step_mask & action.eq(REAL_ACTIONS.index("declare_leaf")),
        }
        for name, target in checks.items():
            active = masks[name]
            if not active.any():
                continue
            predicted = output[name][active].argmax(dim=-1)
            correct[name] += int(predicted.eq(target[active]).sum())
            total[name] += int(active.sum())
        budget_correct += int(
            output["node_budget"].argmax(dim=-1).eq(local.node_budget).sum()
        )
        budget_total += len(index)
        if "blueprint_node_logits" in output:
            present = local.blueprint_presence.bool()
            edge_present = present & local.blueprint_parent.ne(0)
            checks = {
                "node": (output["blueprint_node_logits"], local.blueprint_node, present),
                "edge": (output["blueprint_edge_logits"], local.blueprint_edge, edge_present),
                "parent": (output["blueprint_parent_logits"], local.blueprint_parent, present),
            }
            for name, (logits, target, active) in checks.items():
                blueprint_correct[name] += int(logits[active].argmax(dim=-1).eq(target[active]).sum())
                blueprint_total[name] += int(active.sum())
            presence_prediction = output["blueprint_presence_logits"].ge(0)
            blueprint_correct["presence"] += int(
                presence_prediction.eq(local.blueprint_presence.bool()).sum()
            )
            blueprint_total["presence"] += local.blueprint_presence.numel()
    return {
        name: correct[name] / max(1, total[name])
        for name in correct
    } | {
        "node_budget_accuracy": budget_correct / max(1, budget_total),
        "blueprint_accuracy": {
            name: blueprint_correct[name] / max(1, blueprint_total[name])
            if blueprint_total[name]
            else None
            for name in blueprint_correct
        },
        "target_ast_input": False,
        "repair_or_retry": False,
    }


__all__ = ["evaluate_prefix_policy", "train_prefix_tree_policy"]
