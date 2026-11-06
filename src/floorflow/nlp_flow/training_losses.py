"""Joint training losses for the compact NLP experiment stack."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.nlp_flow.model.proxy_losses import proxy_losses


def batch_loss(
    model: torch.nn.Module,
    batch,
    *,
    device: torch.device,
    autocast,
    emergence_weight: float = 0.0,
    dynamic_depth_weight: float = 0.0,
    dynamic_target_depth: float = 2.0,
    conversation_control_weight: float = 0.0,
    benchmark_proxy_weight: float = 0.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    dict[str, float | int] | None,
    dict[str, float] | None,
]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(device)
    turn_ids = batch.get("turn_ids")
    if isinstance(turn_ids, torch.Tensor):
        turn_ids = turn_ids.to(device)
    family_ids = batch.get("family_ids")
    if isinstance(family_ids, torch.Tensor):
        family_ids = family_ids.to(device)
    language_ids = batch.get("language_ids")
    if isinstance(language_ids, torch.Tensor):
        language_ids = language_ids.to(device)
    with autocast:
        model_kwargs = {"attention_mask": attention_mask}
        if turn_ids is not None and getattr(model, "turn_embedding", None) is not None:
            model_kwargs["turn_ids"] = turn_ids
        if (
            family_ids is not None
            and language_ids is not None
            and getattr(model, "task_family_embedding", None) is not None
        ):
            model_kwargs["family_ids"] = family_ids
            model_kwargs["language_ids"] = language_ids
        output = model(input_ids, **model_kwargs)
        logits = output.token_logits
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    emergence_total = None
    emergence_metrics = None
    if emergence_weight > 0.0 and getattr(model, "emergence", None) is not None:
        emergence_result = model.emergence(
            output.hidden_states,
            attention_mask,
            surface_targets=batch.get("surface_valid"),
            execution_targets=batch.get("execution_valid"),
        )
        emergence_total = emergence_result.total
        emergence_metrics = {
            name: float(value.detach())
            for name, value in emergence_result.as_dict().items()
        }
        loss = loss + emergence_weight * emergence_total
    dynamic_metrics = None
    if output.dynamic_diagnostics is not None:
        diagnostics = output.dynamic_diagnostics
        dynamic_metrics = {
            "mean_depth": float(diagnostics["mean_depth"].detach()),
            "max_depth": float(diagnostics["max_depth"].detach()),
            "expected_depth": float(diagnostics["ponder_cost"].detach()),
            "lane_balance_loss": float(diagnostics["lane_balance_loss"].detach()),
            "serial_token_steps": int(diagnostics["serial_token_steps"].detach()),
            "parallel_token_lane_calls": int(
                diagnostics["parallel_token_lane_calls"].detach()
            ),
            "dense_serial_token_calls": int(
                diagnostics["dense_serial_token_calls"]
            ),
            "executed_steps": int(diagnostics["executed_steps"]),
        }
    if dynamic_depth_weight > 0.0 and output.dynamic_diagnostics is not None:
        depth = output.dynamic_diagnostics["ponder_cost"]
        lane = output.dynamic_diagnostics["lane_balance_loss"]
        loss = loss + dynamic_depth_weight * (
            (depth - depth.new_tensor(dynamic_target_depth)).square() + lane
        )
    loss = loss + _conversation_loss(
        model,
        output,
        batch,
        device=device,
        attention_mask=attention_mask,
        weight=conversation_control_weight,
    )
    loss = loss + _benchmark_proxy_loss(
        model,
        output,
        batch,
        device=device,
        weight=benchmark_proxy_weight,
    )
    return loss, emergence_total, dynamic_metrics, emergence_metrics


def _conversation_loss(
    model: torch.nn.Module,
    output,
    batch,
    *,
    device: torch.device,
    attention_mask: torch.Tensor | None,
    weight: float,
) -> torch.Tensor:
    if (
        weight <= 0.0
        or output.conversation_logits is None
        or getattr(model, "conversation_head", None) is None
        or "continue_targets" not in batch
        or "validity_targets" not in batch
    ):
        return output.token_logits.new_zeros(())
    targets = {
        "continue_logits": batch["continue_targets"].to(device),
        "turn_boundary_logits": batch["turn_boundary_targets"].to(device),
        "reply_mode_logits": batch["reply_mode_targets"].to(device),
        "language_logits": batch["language_targets"].to(device),
        "validity_logits": batch["validity_targets"].to(device),
    }
    masks = {
        "continue_logits": (
            attention_mask
            if attention_mask is not None
            else torch.ones_like(targets["continue_logits"], dtype=torch.bool)
        ),
        "turn_boundary_logits": targets["turn_boundary_logits"].ge(0),
        "reply_mode_logits": targets["reply_mode_logits"].ge(0),
        "language_logits": targets["language_logits"].ge(0),
        "validity_logits": batch["validity_mask"].to(device),
    }
    losses = model.conversation_head.losses(
        output.conversation_logits,
        targets,
        masks,
    )
    return weight * sum(losses.values())


def _benchmark_proxy_loss(
    model: torch.nn.Module,
    output,
    batch,
    *,
    device: torch.device,
    weight: float,
) -> torch.Tensor:
    if (
        weight <= 0.0
        or output.benchmark_proxy is None
        or getattr(model, "benchmark_proxy", None) is None
    ):
        return output.token_logits.new_zeros(())
    attention_mask = batch.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        mask = attention_mask.to(device=device, dtype=torch.bool)
        weights = mask.to(output.hidden_states.dtype)[..., None]
        pooled = (output.hidden_states * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
    else:
        pooled = output.hidden_states.mean(dim=1)
    losses = proxy_losses(
        model.benchmark_proxy,
        output.benchmark_proxy,
        batch,
        device=device,
        hidden=output.hidden_states,
        pooled=pooled,
    )
    return weight * sum(losses.values()) if losses else output.token_logits.new_zeros(())


__all__ = ["batch_loss"]
