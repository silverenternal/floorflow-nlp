"""One routed loss registry for every generated benchmark-proxy target."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from floorflow.nlp_flow._generated_protocol import (
    BENCHMARK_REGRESSION_TARGETS,
    BENCHMARK_SCALAR_TARGETS,
    BENCHMARK_TOKEN_TARGETS,
)
from floorflow.nlp_flow.model.benchmark_proxy import (
    BenchmarkProxyObjective,
    BenchmarkProxyPrediction,
)


def proxy_losses(
    objective: BenchmarkProxyObjective,
    prediction: BenchmarkProxyPrediction,
    batch: Mapping[str, object],
    *,
    device: torch.device,
    hidden: torch.Tensor,
    pooled: torch.Tensor,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    for name in BENCHMARK_SCALAR_TARGETS:
        target = batch.get(f"proxy_{name}")
        if not isinstance(target, torch.Tensor):
            continue
        target = target.to(device)
        active = target.ge(0.0)
        if active.any():
            losses[f"L_{name}"] = F.binary_cross_entropy_with_logits(
                prediction.scalar_logits[name][active],
                target[active],
            )

    regression_targets = {
        name: batch[f"proxy_regression_{name}"].to(device)
        for name in BENCHMARK_REGRESSION_TARGETS
        if isinstance(batch.get(f"proxy_regression_{name}"), torch.Tensor)
    }
    regression_mask = batch.get("proxy_regression_mask")
    masks = None
    if isinstance(regression_mask, torch.Tensor):
        masks = {
            name: regression_mask[..., index].to(device)
            for index, name in enumerate(BENCHMARK_REGRESSION_TARGETS)
        }
    if pooled.shape != (
        hidden.shape[0],
        objective.config.hidden_size,
    ):
        raise ValueError("pooled hidden must be [batch, hidden]")
    losses.update(objective.regression_losses(pooled, regression_targets, masks))

    token_targets = {
        name: batch[f"proxy_token_{name}"].to(device)
        for name in BENCHMARK_TOKEN_TARGETS
        if isinstance(batch.get(f"proxy_token_{name}"), torch.Tensor)
    }
    token_masks = {
        name: batch[f"proxy_token_mask_{name}"].to(device)
        for name in token_targets
        if isinstance(batch.get(f"proxy_token_mask_{name}"), torch.Tensor)
    }
    if token_targets:
        losses.update(objective.token_losses(hidden, token_targets, token_masks))

    test_output = batch.get("proxy_test_output_class")
    if isinstance(test_output, torch.Tensor):
        losses["L_test_output_class"] = objective.test_output_loss(
            pooled,
            test_output.to(device),
        )
    test_token = batch.get("proxy_test_output_token")
    if isinstance(test_token, torch.Tensor):
        test_loss = objective.test_output_token_loss(
            hidden,
            test_token.to(device),
            (
                batch["proxy_test_output_mask"].to(device)
                if isinstance(batch.get("proxy_test_output_mask"), torch.Tensor)
                else None
            ),
        )
        if test_loss is not None:
            losses["L_test_output_token"] = test_loss

    choice_targets = batch.get("proxy_multiple_choice")
    choice_embeddings = batch.get("proxy_choice_embeddings")
    if isinstance(choice_targets, torch.Tensor) and isinstance(
        choice_embeddings,
        torch.Tensor,
    ):
        active = choice_targets.to(device).ge(0)
        if active.any():
            losses["L_multiple_choice"] = objective.multiple_choice_loss(
                pooled[active],
                choice_embeddings.to(device)[active],
                choice_targets.to(device)[active],
            )

    retrieval_target = batch.get("proxy_retrieval_target")
    retrieval_candidates = batch.get("proxy_retrieval_candidates")
    if isinstance(retrieval_target, torch.Tensor) and isinstance(
        retrieval_candidates,
        torch.Tensor,
    ):
        losses["L_retrieval_contrastive"] = objective.retrieval_loss(
            pooled,
            retrieval_candidates.to(device),
            retrieval_target.to(device),
        )

    validity_target = batch.get("validity_targets")
    validity_mask = batch.get("validity_mask")
    if isinstance(validity_target, torch.Tensor) and isinstance(
        validity_mask,
        torch.Tensor,
    ):
        target = validity_target.to(device)
        mask = validity_mask.to(device)
        active = mask.any(dim=1)
        if active.any():
            pooled_target = (
                (target * mask.float()).sum(dim=1)
                / mask.float().sum(dim=1).clamp_min(1.0)
            )
            losses["L_instruction_constraints"] = (
                F.binary_cross_entropy_with_logits(
                    prediction.constraint_logits[..., : target.shape[-1]][active],
                    pooled_target[active],
                )
            )
    return losses


__all__ = ["proxy_losses"]
