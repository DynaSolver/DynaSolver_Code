from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch


TENSOR_KEYS = (
    "query_xyz",
    "static_fx",
    "previous_state",
    "previous_mask",
    "boundary_feat",
    "target_state",
    "loss_mask",
    "loss_weight",
    "surface_xyz_normalized",
    "surface_normal_current",
    "surface_state_current",
    "surface_valid_mask",
)


def model_forward(model: torch.nn.Module, batch: dict[str, Any], previous_state: torch.Tensor | None = None) -> torch.Tensor:
    """Dispatch optional surface inputs while preserving the V0 four-argument API."""
    previous = batch["previous_state"] if previous_state is None else previous_state
    if getattr(model, "requires_surface_branch", False):
        required = ("surface_xyz_normalized", "surface_normal_current", "surface_state_current", "surface_valid_mask")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"surface model requires batch fields: {missing}")
        return model(batch["query_xyz"], batch["static_fx"], previous, batch["boundary_feat"], batch["surface_xyz_normalized"], batch["surface_normal_current"], batch["surface_state_current"], batch["surface_valid_mask"])
    return model(batch["query_xyz"], batch["static_fx"], previous, batch["boundary_feat"])


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in TENSOR_KEYS:
        if key in moved:
            moved[key] = moved[key].to(device, non_blocking=True)
    return moved


def weighted_masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Return point-weighted channel MSE, normalized per batch sample."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and target must have the same [B,N,C] shape; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if mask.shape != prediction.shape[:2] or weight.shape != mask.shape:
        raise ValueError("mask and weight must have shape [B,N] matching prediction")
    active_weight = weight.to(prediction.dtype) * mask.to(prediction.dtype)
    denominator = active_weight.sum(dim=1)
    if torch.any(denominator <= 0):
        bad = torch.nonzero(denominator <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"batch samples have no positive supervised weight: {bad}")
    point_mse = torch.mean(torch.square(prediction - target), dim=-1)
    return torch.mean(torch.sum(point_mse * active_weight, dim=1) / denominator)


def weighted_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have the same [B,N,C] shape")
    active_weight = (weight * mask.to(weight.dtype)).unsqueeze(-1)
    numerator = torch.sum(active_weight * torch.square(prediction - target), dim=(1, 2))
    denominator = torch.sum(active_weight * torch.square(target), dim=(1, 2))
    return torch.sqrt(numerator / torch.clamp(denominator, min=eps))


def channel_weighted_masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    channel_weights: Sequence[float],
) -> torch.Tensor:
    """Point-weighted MSE with per-channel weights; returns scalar mean over batch.

    Equivalent to ``weighted_masked_mse`` when ``channel_weights == [1,...,1]``.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and target must have the same [B,N,C] shape; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    n_ch = int(prediction.shape[-1])
    if len(channel_weights) != n_ch:
        raise ValueError(
            f"channel_weights length {len(channel_weights)} != channels {n_ch}"
        )
    if all(abs(float(w) - 1.0) < 1.0e-12 for w in channel_weights):
        return weighted_masked_mse(prediction, target, mask, weight)
    active_weight = weight.to(prediction.dtype) * mask.to(prediction.dtype)
    denominator = active_weight.sum(dim=1)
    if torch.any(denominator <= 0):
        bad = torch.nonzero(denominator <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"batch samples have no positive supervised weight: {bad}")
    sq = torch.square(prediction - target)  # [B,N,C]
    # Per-channel point-weighted MSE → [B,C]
    per_sample = torch.sum(sq * active_weight.unsqueeze(-1), dim=1) / denominator.unsqueeze(-1)
    w = torch.as_tensor(channel_weights, device=prediction.device, dtype=prediction.dtype)
    w_sum = w.sum().clamp_min(1.0e-12)
    return torch.mean(torch.sum(per_sample * w.view(1, -1), dim=-1) / w_sum)


def rel_vp_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """``(α·relL2(vel) + β·relL2(p)) / (α+β)`` — aligned with table V/P metrics."""
    if prediction.shape[-1] < 4:
        raise ValueError(f"rel_vp_loss expects >=4 channels, got {prediction.shape[-1]}")
    a = float(alpha)
    b = float(beta)
    if a < 0 or b < 0 or (a + b) <= 0:
        raise ValueError(f"rel_vp alpha/beta must be >=0 and sum>0, got {a}, {b}")
    l_v = weighted_relative_l2(
        prediction[..., :3], target[..., :3], mask, weight, eps=eps
    ).mean()
    l_p = weighted_relative_l2(
        prediction[..., 3:4], target[..., 3:4], mask, weight, eps=eps
    ).mean()
    return (a * l_v + b * l_p) / (a + b)


def supervision_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    *,
    loss_mode: str = "mse",
    channel_weights: Sequence[float] | None = None,
    rel_vp_alpha: float = 1.0,
    rel_vp_beta: float = 1.0,
) -> torch.Tensor:
    """Train/eval supervision used by TemporalTransolver CFD training."""
    mode = str(loss_mode or "mse").strip().lower()
    if mode == "mse":
        if channel_weights is None:
            return weighted_masked_mse(prediction, target, mask, weight)
        return channel_weighted_masked_mse(
            prediction, target, mask, weight, channel_weights
        )
    if mode in ("rel_vp", "relvp", "relative_vp"):
        return rel_vp_loss(
            prediction,
            target,
            mask,
            weight,
            alpha=rel_vp_alpha,
            beta=rel_vp_beta,
        )
    raise ValueError(f"unknown train loss_mode={loss_mode!r}; use mse|rel_vp")


def run_one_step_epoch(
    model: torch.nn.Module,
    batches: Iterable[dict[str, Any]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    persistence_sum = 0.0
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in batches:
            batch = _to_device(raw_batch, device)
            prediction = model_forward(model, batch)
            loss = weighted_masked_mse(
                prediction,
                batch["target_state"],
                batch["loss_mask"],
                batch["loss_weight"],
            )
            prev_for_persistence = batch["previous_state"]
            if prev_for_persistence.ndim == 4:
                prev_for_persistence = prev_for_persistence[:, -1]
            persistence = weighted_masked_mse(
                prev_for_persistence,
                batch["target_state"],
                batch["loss_mask"],
                batch["loss_weight"],
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            batch_size = int(prediction.shape[0])
            loss_sum += float(loss.detach()) * batch_size
            persistence_sum += float(persistence.detach()) * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise ValueError("one-step epoch received no samples")
    return {
        "loss": loss_sum / sample_count,
        "persistence_loss": persistence_sum / sample_count,
        "samples": float(sample_count),
    }


def rollout_trajectory(
    model: torch.nn.Module,
    dataset: Any,
    trajectory_id: str,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run a pure autoregressive rollout after the first ground-truth state."""
    indices = dataset.sample_indices_for_trajectory(trajectory_id)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        indices = indices[:max_steps]
    if not indices:
        raise ValueError(f"trajectory {trajectory_id!r} contains no transitions")

    model.eval()
    history = int(getattr(model, "history", 1) or 1)
    per_step_relative_l2: list[float] = []
    per_step_mse: list[float] = []
    channel_squared_error: np.ndarray | None = None
    channel_target_square: np.ndarray | None = None
    supervised_weight = 0.0
    current_state: torch.Tensor | None = None
    state_buffer: list[torch.Tensor] = []
    with torch.no_grad():
        for sample_index in indices:
            sample = _to_device(dataset[sample_index], device)
            previous_mask = sample["previous_mask"].unsqueeze(0)
            gt_previous = sample["previous_state"]
            if gt_previous.ndim == 3:
                # Dataset history mode: [H,N,C]
                gt_last = gt_previous[-1].unsqueeze(0)
            else:
                gt_last = gt_previous.unsqueeze(0)
            if current_state is None:
                current_state = gt_last
                if history > 1 and gt_previous.ndim == 3:
                    state_buffer = [gt_previous[h].unsqueeze(0) for h in range(gt_previous.shape[0])]
                else:
                    state_buffer = [current_state.clone() for _ in range(history)]
            else:
                current_state = current_state * previous_mask.unsqueeze(-1).to(current_state.dtype)
                if history > 1:
                    state_buffer[-1] = current_state
            target = sample["target_state"].unsqueeze(0)
            loss_mask = sample["loss_mask"].unsqueeze(0)
            loss_weight = sample["loss_weight"].unsqueeze(0)
            if history > 1:
                model_prev = torch.stack([s.squeeze(0) for s in state_buffer], dim=0).unsqueeze(0)
            else:
                model_prev = current_state
            rollout_batch = {
                key: (value.unsqueeze(0) if torch.is_tensor(value) and key != "previous_state" else value)
                for key, value in sample.items()
            }
            rollout_batch["previous_state"] = model_prev
            current_state = model_forward(model, rollout_batch)
            if history > 1:
                state_buffer = state_buffer[1:] + [current_state]
            mse = weighted_masked_mse(current_state, target, loss_mask, loss_weight)
            relative_l2 = weighted_relative_l2(current_state, target, loss_mask, loss_weight)
            per_step_mse.append(float(mse))
            per_step_relative_l2.append(float(relative_l2.item()))
            effective = (
                loss_weight * loss_mask.to(loss_weight.dtype)
            ).unsqueeze(-1)
            squared_error = torch.sum(
                effective * torch.square(current_state - target), dim=(0, 1)
            ).cpu().numpy()
            target_square = torch.sum(
                effective * torch.square(target), dim=(0, 1)
            ).cpu().numpy()
            if channel_squared_error is None:
                channel_squared_error = np.zeros_like(squared_error, dtype=np.float64)
                channel_target_square = np.zeros_like(target_square, dtype=np.float64)
            channel_squared_error += squared_error
            channel_target_square += target_square
            supervised_weight += float(effective.sum())

    assert channel_squared_error is not None and channel_target_square is not None
    channel_rmse = np.sqrt(channel_squared_error / supervised_weight)
    channel_relative_l2 = np.sqrt(
        channel_squared_error / np.maximum(channel_target_square, np.finfo(np.float64).tiny)
    )
    return {
        "trajectory_id": trajectory_id,
        "steps": len(indices),
        "mse_mean": float(np.mean(per_step_mse)),
        "relative_l2_mean": float(np.mean(per_step_relative_l2)),
        "relative_l2_final": per_step_relative_l2[-1],
        "channel_rmse": channel_rmse.tolist(),
        "channel_relative_l2": channel_relative_l2.tolist(),
        "per_step_mse": per_step_mse,
        "per_step_relative_l2": per_step_relative_l2,
    }
