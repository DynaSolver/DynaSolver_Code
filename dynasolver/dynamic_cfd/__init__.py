"""Training and evaluation helpers for prescribed-boundary dynamic CFD."""

from .training import rollout_trajectory, run_one_step_epoch, weighted_masked_mse

__all__ = ["rollout_trajectory", "run_one_step_epoch", "weighted_masked_mse"]
