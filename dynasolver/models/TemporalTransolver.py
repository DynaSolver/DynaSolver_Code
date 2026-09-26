"""DynaSolver / TemporalTransolver (submission stub).

Public training/eval scripts import ``Model`` from this module. The full
network body is withheld; hyperparameters remain in ``configs/``.

Paper: architecture, training objective, and AR evaluation protocol.
Full source is withheld in this anonymous conference submission to protect unpublished technical details. The complete implementation will be released upon acceptance. See the paper for algorithms and equations.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

_NOT_IMPL = 'Full source is withheld in this anonymous conference submission to protect unpublished technical details. The complete implementation will be released upon acceptance. See the paper for algorithms and equations.'


class Model(nn.Module):
    """Interface-compatible stub for TemporalTransolver / DynaSolver."""

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.__name__ = "TemporalTransolver"
        # Record key hyper-parameters so configs remain inspectable.
        self.space_dim = int(getattr(args, "space_dim", 3))
        self.fun_dim = int(getattr(args, "fun_dim", 11))
        self.state_dim = int(getattr(args, "state_dim", 4))
        self.boundary_dim = int(getattr(args, "boundary_dim", 8))
        self.n_hidden = int(getattr(args, "n_hidden", 256))
        self.n_layers = int(getattr(args, "n_layers", 8))
        self.slice_num = int(getattr(args, "slice_num", 16))
        self.window_frames = int(getattr(args, "window_frames", 4))
        self.use_long_memory = bool(getattr(args, "use_long_memory", False))
        self.use_short_window = bool(getattr(args, "use_short_window", True))
        self.branch_mode = str(getattr(args, "branch_mode", "unified"))
        # Tiny placeholder param so ``parameters()`` / optim construction works.
        self._placeholder = nn.Parameter(torch.zeros(1))

    def init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError(_NOT_IMPL)

    def forward_window(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        state: torch.Tensor,
        boundary_feat: torch.Tensor,
        memory_prev: torch.Tensor | None = None,
        surface_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError(_NOT_IMPL)

    def forward(
        self,
        query_xyz: torch.Tensor,
        static_fx: torch.Tensor,
        state: torch.Tensor,
        boundary_feat: torch.Tensor,
        memory_prev: torch.Tensor | None = None,
        surface_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(_NOT_IMPL)


def _geopt_loadable_keys(model: Model) -> set[str]:
    return {"placeholder"}


def load_geopt_backbone(
    model: Model,
    checkpoint_path: str | Path,
    expected_sha256: str | None = None,
) -> dict:
    raise NotImplementedError(_NOT_IMPL)
