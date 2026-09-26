"""Model registry for the submission package (DynaSolver only)."""
from __future__ import annotations
import models.TemporalTransolver as TemporalTransolver


def get_model(args):
    name = getattr(args, "model", "TemporalTransolver")
    if name not in {"TemporalTransolver", "DynaSolver"}:
        raise KeyError(
            f"submission package only exposes TemporalTransolver/DynaSolver, got {name!r}"
        )
    return TemporalTransolver.Model(args)
