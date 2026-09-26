"""Load a trained TemporalTransolver / DynaSolver checkpoint for eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from data_provider.dynamic_cfd import DynamicStateNormalizer
from models.TemporalTransolver import Model as TemporalModel
from scripts.eval_temporal_vs_persistence import make_model_args


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_temporal_model(
    run_dir: Path,
    device: torch.device,
    *,
    checkpoint: str = "checkpoint_best.pt",
) -> tuple[TemporalModel, dict, DynamicStateNormalizer, dict]:
    """Load model + config + normalizer + run_manifest from a training run dir."""
    run_dir = Path(run_dir)
    manifest = load_json(run_dir / "run_manifest.json")
    snap = run_dir / "config.snapshot.yaml"
    cfg_path = snap if snap.is_file() else Path(str(manifest.get("config") or ""))
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"config.snapshot.yaml / manifest config missing under {run_dir}"
        )
    cfg = load_yaml(cfg_path)
    model = TemporalModel(make_model_args(cfg["model"])).to(device)
    ckpt_path = run_dir / checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    normalizer = DynamicStateNormalizer.from_dict(load_json(run_dir / "normalizer.json"))
    return model, cfg, normalizer, manifest
