from __future__ import annotations

import argparse
import os
import copy
import csv
import json
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "dynasolver"
for _p in (REPO_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_provider.dynamic_cfd import (
    SPLITS,
    DynamicStateNormalizer,
    load_trajectory_manifest,
    resolve_point_indices,
    validate_trajectory_hdf5,
)
from data_provider.temporal_cfd import (
    SameWindowLengthBatchSampler,
    TemporalCFDWindowDataset,
    advance_causal_window_tensors,
    causal_window_frame_ids,
    resolve_keyframe_timeline,
)
from layers.Temporal_Physics_Block import CrossStepHistCache
from dynamic_cfd.training import (
    supervision_loss,
    weighted_masked_mse,
    weighted_relative_l2,
)
from models.TemporalTransolver import Model, _geopt_loadable_keys, load_geopt_backbone
from scripts.eval_temporal_vs_persistence import load_trajectory_arrays
from utils.constants import GEOPT_CHECKPOINT_SHA256
from utils.paths import GEOPT_CHECKPOINT

# Keys that must match between CLI/resolved run and a resume checkpoint.
_RESUME_IDENTITY_KEYS = ("manifest", "seed", "points")
_RESUME_MODEL_KEYS = (
    "n_hidden",
    "n_heads",
    "n_layers",
    "slice_num",
    "window_frames",
    "use_long_memory",
    "long_memory_placement",
    "mid_long_after_layer",
    "use_short_window",
    "use_rope",
    "use_temporal_scale_gate",
    "use_boundary_feat",
    "use_hist_state_embed",
    "slice_mode",
    "state_dim",
    "fun_dim",
    "space_dim",
    "boundary_dim",
    "branch_mode",
    "share_cross_weights",
    "point_decoder_layers",
)
_RESUME_TRAIN_KEYS = (
    "batch_size",
    "lr",
    "weight_decay",
    "history_frames",
    "subset_seed",
    # early_stop_* / epochs intentionally omitted: schedule may change on resume
    "max_grad_norm",
    "hist_noise_std",
    "selection_metric",
    "validation_rollout_steps",
    "unroll_steps",
    "sample_stride",
    "ss_prob",
    "detach_unroll",
)
_CHANNEL_NAMES = ("u", "v", "w", "p")
_VALID_SELECTION_METRICS = frozenset(
    {"validation_loss", "validation_rollout_relative_l2"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Temporal Transolver on 16-wind CFD windows")
    parser.add_argument("--config", type=Path, required=True, help="temporal_*.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--points", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--disable-long-memory", action="store_true")
    parser.add_argument("--disable-short-window", action="store_true")
    parser.add_argument("--allow-unhashed-data", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from output-dir/checkpoint_last.pt (model, optimizer, history, experiment)",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="explicit checkpoint path (overrides --resume default last/best lookup)",
    )
    parser.add_argument(
        "--resume-allow-partial",
        action="store_true",
        help="allow resume from legacy checkpoint_best without optimizer (re-init AdamW)",
    )
    parser.add_argument(
        "--initialization",
        choices=("from_scratch", "geopt_pretrained", "from_checkpoint"),
        help="override training.initialization",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        help="GeoPT checkpoint for geopt_pretrained (default: paths.geopt_checkpoint)",
    )
    parser.add_argument(
        "--pretrained-sha256",
        default=GEOPT_CHECKPOINT_SHA256,
        help="expected SHA256 of GeoPT checkpoint (empty string to skip check)",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="weights-only init for initialization=from_checkpoint (overrides training.init_checkpoint)",
    )
    parser.add_argument(
        "--normalizer-path",
        type=Path,
        help="frozen normalizer JSON (overrides paths.normalizer; skips fit_from_manifest)",
    )
    return parser.parse_args()


def load_finetune_checkpoint_weights(model: Model, checkpoint_path: Path) -> dict[str, Any]:
    """Load model weights from a training checkpoint; ignore optimizer/history."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"init_checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(
            f"init_checkpoint must be a training checkpoint with a 'model' key: {checkpoint_path}"
        )
    load_result = model.load_state_dict(payload["model"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected finetune load result: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    return {
        "path": str(checkpoint_path),
        "source_epoch": payload.get("epoch"),
        "source_best_epoch": payload.get("best_epoch"),
        "n_tensors": len(payload["model"]),
    }

def parse_lr_mult_schedule(raw: object) -> list[tuple[int, float, str]]:
    """Parse [{until_epoch, mult, mode?}, ...] → (until_epoch, mult, mode).

    mode: ``hold`` (constant mult on the segment) or ``lerp`` (linear from previous
    segment's mult at prev_until to this mult at until_epoch). First segment is
    always treated as hold.
    """
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError("lr_*_mult_schedule must be a non-empty list when set")
    out: list[tuple[int, float, str]] = []
    prev = 0
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"lr mult schedule[{i}] must be a mapping")
        until = int(item.get("until_epoch", item.get("until")))
        mult = float(item.get("mult", item.get("lr_mult", 1.0)))
        mode = str(item.get("mode", "hold" if i == 0 else "lerp")).lower()
        if mode not in ("hold", "lerp"):
            raise ValueError(f"lr mult schedule[{i}].mode must be hold|lerp, got {mode!r}")
        if until < prev:
            raise ValueError("lr mult schedule until_epoch must be non-decreasing")
        if mult < 0:
            raise ValueError("lr mult must be >= 0")
        out.append((until, mult, mode))
        prev = until
    return out


def lr_mult_for_epoch(
    epoch: int,
    schedule: list[tuple[int, float, str]],
    default: float = 1.0,
) -> float:
    """Evaluate spatial/temporal LR multiplier for a 1-indexed epoch."""
    if not schedule:
        return float(default)
    prev_until = 0
    prev_mult = float(schedule[0][1])
    for i, (until, mult, mode) in enumerate(schedule):
        if epoch <= until:
            if i == 0 or mode == "hold":
                return float(mult)
            span = max(until - prev_until, 1)
            t = (epoch - prev_until) / float(span)
            t = min(max(t, 0.0), 1.0)
            return float(prev_mult + t * (mult - prev_mult))
        prev_until = until
        prev_mult = mult
    return float(schedule[-1][1])


def parse_replay_prob_schedule(raw: object) -> list[tuple[int, float, str]]:
    """Parse [{until_epoch, prob, mode?}, ...] — same hold/lerp semantics as LR mult."""
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError("replay_prob_schedule must be a non-empty list when set")
    normalized: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"replay_prob_schedule[{i}] must be a mapping")
        d = dict(item)
        if "prob" in d and "mult" not in d:
            d["mult"] = d["prob"]
        normalized.append(d)
    out = parse_lr_mult_schedule(normalized)
    for until, prob, _mode in out:
        if prob < 0.0 or prob > 1.0:
            raise ValueError(
                f"replay_prob_schedule until_epoch={until}: prob must be in [0, 1], got {prob}"
            )
    return out


def replay_prob_for_epoch(
    epoch: int,
    *,
    clip_replay: bool,
    replay_prob: float,
    schedule: list[tuple[int, float, str]],
) -> float:
    """Effective clip-replay probability for a 1-indexed epoch.

    With a schedule: evaluate hold/lerp keypoints (overrides the default ep1=0 rule).
    Without: ep1 → 0, ep2+ → ``replay_prob`` (legacy warm-start).
    """
    if not clip_replay:
        return 0.0
    if schedule:
        return float(lr_mult_for_epoch(epoch, schedule, default=replay_prob))
    return 0.0 if epoch <= 1 else float(replay_prob)


def split_spatial_temporal_parameters(
    model: Model,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """GeoPT-loadable tensors → spatial group; all other trainable → temporal group."""
    loadable = _geopt_loadable_keys(model)
    spatial: list[torch.nn.Parameter] = []
    temporal: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in loadable:
            spatial.append(param)
        else:
            temporal.append(param)
    if not temporal:
        raise RuntimeError("temporal param group is empty — check model modules")
    if not spatial:
        # from_scratch without GeoPT keys still ok: put everything in temporal
        return [], list(temporal)
    return spatial, temporal


def build_adamw_param_groups(
    model: Model,
    *,
    lr: float,
    weight_decay: float,
    spatial_mult: float = 1.0,
    temporal_mult: float = 1.0,
) -> torch.optim.AdamW:
    spatial, temporal = split_spatial_temporal_parameters(model)
    groups: list[dict[str, Any]] = []
    if spatial:
        groups.append(
            {
                "params": spatial,
                "lr": lr * float(spatial_mult),
                "weight_decay": weight_decay,
                "name": "spatial",
            }
        )
    groups.append(
        {
            "params": temporal if temporal else list(model.parameters()),
            "lr": lr * float(temporal_mult),
            "weight_decay": weight_decay,
            "name": "temporal",
        }
    )
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def apply_param_group_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    spatial_mult: float,
    temporal_mult: float,
) -> dict[str, float]:
    applied: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if name == "spatial":
            group["lr"] = base_lr * float(spatial_mult)
            applied["spatial"] = group["lr"]
        else:
            group["lr"] = base_lr * float(temporal_mult)
            applied["temporal"] = group["lr"]
    return applied

def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a mapping")

    def _expand(obj):
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        return obj

    return _expand(payload)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_history_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    # Union keys across rows: pushforward K>1 adds train_loss_step_last etc.
    # Using only history[0] keys crashes when the schema grows mid-run.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in history:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def make_model_args(cfg_model: dict, overrides: argparse.Namespace) -> Namespace:
    use_long = bool(cfg_model.get("use_long_memory", True)) and not overrides.disable_long_memory
    use_short = bool(cfg_model.get("use_short_window", True)) and not overrides.disable_short_window
    placement = str(
        cfg_model.get(
            "long_memory_placement",
            "per_layer" if use_long else "none",
        )
    )
    if overrides.disable_long_memory:
        placement = "none"
        use_long = False
    return Namespace(
        model="TemporalTransolver",
        fun_dim=int(cfg_model.get("fun_dim", 11)),
        space_dim=int(cfg_model.get("space_dim", 3)),
        state_dim=int(cfg_model["state_dim"]),
        boundary_dim=int(cfg_model.get("boundary_dim", 8)),
        n_hidden=int(cfg_model.get("n_hidden", 256)),
        n_heads=int(cfg_model.get("n_heads", 8)),
        n_layers=int(cfg_model.get("n_layers", 4)),
        mlp_ratio=int(cfg_model.get("mlp_ratio", 2)),
        out_dim=int(cfg_model["state_dim"]),
        slice_num=int(cfg_model.get("slice_num", 32)),
        dropout=float(cfg_model.get("dropout", 0.0)),
        act=str(cfg_model.get("act", "gelu")),
        geotype=str(cfg_model.get("geotype", "unstructured")),
        shapelist=None,
        checkpoint=False,
        unified_pos=False,
        window_frames=int(cfg_model.get("window_frames", 5)),
        use_long_memory=use_long,
        long_memory_placement=placement,
        mid_long_after_layer=int(cfg_model.get("mid_long_after_layer", 4)),
        use_short_window=use_short,
        use_rope=bool(cfg_model.get("use_rope", True)),
        short_scale_init=float(cfg_model.get("short_scale_init", 1.0e-3)),
        long_scale_init=float(cfg_model.get("long_scale_init", 1.0e-3)),
        use_temporal_scale_gate=bool(cfg_model.get("use_temporal_scale_gate", False)),
        use_boundary_feat=bool(cfg_model.get("use_boundary_feat", True)),
        use_hist_state_embed=bool(cfg_model.get("use_hist_state_embed", True)),
        slice_mode=str(cfg_model.get("slice_mode", "legacy")),
        use_flash_attn=bool(cfg_model.get("use_flash_attn", False)),
        branch_mode=str(cfg_model.get("branch_mode", "unified")),
        cross_vs_scale_init=float(cfg_model.get("cross_vs_scale_init", 0.0)),
        share_cross_weights=bool(cfg_model.get("share_cross_weights", False)),
        point_decoder_layers=int(cfg_model.get("point_decoder_layers", 0)),
    )


def build_experiment_snapshot(
    *,
    cfg: dict,
    config_path: Path,
    manifest: Path,
    output_dir: Path,
    seed: int,
    points: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    early_stop_min_epoch: int,
    max_grad_norm: float,
    history_frames: int,
    subset_seed: int,
    architecture: Namespace,
    train_cfg: dict,
    pin_memory: bool,
) -> dict[str, Any]:
    """Full resolved experiment identity for resume / audit (not just a yaml path)."""
    return {
        "config_path": str(config_path.resolve()),
        "config": cfg,
        "resolved": {
            "manifest": str(manifest.resolve()),
            "output_dir": str(output_dir.resolve()),
            "seed": int(seed),
            "points": int(points),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(train_cfg.get("weight_decay", 1.0e-5)),
            "early_stop_patience": int(patience),
            "early_stop_min_epoch": int(early_stop_min_epoch),
            "max_grad_norm": float(max_grad_norm),
            "history_frames": int(history_frames),
            "subset_seed": int(subset_seed),
            "initialization": train_cfg.get("initialization"),
            "lr_spatial_mult_schedule": train_cfg.get("lr_spatial_mult_schedule"),
            "lr_temporal_mult_schedule": train_cfg.get("lr_temporal_mult_schedule"),
            "detach_memory_each_step": bool(train_cfg.get("detach_memory_each_step", True)),
            "hist_noise_std": float(train_cfg.get("hist_noise_std", 0.0)),
            "unroll_steps": int(train_cfg.get("unroll_steps", 1)),
            "sample_stride": int(train_cfg.get("sample_stride", 1)),
            "ss_prob": float(train_cfg.get("ss_prob", 0.0)),
            "detach_unroll": bool(train_cfg.get("detach_unroll", True)),
            "validation_rollout_steps": int(train_cfg.get("validation_rollout_steps", 0)),
            "validation_rollout_max_trajs": int(
                train_cfg.get("validation_rollout_max_trajs", 0)
            ),
            "validation_rollout_every": int(train_cfg.get("validation_rollout_every", 1)),
            "selection_metric": str(train_cfg.get("selection_metric", "validation_loss")),
            "model": dict(vars(architecture)),
            "dataloader": {
                "batch_size": int(batch_size),
                "num_workers": int(train_cfg.get("num_workers", 0)),
                "prefetch_factor": int(train_cfg.get("prefetch_factor", 2)),
                "pin_memory": bool(pin_memory),
            },
        },
    }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_rng_byte_tensor(value: Any) -> torch.Tensor:
    """torch.set_rng_state requires a CPU ByteTensor; ckpt reload often yields uint8 Tensor."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu").contiguous()
    else:
        tensor = torch.as_tensor(value)
        tensor = tensor.detach().to("cpu").contiguous()
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(dtype=torch.uint8)
    # PyTorch still type-checks ByteTensor specifically on some builds.
    return tensor if type(tensor) is torch.ByteTensor else tensor.byte()


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_rng_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = state["cuda"]
        if isinstance(cuda_states, (list, tuple)):
            torch.cuda.set_rng_state_all([_as_rng_byte_tensor(s) for s in cuda_states])
        else:
            torch.cuda.set_rng_state(_as_rng_byte_tensor(cuda_states))


def build_training_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    normalizer: Any,
    run_manifest: dict[str, Any],
    epoch: int,
    best_validation_loss: float,
    best_epoch: int,
    history: list[dict],
    experiment: dict[str, Any] | None = None,
    include_optimizer: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "model": model.state_dict(),
        "normalizer": normalizer.to_dict() if hasattr(normalizer, "to_dict") else normalizer,
        "run_manifest": run_manifest,
        "experiment": experiment or run_manifest.get("experiment"),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "validation_loss": float(best_validation_loss),
        "history": list(history),
        "rng_state": capture_rng_state(),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def save_training_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def best_metric_from_history(
    history: list[dict], metric: str
) -> tuple[float, int] | None:
    """Return (best_value, epoch) for ``metric``, or None if never recorded."""
    scored: list[tuple[float, int]] = []
    for row in history:
        if metric not in row or row[metric] is None:
            continue
        scored.append((float(row[metric]), int(row["epoch"])))
    if not scored:
        return None
    value, epoch = min(scored, key=lambda item: item[0])
    return value, epoch


def _as_resolved_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve())


def validate_resume_compatibility(
    payload: dict[str, Any],
    *,
    manifest: Path,
    seed: int,
    points: int,
    experiment: dict[str, Any] | None = None,
) -> None:
    """Reject resume when identity / architecture / training knobs disagree."""
    run_manifest = payload.get("run_manifest") or {}
    ckpt_experiment = payload.get("experiment") or run_manifest.get("experiment") or {}
    ckpt_resolved = dict(ckpt_experiment.get("resolved") or {})
    if not ckpt_resolved:
        # Legacy best-only checkpoints: fall back to flat run_manifest fields.
        ckpt_resolved = {
            "manifest": run_manifest.get("manifest"),
            "seed": run_manifest.get("seed"),
            "points": run_manifest.get("points"),
            "model": run_manifest.get("model") or {},
            "batch_size": (run_manifest.get("dataloader") or {}).get("batch_size"),
            "lr": (run_manifest.get("training") or {}).get("lr"),
            "weight_decay": (run_manifest.get("training") or {}).get("weight_decay"),
            "history_frames": (run_manifest.get("training") or {}).get("history_frames"),
            "subset_seed": (run_manifest.get("training") or {}).get("subset_seed"),
            "early_stop_patience": (run_manifest.get("training") or {}).get("early_stop_patience"),
            "max_grad_norm": (run_manifest.get("training") or {}).get("max_grad_norm"),
        }

    expected = {
        "manifest": _as_resolved_path(manifest),
        "seed": int(seed),
        "points": int(points),
    }
    for key in _RESUME_IDENTITY_KEYS:
        got = ckpt_resolved.get(key, run_manifest.get(key))
        if got is None:
            raise ValueError(f"resume checkpoint missing identity field {key!r}")
        if key == "manifest":
            if _as_resolved_path(got) != expected[key]:
                raise ValueError(
                    f"resume manifest mismatch: checkpoint={got!r} current={expected[key]!r}"
                )
        elif int(got) != int(expected[key]):
            raise ValueError(
                f"resume {key} mismatch: checkpoint={got!r} current={expected[key]!r}"
            )

    if experiment is None:
        return

    cur_resolved = experiment.get("resolved") or {}
    ckpt_model = dict(ckpt_resolved.get("model") or {})
    cur_model = dict(cur_resolved.get("model") or {})
    for key in _RESUME_MODEL_KEYS:
        if key not in ckpt_model or key not in cur_model:
            continue
        if ckpt_model[key] != cur_model[key]:
            raise ValueError(
                f"resume model.{key} mismatch: checkpoint={ckpt_model[key]!r} "
                f"current={cur_model[key]!r}"
            )
    for key in _RESUME_TRAIN_KEYS:
        if key not in ckpt_resolved or key not in cur_resolved:
            continue
        left, right = ckpt_resolved[key], cur_resolved[key]
        # Legacy best-only ckpts often omit training knobs → skip missing fields.
        if left is None or right is None:
            continue
        if isinstance(left, (float, int)) or isinstance(right, (float, int)):
            if abs(float(left) - float(right)) > 1e-12:
                raise ValueError(
                    f"resume resolved.{key} mismatch: checkpoint={left!r} current={right!r}"
                )
        elif left != right:
            raise ValueError(
                f"resume resolved.{key} mismatch: checkpoint={left!r} current={right!r}"
            )


def _normalizer_from_payload(payload_norm: Any) -> DynamicStateNormalizer:
    if isinstance(payload_norm, DynamicStateNormalizer):
        return payload_norm
    if not isinstance(payload_norm, dict):
        raise TypeError(f"normalizer payload must be a dict, got {type(payload_norm)}")
    return DynamicStateNormalizer.from_dict(payload_norm)


def load_training_resume(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    manifest: Path,
    seed: int,
    points: int,
    experiment: dict[str, Any] | None = None,
    allow_partial: bool = False,
) -> tuple[DynamicStateNormalizer, list[dict], float, int, int]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"invalid resume checkpoint: {checkpoint_path}")
    validate_resume_compatibility(
        payload,
        manifest=manifest,
        seed=seed,
        points=points,
        experiment=experiment,
    )
    model.load_state_dict(payload["model"])
    has_optimizer = "optimizer" in payload and payload["optimizer"] is not None
    if has_optimizer:
        optimizer.load_state_dict(payload["optimizer"])
    elif not allow_partial:
        raise ValueError(
            f"checkpoint missing optimizer ({checkpoint_path}); "
            "re-run with a v2 checkpoint_last or pass --resume-allow-partial"
        )

    history = list(payload.get("history") or [])
    if not history:
        history_path = checkpoint_path.parent / "history.json"
        if history_path.is_file():
            history = list(json.loads(history_path.read_text(encoding="utf-8")))
    if not history:
        raise ValueError(f"resume checkpoint has empty history: {checkpoint_path}")

    # Prefer selection metric over validation_loss. Loading best_validation_loss into
    # ``best`` when selection_metric is rollout (~0.3) makes every resume epoch look
    # worse than a tiny val-loss (~0.007), so checkpoint_best never updates.
    selection_metric = (
        payload.get("best_selection_metric")
        or payload.get("selection_metric")
        or next(
            (
                str(row.get("selection_metric"))
                for row in reversed(history)
                if row.get("selection_metric")
            ),
            None,
        )
        or "validation_loss"
    )
    scored: list[tuple[int, float]] = []
    for row in history:
        if selection_metric not in row or row[selection_metric] is None:
            continue
        scored.append((int(row["epoch"]), float(row[selection_metric])))
    if scored:
        best_epoch, best = min(scored, key=lambda item: item[1])
    elif payload.get("best_selection_value") is not None:
        best = float(payload["best_selection_value"])
        best_epoch = int(payload.get("best_epoch", payload.get("epoch", history[-1]["epoch"])))
    else:
        best = float(
            payload.get("best_validation_loss", payload.get("validation_loss", float("inf")))
        )
        best_epoch = int(payload.get("best_epoch", payload.get("epoch", history[-1]["epoch"])))
    start_epoch = int(history[-1]["epoch"]) + 1
    normalizer = _normalizer_from_payload(payload["normalizer"])
    restore_rng_state(payload.get("rng_state"))
    return normalizer, history, best, best_epoch, start_epoch


def resolve_resume_checkpoint(output_dir: Path, args: argparse.Namespace) -> Path | None:
    if args.resume_checkpoint is not None:
        path = Path(args.resume_checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--resume-checkpoint not found: {path}")
        return path
    if not args.resume:
        return None
    last_path = output_dir / "checkpoint_last.pt"
    best_path = output_dir / "checkpoint_best.pt"
    if last_path.is_file():
        return last_path
    if best_path.is_file():
        return best_path
    raise FileNotFoundError(
        f"--resume requested but missing {last_path} and {best_path}"
    )


def batch_to_device(
    batch: dict,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    out = {}
    keys = (
        "query_xyz",
        "static_fx",
        "state",
        "boundary_feat",
        "target_state",
        "loss_weight",
        "target_states",
        "future_static_fx",
        "future_boundary_feat",
        "future_loss_weight",
        "surface_mask",
    )
    for key in keys:
        if key in batch:
            dtype = torch.bool if key == "surface_mask" else torch.float32
            out[key] = batch[key].to(device=device, dtype=dtype, non_blocking=non_blocking)
    return out


def scheduled_sampling_prob(
    epoch: int,
    *,
    ss_prob: float,
    ss_warmup_epochs: int,
    phase_epoch: int | None = None,
) -> float:
    """Linear ramp from 0 → ss_prob over ss_warmup_epochs (then hold).

    If ``phase_epoch`` is set (1-based index within closed-loop phase), ramp
    uses that instead of the global epoch — so a late finetune phase can ramp
    independently of the preceding causal epochs.
    """
    ss_prob = float(ss_prob)
    if ss_prob <= 0:
        return 0.0
    warm = max(int(ss_warmup_epochs), 0)
    step = int(epoch if phase_epoch is None else phase_epoch)
    if warm <= 0:
        return ss_prob
    return ss_prob * min(1.0, float(step) / float(warm))


def parse_pushforward_k_schedule(raw: object) -> list[tuple[int, int]]:
    """Parse [{until_epoch, k_max}, ...] → sorted (until_epoch, k_max) list."""
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError("pushforward_k_schedule must be a non-empty list")
    out: list[tuple[int, int]] = []
    prev = 0
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"pushforward_k_schedule[{i}] must be a mapping")
        until = int(item.get("until_epoch", item.get("until")))
        k_max = int(item.get("k_max", item.get("k")))
        if until <= prev:
            raise ValueError(
                f"pushforward_k_schedule until_epoch must be increasing "
                f"(got {until} after {prev})"
            )
        if k_max < 0:
            raise ValueError("k_max must be >= 0")
        out.append((until, k_max))
        prev = until
    return out


def k_max_for_epoch(epoch: int, schedule: list[tuple[int, int]]) -> int:
    """Return schedule k_max for 1-based epoch."""
    if not schedule:
        raise ValueError("empty pushforward schedule")
    for until, k_max in schedule:
        if int(epoch) <= until:
            return int(k_max)
    return int(schedule[-1][1])


def unroll_steps_from_k_max(k_max: int) -> int:
    """k_max=0 → one-step teacher-forced; k_max>=1 → that many pushforward steps."""
    return max(int(k_max), 1)


def _dataloader_worker_init(_worker_id: int) -> None:
    """Drop inherited HDF5 handles after fork; each worker reopens lazily."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


@torch.no_grad()
def build_clip_replay_state(
    replay_model: Model,
    state: torch.Tensor,
    static: torch.Tensor,
    boundary: torch.Tensor,
    query: torch.Tensor,
    *,
    detach_memory: bool,
    parallel_causal_tf: bool,
    surface_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Clip-local lagged AR hist: GT_0 → Ŷ1 → … → Ŷ_{W-1} (detached).

    Starts from the clip's GT initial frame and rolls ``W-1`` steps with a frozen
    ``replay_model`` so training hist matches short closed-loop error structure.
    """
    replay_model.eval()
    batch, window, _, _ = state.shape
    if window < 2:
        return state.detach()
    replay_state = state.clone()
    device = state.device
    dtype = state.dtype
    for t in range(1, window):
        prefix_len = t
        state_p = replay_state[:, :prefix_len]
        static_p = static[:, :prefix_len]
        boundary_p = boundary[:, :prefix_len]
        if query.shape[1] == window:
            query_p = query[:, :prefix_len]
        else:
            query_p = query[:, :1].expand(-1, prefix_len, -1, -1).contiguous()
        surf_p = None
        if surface_mask is not None:
            surf_p = (
                surface_mask[:, :prefix_len]
                if surface_mask.ndim >= 3 and surface_mask.shape[1] == window
                else surface_mask
            )
        memory = replay_model.init_memory(batch, device, dtype)
        if detach_memory:
            memory = memory.detach()
        out = replay_model.forward_window(
            query_p,
            static_p,
            state_p,
            boundary_p,
            memory_prev=memory,
            parallel_causal_tf=parallel_causal_tf,
            surface_mask=surf_p,
        )
        replay_state[:, t] = out["prediction"]
    return replay_state.detach()


def sync_clip_replay_model(replay_model: Model | None, model: Model) -> Model:
    """Refresh lagged replay weights from the live trainer (epoch-end snapshot)."""
    if replay_model is None:
        replay_model = copy.deepcopy(model)
    else:
        replay_model.load_state_dict(model.state_dict())
    replay_model.eval()
    for param in replay_model.parameters():
        param.requires_grad_(False)
    return replay_model


def trainable_chunk_from_mode(mode: str, *, window_frames: int) -> int | None:
    """Map temporal_bptt_mode → trainable_chunk (None = legacy full-window grads)."""
    key = str(mode or "legacy").strip().lower()
    if key in ("legacy", "full", "recompute"):
        return None
    if key in ("plan_a", "plana", "a"):
        return 1
    if key in ("plan_b", "planb", "b", "xl"):
        return 2
    raise ValueError(
        f"temporal_bptt_mode must be plan_a|plan_b|legacy, got {mode!r}"
    )


def build_dataloader(
    dataset: TemporalCFDWindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    train_cfg: dict,
    device: torch.device,
    seed: int,
) -> DataLoader:
    """Honor yaml ``num_workers`` / ``pin_memory`` / ``prefetch_factor`` (same as baseline).

    Uses same-window-length batching so variable-length no-pad prefixes can stack.
    """
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", False)) and device.type == "cuda"
    batch_sampler = SameWindowLengthBatchSampler(
        dataset,
        batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )
    kwargs: dict = {
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = True
        kwargs["worker_init_fn"] = _dataloader_worker_init
    return DataLoader(dataset, **kwargs)


# Alias used by benchmark_temporal_training_step.py
def _build_dataloader(
    dataset: TemporalCFDWindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    train_cfg = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
    }
    device = torch.device("cuda" if pin_memory and torch.cuda.is_available() else "cpu")
    return build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        train_cfg=train_cfg,
        device=device,
        seed=0,
    )


def _channel_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Per-channel point-weighted MSE, shape [C]."""
    active = weight.to(prediction.dtype) * mask.to(prediction.dtype)
    denom = active.sum(dim=1).clamp_min(1.0e-12)
    sq = torch.square(prediction - target)
    # [B,N,C] -> weight over N -> mean over B
    per_sample = torch.sum(sq * active.unsqueeze(-1), dim=1) / denom.unsqueeze(-1)
    return per_sample.mean(dim=0)


def _accumulate_metric_dict(
    totals: dict[str, float],
    counts: dict[str, float],
    metrics: dict[str, float],
    batch_weight: float,
) -> None:
    w = max(batch_weight, 1.0)
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value) * w
        counts[key] = counts.get(key, 0.0) + w


def _finalize_metric_dict(
    totals: dict[str, float], counts: dict[str, float]
) -> dict[str, float]:
    return {k: totals[k] / max(counts.get(k, 1.0), 1.0) for k in totals}


def _phase_timing_start(device: torch.device) -> dict[str, Any]:
    """Wall + CUDA event bookkeeping for one train/eval/roll phase."""
    state: dict[str, Any] = {"wall0": time.perf_counter(), "device": device}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        state["cuda_start"] = start
        state["cuda_end"] = end
    return state


def _phase_timing_finish(state: dict[str, Any]) -> dict[str, float]:
    device: torch.device = state["device"]
    wall_s = float(time.perf_counter() - state["wall0"])
    if device.type == "cuda":
        end: torch.cuda.Event = state["cuda_end"]
        start: torch.cuda.Event = state["cuda_start"]
        end.record()
        torch.cuda.synchronize(device)
        cuda_ms = float(start.elapsed_time(end))
        peak_alloc_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
        peak_reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024.0**2))
    else:
        cuda_ms = float("nan")
        peak_alloc_mb = float("nan")
        peak_reserved_mb = float("nan")
    return {
        "wall_s": wall_s,
        "cuda_ms": cuda_ms,
        "peak_alloc_mb": peak_alloc_mb,
        "peak_reserved_mb": peak_reserved_mb,
    }


ROLLOUT_REPORT_DEPTHS: tuple[int, ...] = (1, 4, 10, 25, 50, 100)


def run_epoch(
    model: Model,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float = 1.0,
    detach_memory: bool = True,
    pin_memory: bool = False,
    hist_noise_std: float = 0.0,
    unroll_steps: int = 1,
    ss_prob: float = 0.0,
    detach_unroll: bool = True,
    pushforward_last_only: bool = False,
    ar_stream: bool = False,
    mixed_clean_pushforward: bool = False,
    pf_loss_weight: float = 0.5,
    start_singleton: bool = False,
    mixed_parallel_noise: bool = False,
    parallel_noise_eps: float = 0.0,
    parallel_noise_a: float = 0.0,
    clip_replay: bool = False,
    replay_prob: float = 0.0,
    replay_model: Model | None = None,
    clip_replay_horizon: int | None = None,
    trainable_chunk: int | None = None,
    history_frames: int | None = None,
    parallel_causal_tf: bool = False,
    position_loss_weights: Sequence[float] | None = None,
    train_loss_mode: str = "mse",
    channel_loss_weights: Sequence[float] | None = None,
    rel_vp_alpha: float = 1.0,
    rel_vp_beta: float = 1.0,
) -> dict[str, float]:
    """Teacher-forced and/or short AR-unrolled window training.

    unroll_steps>1 feeds predictions back into the state window (push-forward
    when detach_unroll=True) so training sees non-GT history.

    If pushforward_last_only=True, intermediate steps run under no_grad and only
    the final step contributes to the backward (cheap pushforward).

    If mixed_clean_pushforward=True (train only), also supervise a clean one-step
    TF branch on the original GT window and optimize
    ``L_clean + pf_loss_weight * L_pf`` (PF branch uses last-only semantics).

    If start_singleton=True, the PF/unroll branch restarts from the last GT
    frame only and grows/slides up to ``history_frames+1``. With
    mixed_clean_pushforward, L_clean still uses the full GT causal window
    before that slice.

    If mixed_parallel_noise=True with parallel_causal_tf (train only), run a
    clean parallel-TF branch plus a noisy-input parallel-TF branch with
    schedule ``σ=[0,ε,ε+a,ε+2a,…]`` on input states only; noisy branch uses
    Plan-A hist detach (``trainable_chunk=1``). Optimize
    ``L_clean + pf_loss_weight * L_noisy``.

    If clip_replay=True with parallel_causal_tf (train only), sample-level mix
    clean GT windows vs lagged clip-local replay hist (GT_0→Ŷ1→…→Ŷ_{W-1} from
    ``replay_model`` under eval/no_grad/detach). Both supervise the same GT
    next-state targets; one forward+backward per sample. ``replay_prob`` is the
    per-sample replay chance (epoch-1 callers should pass 0).

    If ``clip_replay_horizon`` > model ``window_frames`` (e.g. W=4, H=5), train
    clips are length H: teacher rolls GT0→Ŷ1→…→Ŷ_{H-1}; student Pass A on
    ``[:,0:W]`` (all W losses) and Pass B on ``[:,1:W+1]`` (last-only L_H);
    ``L=mean(L_1..L_H)``. GT branch uses the same two passes on GT hist.
    Validation with T=W stays single-pass.

    If ar_stream=True (train only), each AR step does its own loss.backward() +
    optimizer.step(), then writes pred.detach() into the state window — one
    supervise/update per time step with closed-loop inputs (not GT hist).

    ``trainable_chunk``: Plan A=1 / Plan B=2 / None=legacy full-window grads.

    ``parallel_causal_tf``: one forward over a fixed W-clip producing all-position
    residuals; optional ``position_loss_weights`` (default uniform).

    ``train_loss_mode``: ``mse`` (optional ``channel_loss_weights``) or ``rel_vp``
    (``rel_vp_alpha`` / ``rel_vp_beta`` on joint-velocity / pressure relative L2).
    """
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    counts: dict[str, float] = {}
    unroll_steps = max(int(unroll_steps), 1)
    pf_loss_weight = float(pf_loss_weight)

    def _sup(
        pred: torch.Tensor, tgt: torch.Tensor, mk: torch.Tensor, wt: torch.Tensor
    ) -> torch.Tensor:
        return supervision_loss(
            pred,
            tgt,
            mk,
            wt,
            loss_mode=train_loss_mode,
            channel_weights=channel_loss_weights,
            rel_vp_alpha=rel_vp_alpha,
            rel_vp_beta=rel_vp_beta,
        )
    # Stream AR: always closed-loop writeback; never last-only no_grad path.
    if ar_stream:
        if not train:
            raise ValueError("ar_stream is only valid during training epochs")
        if optimizer is None:
            raise ValueError("ar_stream requires an optimizer")
        if unroll_steps < 2:
            raise ValueError("ar_stream requires unroll_steps>=2 (need pred writeback)")
        if mixed_clean_pushforward:
            raise ValueError("ar_stream cannot combine with mixed_clean_pushforward")
        pushforward_last_only = False
        ss_prob = 1.0
        detach_unroll = True
    if mixed_clean_pushforward:
        if not train:
            # Validation callers keep unroll_steps=1; ignore the flag.
            mixed_clean_pushforward = False
        else:
            if unroll_steps < 2:
                raise ValueError("mixed_clean_pushforward requires unroll_steps>=2")
            if ar_stream:
                raise ValueError("mixed_clean_pushforward cannot combine with ar_stream")
            # PF branch is always last-only; clean branch is separate.
            pushforward_last_only = True
            ss_prob = 1.0
            detach_unroll = True
    max_window = int(
        history_frames + 1
        if history_frames is not None
        else getattr(model, "window_frames", 4)
    )
    # Validation must not retain autograd graphs across batches.
    # parallel_causal_tf ~47GiB/graph at B=2; a second batch OOMs ~80GB.
    _prev_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(bool(train))
    try:
        for batch in loader:
            tensors = batch_to_device(batch, device, non_blocking=pin_memory)
            state = tensors["state"]
            static = tensors["static_fx"]
            boundary = tensors["boundary_feat"]
            query = tensors["query_xyz"]
            if "target_states" in tensors:
                targets = tensors["target_states"]
                future_static = tensors["future_static_fx"]
                future_boundary = tensors["future_boundary_feat"]
                future_weight = tensors["future_loss_weight"]
            else:
                targets = tensors["target_state"].unsqueeze(1)
                future_static = None
                future_boundary = None
                future_weight = tensors["loss_weight"].unsqueeze(1)

            if targets.shape[1] < unroll_steps:
                raise RuntimeError(
                    f"batch target horizon {targets.shape[1]} < unroll_steps {unroll_steps}"
                )

            if train and hist_noise_std > 0.0 and state.shape[1] > 1 and not mixed_parallel_noise and not clip_replay:
                state = state.clone()
                state[:, :-1] = state[:, :-1] + hist_noise_std * torch.randn_like(state[:, :-1])

            if parallel_causal_tf:
                if "target_states" not in tensors:
                    raise RuntimeError("parallel_causal_tf requires target_states")
                clip_t = int(state.shape[1])
                if targets.shape[1] != clip_t:
                    raise RuntimeError(
                        f"parallel_causal_tf expects targets T={clip_t}, got {targets.shape[1]}"
                    )
                model_w = int(getattr(model, "window_frames", clip_t))
                horizon_cfg = (
                    int(clip_replay_horizon) if clip_replay_horizon is not None else None
                )
                use_slide = bool(
                    horizon_cfg is not None
                    and horizon_cfg > model_w
                    and clip_t == horizon_cfg
                )
                if use_slide and clip_t != model_w + 1:
                    raise RuntimeError(
                        "clip_replay_horizon slide currently supports H=W+1 only "
                        f"(got H={clip_t}, W={model_w})"
                    )
                w = model_w if use_slide else clip_t
                if not use_slide and clip_t > model_w:
                    raise RuntimeError(
                        f"parallel clip T={clip_t} exceeds model.window_frames={model_w}"
                    )
                if position_loss_weights is None:
                    lambdas = [1.0] * w
                else:
                    lambdas = [float(x) for x in position_loss_weights]
                    if len(lambdas) != w:
                        raise ValueError(
                            f"position_loss_weights length {len(lambdas)} != window {w}"
                        )
                lam_sum = sum(lambdas)
                if lam_sum <= 0:
                    raise ValueError("position_loss_weights must sum to > 0")

                surf_full = tensors.get("surface_mask")

                def _slice_surf(
                    start: int, length: int
                ) -> torch.Tensor | None:
                    if surf_full is None:
                        return None
                    if surf_full.ndim >= 3 and int(surf_full.shape[1]) == clip_t:
                        return surf_full[:, start : start + length]
                    return surf_full

                def _parallel_forward(
                    state_in: torch.Tensor,
                    static_in: torch.Tensor,
                    boundary_in: torch.Tensor,
                    query_in: torch.Tensor,
                    surf_in: torch.Tensor | None,
                    *,
                    chunk: int | None,
                ) -> torch.Tensor:
                    memory_b = model.init_memory(state_in.shape[0], device, state_in.dtype)
                    if detach_memory:
                        memory_b = memory_b.detach()
                    t_b = int(state_in.shape[1])
                    if int(query_in.shape[1]) != t_b:
                        query_b = query_in[:, :1].expand(-1, t_b, -1, -1).contiguous()
                    else:
                        query_b = query_in
                    result_b = model.forward_window(
                        query_b,
                        static_in,
                        state_in,
                        boundary_in,
                        memory_prev=memory_b,
                        trainable_chunk=chunk,
                        parallel_causal_tf=True,
                        surface_mask=surf_in,
                    )
                    return result_b["predictions"]

                def _parallel_branch_loss(
                    state_in: torch.Tensor,
                    *,
                    chunk: int | None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    preds_b = _parallel_forward(
                        state_in,
                        static,
                        boundary,
                        query,
                        surf_full,
                        chunk=chunk,
                    )
                    step_losses_b = []
                    for k in range(w):
                        wt = future_weight[:, k]
                        mk = (wt > 0).to(dtype=wt.dtype)
                        step_losses_b.append(
                            lambdas[k]
                            * _sup(preds_b[:, k], targets[:, k], mk, wt)
                        )
                    return torch.stack(step_losses_b).sum() / lam_sum, preds_b

                if train:
                    optimizer.zero_grad(set_to_none=True)

                use_mixed_noise = bool(train and mixed_parallel_noise)
                use_clip_replay = bool(train and clip_replay and replay_prob > 0.0)
                if use_clip_replay and use_mixed_noise:
                    raise RuntimeError("clip_replay cannot combine with mixed_parallel_noise")
                if use_slide and use_mixed_noise:
                    raise RuntimeError(
                        "clip_replay_horizon slide cannot combine with mixed_parallel_noise"
                    )
                if use_clip_replay and replay_model is None:
                    raise RuntimeError("clip_replay requires a lagged replay_model")

                state_train = state
                replay_frac = 0.0
                if use_clip_replay:
                    replay_state = build_clip_replay_state(
                        replay_model,
                        state,
                        static,
                        boundary,
                        query,
                        detach_memory=detach_memory,
                        parallel_causal_tf=True,
                        surface_mask=surf_full,
                    )
                    batch_size = int(state.shape[0])
                    use_replay = (
                        torch.rand(batch_size, device=state.device) < float(replay_prob)
                    )
                    replay_frac = float(use_replay.float().mean().item())
                    mask = use_replay.view(batch_size, 1, 1, 1)
                    state_train = torch.where(mask, replay_state, state)

                loss_l5: torch.Tensor | None = None
                if use_slide:
                    # Pass A: [0:W] → L1..LW ; Pass B: [1:W+1] → last-only L_{W+1}.
                    preds_a = _parallel_forward(
                        state_train[:, :w],
                        static[:, :w],
                        boundary[:, :w],
                        query[:, :w] if int(query.shape[1]) == clip_t else query,
                        _slice_surf(0, w),
                        chunk=None,
                    )
                    step_losses_a = []
                    for k in range(w):
                        wt = future_weight[:, k]
                        mk = (wt > 0).to(dtype=wt.dtype)
                        step_losses_a.append(
                            _sup(preds_a[:, k], targets[:, k], mk, wt)
                        )
                    preds_b = _parallel_forward(
                        state_train[:, 1 : w + 1],
                        static[:, 1 : w + 1],
                        boundary[:, 1 : w + 1],
                        query[:, 1 : w + 1] if int(query.shape[1]) == clip_t else query,
                        _slice_surf(1, w),
                        chunk=None,
                    )
                    wt5 = future_weight[:, w]
                    mk5 = (wt5 > 0).to(dtype=wt5.dtype)
                    loss_l5 = _sup(
                        preds_b[:, -1], targets[:, w], mk5, wt5
                    )
                    loss_clean_p = (torch.stack(step_losses_a).sum() + loss_l5) / float(
                        w + 1
                    )
                    preds = preds_a
                    loss_noisy_p = None
                    loss = loss_clean_p
                else:
                    loss_clean_p, preds = _parallel_branch_loss(state_train, chunk=None)
                    loss_noisy_p = None
                    if use_mixed_noise:
                        if parallel_noise_eps < 0 or parallel_noise_a < 0:
                            raise ValueError("parallel_noise_eps/a must be >= 0")
                        # σ_t = 0, ε, ε+a, ε+2a, ... for t=0..W-1
                        sigmas = [
                            0.0
                            if t == 0
                            else float(parallel_noise_eps + (t - 1) * parallel_noise_a)
                            for t in range(w)
                        ]
                        state_noisy = state.clone()
                        for t, sigma in enumerate(sigmas):
                            if sigma > 0.0:
                                state_noisy[:, t] = state_noisy[:, t] + sigma * torch.randn_like(
                                    state_noisy[:, t]
                                )
                        # Detach middle (Plan A): only last frame keeps spatial grads.
                        loss_noisy_p, _preds_noisy = _parallel_branch_loss(
                            state_noisy, chunk=1
                        )
                        loss = loss_clean_p + pf_loss_weight * loss_noisy_p
                    else:
                        loss = loss_clean_p

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                with torch.no_grad():
                    first_pred = preds[:, 0]
                    last_pred = preds[:, -1]
                    first_target = targets[:, 0]
                    first_weight = future_weight[:, 0]
                    last_target = targets[:, w - 1]
                    last_weight = future_weight[:, w - 1]
                    mask0 = (first_weight > 0).to(dtype=first_weight.dtype)
                    # One-step persistence: GT prev frame vs next (targets[:,0]).
                    persistence = tensors["state"][:, 0]
                    pers_mse = weighted_masked_mse(
                        persistence, first_target, mask0, first_weight
                    )
                    pers_rel = weighted_relative_l2(
                        persistence, first_target, mask0, first_weight
                    ).mean()
                    batch_metrics = {
                        "loss": float(loss.detach()),
                        "loss_step0": float(
                            weighted_masked_mse(
                                first_pred, first_target, mask0, first_weight
                            ).detach()
                        ),
                        "relative_l2": float(
                            weighted_relative_l2(
                                first_pred, first_target, mask0, first_weight
                            )
                            .mean()
                            .detach()
                        ),
                        "persistence_mse": float(pers_mse),
                        "persistence_relative_l2": float(pers_rel),
                        # Alias for plot_loss_compare source=persistence.
                        "persistence_loss": float(pers_mse),
                        "unroll_steps": float(clip_t if use_slide else w),
                        "ss_prob": 0.0,
                        "pushforward_last_only": 0.0,
                        "mixed_clean_pushforward": 0.0,
                        "mixed_parallel_noise": float(1.0 if use_mixed_noise else 0.0),
                        "clip_replay": float(1.0 if use_clip_replay else 0.0),
                        "replay_prob": float(replay_prob if use_clip_replay else 0.0),
                        "replay_frac": float(replay_frac if use_clip_replay else 0.0),
                        "clip_replay_horizon": float(horizon_cfg or 0.0),
                        "clip_replay_slide": float(1.0 if use_slide else 0.0),
                        "pf_loss_weight": float(
                            pf_loss_weight if use_mixed_noise else 0.0
                        ),
                        "ar_stream": 0.0,
                        "parallel_causal_tf": 1.0,
                        "trainable_chunk": -1.0,
                        "window_t": float(w),
                    }
                    if loss_l5 is not None:
                        batch_metrics["loss_l5"] = float(loss_l5.detach())
                    if loss_noisy_p is not None:
                        batch_metrics["loss_clean"] = float(loss_clean_p.detach())
                        batch_metrics["loss_noisy"] = float(loss_noisy_p.detach())
                        batch_metrics["noise_eps"] = float(parallel_noise_eps)
                        batch_metrics["noise_a"] = float(parallel_noise_a)
                    for name, pred_t, tgt_t, wt_t in (
                        ("step0", first_pred, first_target, first_weight),
                        ("step_last", last_pred, last_target, last_weight),
                    ):
                        mk = (wt_t > 0).to(dtype=wt_t.dtype)
                        batch_metrics[f"mse_{name}"] = float(
                            weighted_masked_mse(pred_t, tgt_t, mk, wt_t).detach()
                        )
                    for key, value in batch_metrics.items():
                        totals[key] = totals.get(key, 0.0) + float(value)
                        counts[key] = counts.get(key, 0.0) + 1.0
                del preds, loss, loss_clean_p
                if loss_noisy_p is not None:
                    del loss_noisy_p
                if loss_l5 is not None:
                    del loss_l5
                continue

            first_target = targets[:, 0]
            first_weight = future_weight[:, 0]
            last_target = targets[:, unroll_steps - 1]
            last_weight = future_weight[:, unroll_steps - 1]
            mask0 = (first_weight > 0).to(dtype=first_weight.dtype)

            loss_clean: torch.Tensor | None = None
            clean_pred: torch.Tensor | None = None
            clean_window_t = float(state.shape[1])
            if mixed_clean_pushforward:
                memory_clean = model.init_memory(state.shape[0], device, state.dtype)
                if detach_memory:
                    memory_clean = memory_clean.detach()
                t_cur = state.shape[1]
                query_clean = (
                    query[:, :1].expand(-1, t_cur, -1, -1).contiguous()
                    if query.shape[1] != t_cur
                    else query
                )
                result_clean = model.forward_window(
                    query_clean,
                    static,
                    state,
                    boundary,
                    memory_prev=memory_clean,
                    trainable_chunk=trainable_chunk,
                    surface_mask=tensors.get("surface_mask"),
                )
                clean_pred = result_clean["prediction"]
                loss_clean = _sup(
                    clean_pred, first_target, mask0, first_weight
                )

            # PF grow/slide from last GT frame (pure PF or after L_clean).
            surface_mask_pf = tensors.get("surface_mask")
            if start_singleton and unroll_steps > 1 and state.shape[1] > 1:
                state = state[:, -1:].contiguous()
                static = static[:, -1:].contiguous()
                boundary = boundary[:, -1:].contiguous()
                if query.shape[1] > 1:
                    query = query[:, -1:].contiguous()
                if surface_mask_pf is not None and surface_mask_pf.shape[1] > 1:
                    surface_mask_pf = surface_mask_pf[:, -1:].contiguous()

            memory = model.init_memory(state.shape[0], device, state.dtype)
            if detach_memory:
                memory = memory.detach()

            # Last-only PF: weights frozen across the K-step window → reuse slice/KV.
            use_pf_cross_cache = bool(
                train
                and pushforward_last_only
                and unroll_steps > 1
                and (not ar_stream)
                and str(getattr(model, "branch_mode", "unified"))
                not in (
                    "surface_volume_asym",
                    "surface_volume_latent",
                    "surface_volume_per_latent",
                )
            )
            pf_cross_cache = CrossStepHistCache() if use_pf_cross_cache else None
            # Stable ids within this batch's PF episode (content identity for cache).
            pf_frame_ids: list[int] = list(range(int(state.shape[1])))
            next_pf_id = int(state.shape[1])

            step_losses: list[torch.Tensor] = []
            first_pred = None
            last_pred = None

            for step in range(unroll_steps):
                is_last = step + 1 >= unroll_steps
                use_no_grad = bool(
                    train
                    and (not ar_stream)
                    and pushforward_last_only
                    and unroll_steps > 1
                    and not is_last
                )
                t_cur = state.shape[1]
                if query.shape[1] != t_cur:
                    query = query[:, :1].expand(-1, t_cur, -1, -1).contiguous()
                if len(pf_frame_ids) != t_cur:
                    raise RuntimeError(
                        f"pf_frame_ids length {len(pf_frame_ids)} != window T={t_cur}"
                    )
                fwd_kwargs = dict(
                    memory_prev=memory,
                    trainable_chunk=trainable_chunk,
                    surface_mask=surface_mask_pf,
                    frame_ids=pf_frame_ids if pf_cross_cache is not None else None,
                    cross_step_cache=pf_cross_cache,
                )
                if use_no_grad:
                    with torch.no_grad():
                        result = model.forward_window(
                            query,
                            static,
                            state,
                            boundary,
                            **fwd_kwargs,
                        )
                        pred = result["prediction"]
                        memory = result["memory_new"]
                else:
                    result = model.forward_window(
                        query,
                        static,
                        state,
                        boundary,
                        **fwd_kwargs,
                    )
                    pred = result["prediction"]
                    memory = result["memory_new"]
                if detach_memory and memory is not None:
                    memory = memory.detach()
                target = targets[:, step]
                weight = future_weight[:, step]
                mask = (weight > 0).to(dtype=weight.dtype)
                step_loss = _sup(pred, target, mask, weight)
                if use_no_grad:
                    step_loss = step_loss.detach()
                step_losses.append(step_loss.detach() if ar_stream else step_loss)
                if step == 0:
                    first_pred = pred.detach() if (use_no_grad or ar_stream) else pred
                if is_last:
                    last_pred = pred.detach() if ar_stream else pred

                if ar_stream:
                    # One supervise + one parameter update per time step.
                    optimizer.zero_grad(set_to_none=True)
                    step_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                if not is_last:
                    if future_static is None or future_boundary is None:
                        raise RuntimeError("multi-step unroll requires future_* tensors from dataset")
                    if ar_stream or (pushforward_last_only and unroll_steps > 1):
                        next_state = pred.detach()
                    else:
                        next_state = pred.detach() if detach_unroll else pred
                        if train and ss_prob < 1.0:
                            use_pred = (torch.rand(pred.shape[0], device=device) < ss_prob).to(
                                pred.dtype
                            )
                            use_pred = use_pred.view(-1, 1, 1)
                            next_state = use_pred * next_state + (1.0 - use_pred) * targets[:, step]
                    state, static, boundary = advance_causal_window_tensors(
                        state,
                        static,
                        boundary,
                        next_state,
                        future_static[:, step],
                        future_boundary[:, step],
                        max_window=max_window,
                    )
                    pf_frame_ids = pf_frame_ids + [next_pf_id]
                    next_pf_id += 1
                    if len(pf_frame_ids) > max_window:
                        pf_frame_ids = pf_frame_ids[-max_window:]
                    # Keep surface_mask time-aligned with the growing window if present.
                    if surface_mask_pf is not None:
                        # Reuse last surface mask frame (pose changes via static/boundary).
                        if state.shape[1] > surface_mask_pf.shape[1]:
                            surface_mask_pf = torch.cat(
                                [surface_mask_pf, surface_mask_pf[:, -1:]], dim=1
                            )
                        if surface_mask_pf.shape[1] > max_window:
                            surface_mask_pf = surface_mask_pf[:, -max_window:]
                    # Fresh graph for the next stream step (pred already detached).
                    if ar_stream:
                        memory = memory.detach() if memory is not None else memory

            if ar_stream:
                loss = torch.stack(step_losses).mean()
            elif pushforward_last_only and unroll_steps > 1:
                loss_pf = step_losses[-1]
                if loss_clean is not None:
                    loss = loss_clean + pf_loss_weight * loss_pf
                else:
                    loss = loss_pf
            else:
                loss = torch.stack(step_losses).mean()
            # Prefer clean-branch pred for one-step diagnostics when mixed.
            if clean_pred is not None:
                first_pred = clean_pred.detach()
            assert first_pred is not None and last_pred is not None
            with torch.no_grad():
                rel = weighted_relative_l2(first_pred, first_target, mask0, first_weight).mean()
                channel_mse = _channel_weighted_mse(first_pred, first_target, mask0, first_weight)
                persistence = tensors["state"][:, -1]
                pers_mse = weighted_masked_mse(persistence, first_target, mask0, first_weight)
                pers_rel = weighted_relative_l2(persistence, first_target, mask0, first_weight).mean()
                mask_last = (last_weight > 0).to(last_weight.dtype)
                batch_metrics = {
                    "loss": float(loss.detach()),
                    "loss_step0": float(step_losses[0].detach()),
                    "relative_l2": float(rel),
                    "persistence_mse": float(pers_mse),
                    "persistence_relative_l2": float(pers_rel),
                    # Alias for plot_loss_compare source=persistence.
                    "persistence_loss": float(pers_mse),
                    "mse_velocity": float(channel_mse[:3].mean()),
                    "mse_pressure": float(channel_mse[3]),
                    "unroll_steps": float(unroll_steps),
                    "ss_prob": float(ss_prob),
                    "pushforward_last_only": float(1.0 if pushforward_last_only else 0.0),
                    "mixed_clean_pushforward": float(1.0 if mixed_clean_pushforward else 0.0),
                    "pf_loss_weight": float(pf_loss_weight if mixed_clean_pushforward else 0.0),
                    "ar_stream": float(1.0 if ar_stream else 0.0),
                    "trainable_chunk": float(
                        -1.0 if trainable_chunk is None else float(trainable_chunk)
                    ),
                    # Log the clean/GT window length (not the PF-sliced T=1).
                    "window_t": float(clean_window_t),
                }
                if pf_cross_cache is not None:
                    batch_metrics["pf_cache_slice_hits"] = float(pf_cross_cache.slice_hits)
                    batch_metrics["pf_cache_slice_misses"] = float(
                        pf_cross_cache.slice_misses
                    )
                if loss_clean is not None:
                    batch_metrics["loss_clean"] = float(loss_clean.detach())
                    batch_metrics["loss_pf"] = float(step_losses[-1].detach())
                if len(step_losses) > 1:
                    batch_metrics["loss_step_last"] = float(step_losses[-1].detach())
                    batch_metrics["relative_l2_last"] = float(
                        weighted_relative_l2(
                            last_pred, last_target, mask_last, last_weight
                        ).mean()
                    )
                for name, value in zip(_CHANNEL_NAMES, channel_mse.tolist()):
                    batch_metrics[f"mse_{name}"] = float(value)
            if train and not ar_stream:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            _accumulate_metric_dict(totals, counts, batch_metrics, float(first_weight.sum().item()))
    finally:
        torch.set_grad_enabled(_prev_grad_enabled)

    return _finalize_metric_dict(totals, counts)


@torch.no_grad()
def rollout_temporal_traj(
    model: Model,
    traj: dict[str, Any],
    *,
    history_frames: int,
    device: torch.device,
    max_steps: int | None,
    keyframe_timeline: Sequence[int] | None = None,
    use_cross_step_cache: bool = True,
    parallel_causal_tf: bool = False,
) -> dict[str, Any]:
    """AR over input timeline with growing-prefix windows (no left-pad).

    Timeline ``F[0..K-1]`` (e.g. uniform_20). At input index ``s``:
      window on ``F`` ending at ``F[s]`` (``s=0 → [F[0]]``);
      static at next input ``F[s+1]`` for the current slot;
      write ``ar[F[s+1]] = pred`` and score against GT at input ``t+1``.

    ``parallel_causal_tf``: match Stage-1 TF training by refining all window
    frames each layer (not last-only). Disables cross-step cache (incompatible).
    """
    model.eval()
    window = history_frames + 1
    frames = int(traj["frames"])
    if keyframe_timeline is not None:
        timeline = tuple(int(t) for t in keyframe_timeline)
    else:
        timeline = resolve_keyframe_timeline(n_frames=frames)
    if len(timeline) < 2:
        raise ValueError("keyframe timeline too short for temporal window")
    if any(t < 0 or t >= frames for t in timeline):
        raise ValueError(f"keyframe out of range for {frames} frames: {timeline}")

    # Predict next keyframe for each input index except the last.
    n_steps = len(timeline) - 1
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))
    if n_steps < 1:
        raise ValueError("no AR steps on keyframe timeline")

    gt = torch.as_tensor(traj["state"], device=device)
    query = torch.as_tensor(traj["query_xyz"], device=device)
    static = torch.as_tensor(traj["static_fx"], device=device)
    boundary = torch.as_tensor(traj["boundary_feat"], device=device)
    base_weight = torch.as_tensor(traj["base_weight"], device=device)
    valid = torch.as_tensor(traj["valid"], device=device)
    fluid = torch.as_tensor(traj["fluid"], device=device)

    ar = gt.clone()
    memory = None
    asym = str(getattr(model, "branch_mode", "unified")) in (
        "surface_volume_asym",
        "surface_volume_latent",
        "surface_volume_per_latent",
    )
    # parallel TF refines all frames; CrossStepHistCache is last-only / incompatible.
    use_cache = bool(use_cross_step_cache) and (not parallel_causal_tf) and (not asym)
    cross_cache = CrossStepHistCache() if use_cache else None
    surf_all = None
    if asym:
        if "surface_mask" not in traj:
            raise RuntimeError(f"{getattr(model, 'branch_mode', 'unified')} rollout requires traj['surface_mask']")
        surf_all = torch.as_tensor(traj["surface_mask"], device=device, dtype=torch.bool)
    mse_list: list[float] = []
    rel_list: list[float] = []
    channel_sq = torch.zeros(gt.shape[-1], device=device, dtype=torch.float64)
    channel_tgt = torch.zeros(gt.shape[-1], device=device, dtype=torch.float64)
    weight_sum = 0.0

    for seq_pos in range(n_steps):
        local_ids = causal_window_frame_ids(seq_pos, history_frames)
        times = [int(timeline[i]) for i in local_ids]
        if not times or len(times) > window:
            raise RuntimeError(f"bad window ids {times} (max window={window})")
        t_len = len(times)
        # Static at the *next keyframe* for each input slot (matches train dataset).
        static_times = [int(timeline[i + 1]) for i in local_ids]
        target_t = int(timeline[seq_pos + 1])
        state_win = torch.stack([ar[t] for t in times], dim=0).unsqueeze(0)
        query_win = query.unsqueeze(0).unsqueeze(0).expand(1, t_len, -1, -1).contiguous()
        static_win = torch.stack([static[t] for t in static_times], dim=0).unsqueeze(0)
        boundary_win = torch.stack([boundary[t] for t in times], dim=0).unsqueeze(0)
        surf_win = None
        if surf_all is not None:
            surf_win = torch.stack([surf_all[t] for t in times], dim=0).unsqueeze(0)
        out = model.forward_window(
            query_win,
            static_win,
            state_win,
            boundary_win,
            memory_prev=memory,
            frame_ids=None if parallel_causal_tf else local_ids,
            cross_step_cache=None if parallel_causal_tf else cross_cache,
            parallel_causal_tf=parallel_causal_tf,
            surface_mask=surf_win,
        )
        pred = out["prediction"][0]
        memory = out["memory_new"]
        ar[target_t] = pred

        mask = valid[target_t] & fluid[target_t]
        weight = torch.where(mask, base_weight, torch.zeros_like(base_weight))
        target = gt[target_t]
        mse_list.append(
            float(
                weighted_masked_mse(
                    pred.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0), weight.unsqueeze(0)
                )
            )
        )
        rel_list.append(
            float(
                weighted_relative_l2(
                    pred.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0), weight.unsqueeze(0)
                ).item()
            )
        )
        w = weight.to(dtype=torch.float64)
        channel_sq += torch.sum(
            w.unsqueeze(-1) * torch.square(pred.double() - target.double()), dim=0
        )
        channel_tgt += torch.sum(w.unsqueeze(-1) * torch.square(target.double()), dim=0)
        weight_sum += float(w.sum().item())

    channel_rel = [
        float(torch.sqrt(channel_sq[i] / torch.clamp(channel_tgt[i], min=1.0e-12)).item())
        for i in range(channel_sq.numel())
    ]
    cache_stats: dict[str, float] = {}
    if cross_cache is not None:
        cache_stats = {
            "cross_step_geopt_hits": float(cross_cache.geopt_hits),
            "cross_step_geopt_misses": float(cross_cache.geopt_misses),
            "cross_step_slice_hits": float(cross_cache.slice_hits),
            "cross_step_slice_misses": float(cross_cache.slice_misses),
        }
    return {
        "steps": len(mse_list),
        "mse_mean": float(np.mean(mse_list)) if mse_list else float("nan"),
        "relative_l2_mean": float(np.mean(rel_list)) if rel_list else float("nan"),
        "relative_l2_ar0": float(rel_list[0]) if rel_list else float("nan"),
        "relative_l2_final": float(rel_list[-1]) if rel_list else float("nan"),
        "relative_l2_by_step": rel_list,
        "channel_relative_l2": channel_rel,
        "supervised_weight_sum": weight_sum,
        **cache_stats,
    }


@torch.no_grad()
def rollout_validation_split(
    model: Model,
    traj_manifest,
    normalizer: DynamicStateNormalizer,
    *,
    history_frames: int,
    points: int,
    subset_seed: int,
    device: torch.device,
    max_steps: int,
    max_trajs: int = 0,
    parallel_causal_tf: bool = False,
    split: str = "validation",
) -> dict[str, float]:
    if split not in SPLITS:
        raise ValueError(f"rollout split must be one of {SPLITS}, got {split!r}")
    records = [r for r in traj_manifest.records if r.split == split]
    if max_trajs > 0:
        records = records[:max_trajs]
    if not records:
        raise ValueError(f"no {split!r} trajectories for rollout")

    rel_means: list[float] = []
    rel_ar0s: list[float] = []
    rel_finals: list[float] = []
    mse_means: list[float] = []
    step_counts: list[int] = []
    depth_vals: dict[int, list[float]] = {d: [] for d in ROLLOUT_REPORT_DEPTHS}
    channel_acc = np.zeros(4, dtype=np.float64)
    for record in records:
        shape = validate_trajectory_hdf5(record.hdf5_path, validate_values=False)
        indices = resolve_point_indices(
            shape.points, points, seed=subset_seed, point_bank_id=record.point_bank_id
        )
        traj = load_trajectory_arrays(
            record.hdf5_path,
            point_indices=indices,
            normalizer=normalizer,
            speed=record.speed_m_per_s,
        )
        timeline = resolve_keyframe_timeline(
            n_frames=shape.frames,
            valid_frame_ids=record.valid_frame_ids,
            accepted_transition_indices=record.accepted_transition_indices,
        )
        out = rollout_temporal_traj(
            model,
            traj,
            history_frames=history_frames,
            device=device,
            max_steps=max_steps,
            keyframe_timeline=timeline,
            use_cross_step_cache=not parallel_causal_tf,
            parallel_causal_tf=parallel_causal_tf,
        )
        rel_means.append(out["relative_l2_mean"])
        rel_ar0s.append(out["relative_l2_ar0"])
        rel_finals.append(out["relative_l2_final"])
        mse_means.append(out["mse_mean"])
        step_counts.append(int(out["steps"]))
        channel_acc += np.asarray(out["channel_relative_l2"], dtype=np.float64)
        rel_by_step = out.get("relative_l2_by_step") or []
        for depth in ROLLOUT_REPORT_DEPTHS:
            if len(rel_by_step) >= depth:
                depth_vals[depth].append(float(rel_by_step[depth - 1]))

    channel_acc /= max(len(records), 1)
    metrics = {
        "validation_rollout_relative_l2": float(np.mean(rel_means)),
        # First AR step: seq_pos=0 → window [0] (no left-pad), predict input t+1.
        "validation_rollout_relative_l2_ar0": float(np.mean(rel_ar0s)),
        "validation_rollout_relative_l2_final": float(np.mean(rel_finals)),
        "validation_rollout_mse": float(np.mean(mse_means)),
        "validation_rollout_n_trajs": float(len(records)),
        "validation_rollout_steps": float(min(step_counts) if step_counts else max_steps),
    }
    for depth, values in depth_vals.items():
        if values:
            metrics[f"validation_rollout_relL2_at_{depth}"] = float(np.mean(values))
    for name, value in zip(_CHANNEL_NAMES, channel_acc.tolist()):
        metrics[f"validation_rollout_relL2_{name}"] = float(value)
    return metrics


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    manifest = Path(args.manifest or paths["manifest"]).resolve()
    output_dir = Path(
        args.output_dir
        or Path(os.path.expandvars(str(paths.get("runs_root", "runs/temporal"))))
        / f"seed_{args.seed or train_cfg.get('seeds', [17])[0]}"
    ).resolve()
    seed = int(args.seed or train_cfg.get("seeds", [17])[0])
    points = int(args.points or train_cfg.get("points", 70000))
    epochs = int(args.epochs or train_cfg.get("epochs", 200))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 1))
    lr = float(args.lr or train_cfg.get("lr", 1.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-5))
    initialization = str(
        args.initialization or train_cfg.get("initialization", "from_scratch")
    )
    if initialization not in ("from_scratch", "geopt_pretrained", "from_checkpoint"):
        raise ValueError(
            "initialization must be from_scratch|geopt_pretrained|from_checkpoint, "
            f"got {initialization!r}"
        )
    init_checkpoint_raw = args.init_checkpoint or train_cfg.get("init_checkpoint")
    init_checkpoint: Path | None = (
        Path(init_checkpoint_raw).resolve() if init_checkpoint_raw else None
    )
    if initialization == "from_checkpoint" and init_checkpoint is None:
        raise ValueError(
            "initialization=from_checkpoint requires training.init_checkpoint "
            "or --init-checkpoint"
        )
    lr_spatial_schedule = parse_lr_mult_schedule(train_cfg.get("lr_spatial_mult_schedule"))
    lr_temporal_schedule = parse_lr_mult_schedule(train_cfg.get("lr_temporal_mult_schedule"))
    patience = int(train_cfg.get("early_stop_patience", 40))
    early_stop_min_epoch = int(train_cfg.get("early_stop_min_epoch", 0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    history_frames = int(model_cfg.get("history_frames", 4))
    subset_seed = int(train_cfg.get("subset_seed", 20260803))
    hist_noise_std = float(train_cfg.get("hist_noise_std", 0.0))
    if hist_noise_std < 0:
        raise ValueError("hist_noise_std must be >= 0")
    pushforward_k_schedule = parse_pushforward_k_schedule(
        train_cfg.get("pushforward_k_schedule")
    )
    use_pushforward_curriculum = bool(pushforward_k_schedule)
    pushforward_last_only = bool(
        train_cfg.get(
            "pushforward_last_only",
            True if use_pushforward_curriculum else False,
        )
    )
    temporal_bptt_mode = str(train_cfg.get("temporal_bptt_mode", "legacy"))
    window_frames_cfg = int(model_cfg.get("window_frames", history_frames + 1))
    trainable_chunk = trainable_chunk_from_mode(
        temporal_bptt_mode, window_frames=window_frames_cfg
    )
    if use_pushforward_curriculum:
        max_k = max(k for _, k in pushforward_k_schedule)
        # Config unroll_steps is documentary; schedule drives actual K.
        unroll_steps = int(train_cfg.get("unroll_steps", max(1, max_k)))
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        if max_k > 0 and unroll_steps < max_k:
            unroll_steps = max_k
    else:
        unroll_steps = int(train_cfg.get("unroll_steps", 1))
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
    parallel_causal_tf = bool(train_cfg.get("parallel_causal_tf", False))
    parallel_window_frames = int(
        train_cfg.get("parallel_window_frames", model_cfg.get("window_frames", 4))
    )
    raw_pos_w = train_cfg.get("position_loss_weights")
    position_loss_weights: list[float] | None
    if raw_pos_w is None:
        position_loss_weights = None
    else:
        position_loss_weights = [float(x) for x in raw_pos_w]
    if parallel_causal_tf:
        if parallel_window_frames < 1:
            raise ValueError("parallel_window_frames must be >= 1")
        if window_frames_cfg != parallel_window_frames:
            raise ValueError(
                "model.window_frames must equal training.parallel_window_frames "
                f"({window_frames_cfg} vs {parallel_window_frames})"
            )
        if position_loss_weights is not None and len(position_loss_weights) != parallel_window_frames:
            raise ValueError(
                "position_loss_weights length must equal parallel_window_frames"
            )
        # Force clean TF clip training; refuse AR/pushforward combos.
        unroll_steps = 1
        closed_loop_start_epoch = 0
        use_pushforward_curriculum = False
        pushforward_last_only = False
        trainable_chunk = None
    sample_stride = int(train_cfg.get("sample_stride", 1))
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if not parallel_causal_tf:
        closed_loop_start_epoch = int(train_cfg.get("closed_loop_start_epoch", 0))
    if closed_loop_start_epoch < 0:
        raise ValueError("closed_loop_start_epoch must be >= 0")
    if use_pushforward_curriculum and closed_loop_start_epoch not in (0, 1):
        raise ValueError(
            "pushforward_k_schedule cannot be combined with closed_loop_start_epoch>1; "
            "use one curriculum mechanism"
        )
    # 0 / omitted → closed-loop from epoch 1 when unroll_steps>1 (legacy).
    # N>1 → epochs [1, N) causal (K=1); epochs [N, epochs] closed-loop (K=unroll_steps).
    if not use_pushforward_curriculum:
        if closed_loop_start_epoch == 0 and unroll_steps > 1:
            closed_loop_start_epoch = 1
        if closed_loop_start_epoch > epochs + 1:
            raise ValueError("closed_loop_start_epoch must be <= epochs+1")
        if closed_loop_start_epoch > 1 and unroll_steps <= 1:
            raise ValueError(
                "closed_loop_start_epoch>1 requires unroll_steps>1 for the closed-loop phase"
            )
    else:
        closed_loop_start_epoch = 0
    ss_default = (
        0.0
        if use_pushforward_curriculum or pushforward_last_only
        else (1.0 if (unroll_steps > 1 or closed_loop_start_epoch > 1) else 0.0)
    )
    ss_prob_max = float(train_cfg.get("ss_prob", ss_default))
    if ss_prob_max < 0 or ss_prob_max > 1:
        raise ValueError("ss_prob must be in [0, 1]")
    ss_warmup_epochs = int(train_cfg.get("ss_warmup_epochs", 50))
    detach_unroll = bool(train_cfg.get("detach_unroll", True))
    ar_stream = bool(train_cfg.get("ar_stream", False))
    stream_tf_clip = bool(train_cfg.get("stream_tf_clip", False))
    if stream_tf_clip:
        if ar_stream:
            raise ValueError("stream_tf_clip cannot combine with ar_stream")
        if use_pushforward_curriculum:
            raise ValueError("stream_tf_clip cannot combine with pushforward_k_schedule")
        if pushforward_last_only:
            raise ValueError("stream_tf_clip requires pushforward_last_only=false")
        if unroll_steps < 2:
            raise ValueError("stream_tf_clip requires unroll_steps>=2")
        if temporal_bptt_mode not in ("plan_a", "plana", "a"):
            raise ValueError("stream_tf_clip requires temporal_bptt_mode=plan_a")
        # Streaming TF: GT writeback, mean loss, one backward — never SS/AR.
        pushforward_last_only = False
        ss_prob_max = 0.0
        ss_warmup_epochs = 0
        closed_loop_start_epoch = 1
        ar_stream = False
    if ar_stream:
        if unroll_steps < 2:
            raise ValueError("ar_stream=True requires unroll_steps>=2")
        if use_pushforward_curriculum:
            raise ValueError("ar_stream cannot combine with pushforward_k_schedule")
        # Force closed-loop AR writeback semantics from epoch 1.
        pushforward_last_only = False
        ss_prob_max = 1.0
        closed_loop_start_epoch = 1
    mixed_clean_pushforward = bool(train_cfg.get("mixed_clean_pushforward", False))
    mixed_parallel_noise = bool(train_cfg.get("mixed_parallel_noise", False))
    parallel_noise_eps = float(train_cfg.get("parallel_noise_eps", 0.0))
    parallel_noise_a = float(train_cfg.get("parallel_noise_a", 0.0))
    clip_replay = bool(train_cfg.get("clip_replay", False))
    replay_prob = float(train_cfg.get("replay_prob", 0.0))
    replay_prob_schedule = parse_replay_prob_schedule(
        train_cfg.get("replay_prob_schedule")
    )
    raw_horizon = train_cfg.get("clip_replay_horizon")
    clip_replay_horizon: int | None
    if raw_horizon is None:
        clip_replay_horizon = None
    else:
        clip_replay_horizon = int(raw_horizon)
    pf_loss_weight = float(train_cfg.get("pf_loss_weight", 0.5))
    if pf_loss_weight < 0:
        raise ValueError("pf_loss_weight must be >= 0")
    train_loss_mode = str(train_cfg.get("train_loss_mode", "mse")).strip().lower()
    raw_ch_w = train_cfg.get("channel_loss_weights")
    channel_loss_weights: list[float] | None
    if raw_ch_w is None:
        channel_loss_weights = None
    else:
        channel_loss_weights = [float(x) for x in raw_ch_w]
        if len(channel_loss_weights) != int(model_cfg.get("state_dim", 4)):
            raise ValueError(
                "channel_loss_weights length must equal model.state_dim "
                f"({len(channel_loss_weights)} vs {model_cfg.get('state_dim')})"
            )
    rel_vp_alpha = float(train_cfg.get("rel_vp_alpha", 1.0))
    rel_vp_beta = float(train_cfg.get("rel_vp_beta", 1.0))
    if train_loss_mode not in ("mse", "rel_vp", "relvp", "relative_vp"):
        raise ValueError(
            f"train_loss_mode must be mse|rel_vp, got {train_loss_mode!r}"
        )
    if train_loss_mode == "mse" and channel_loss_weights is None:
        pass  # default equal-weight MSE
    if train_loss_mode in ("rel_vp", "relvp", "relative_vp"):
        if channel_loss_weights is not None:
            raise ValueError("channel_loss_weights is only valid with train_loss_mode=mse")
        if rel_vp_alpha < 0 or rel_vp_beta < 0 or (rel_vp_alpha + rel_vp_beta) <= 0:
            raise ValueError("rel_vp_alpha/beta must be >=0 and sum>0")
    if mixed_parallel_noise and mixed_clean_pushforward:
        raise ValueError("mixed_parallel_noise cannot combine with mixed_clean_pushforward")
    if mixed_parallel_noise and not parallel_causal_tf:
        raise ValueError("mixed_parallel_noise requires parallel_causal_tf=true")
    if mixed_parallel_noise and (parallel_noise_eps < 0 or parallel_noise_a < 0):
        raise ValueError("parallel_noise_eps/a must be >= 0")
    if clip_replay and not parallel_causal_tf:
        raise ValueError("clip_replay requires parallel_causal_tf=true")
    if clip_replay and mixed_parallel_noise:
        raise ValueError("clip_replay cannot combine with mixed_parallel_noise")
    if clip_replay and mixed_clean_pushforward:
        raise ValueError("clip_replay cannot combine with mixed_clean_pushforward")
    if clip_replay and (replay_prob < 0.0 or replay_prob > 1.0):
        raise ValueError("replay_prob must be in [0, 1]")
    if replay_prob_schedule and not clip_replay:
        raise ValueError("replay_prob_schedule requires clip_replay=true")
    if clip_replay_horizon is not None:
        if not clip_replay:
            raise ValueError("clip_replay_horizon requires clip_replay=true")
        if clip_replay_horizon < parallel_window_frames:
            raise ValueError(
                "clip_replay_horizon must be >= parallel_window_frames "
                f"({clip_replay_horizon} < {parallel_window_frames})"
            )
        if clip_replay_horizon > parallel_window_frames + 1:
            raise ValueError(
                "clip_replay_horizon currently supports only W or W+1 "
                f"(got {clip_replay_horizon}, W={parallel_window_frames})"
            )
    if parallel_causal_tf and (
        ar_stream or stream_tf_clip or mixed_clean_pushforward or use_pushforward_curriculum
    ):
        raise ValueError(
            "parallel_causal_tf cannot combine with ar_stream / stream_tf_clip / "
            "mixed_clean_pushforward / pushforward_k_schedule"
        )
    if mixed_parallel_noise:
        # Noisy branch uses Plan-A hist detach inside parallel TF.
        if temporal_bptt_mode not in ("plan_a", "plana", "a", "legacy"):
            # Allow legacy cfg; noisy branch still forces chunk=1.
            pass
        ss_prob_max = 0.0
        closed_loop_start_epoch = 1
    if clip_replay:
        ss_prob_max = 0.0
        closed_loop_start_epoch = 1
    if mixed_clean_pushforward:
        if ar_stream:
            raise ValueError("mixed_clean_pushforward cannot combine with ar_stream")
        if stream_tf_clip:
            raise ValueError("mixed_clean_pushforward cannot combine with stream_tf_clip")
        if use_pushforward_curriculum:
            raise ValueError(
                "mixed_clean_pushforward cannot combine with pushforward_k_schedule"
            )
        if unroll_steps < 2:
            raise ValueError("mixed_clean_pushforward requires unroll_steps>=2")
        # PF branch is last-only; clean branch is separate one-step TF.
        pushforward_last_only = True
        ss_prob_max = 1.0
        ss_warmup_epochs = 0
        closed_loop_start_epoch = 1
    validation_rollout_steps = int(train_cfg.get("validation_rollout_steps", 0))
    validation_rollout_max_trajs = int(train_cfg.get("validation_rollout_max_trajs", 0))
    validation_rollout_every = int(train_cfg.get("validation_rollout_every", 1))
    if validation_rollout_every < 1:
        raise ValueError("validation_rollout_every must be >= 1")
    selection_split = str(train_cfg.get("selection_split", "validation"))
    if selection_split not in SPLITS:
        raise ValueError(
            f"selection_split must be one of {SPLITS}, got {selection_split!r}"
        )
    selection_metric_name = str(train_cfg.get("selection_metric", "validation_loss"))
    if selection_metric_name not in _VALID_SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {sorted(_VALID_SELECTION_METRICS)}, "
            f"got {selection_metric_name!r}"
        )
    if (
        selection_metric_name == "validation_rollout_relative_l2"
        and validation_rollout_steps <= 0
    ):
        raise ValueError(
            "selection_metric=validation_rollout_relative_l2 requires "
            "validation_rollout_steps > 0"
        )
    if early_stop_min_epoch < 0:
        raise ValueError("early_stop_min_epoch must be >= 0")
    if early_stop_min_epoch > epochs:
        raise ValueError("early_stop_min_epoch must not exceed epochs")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_checkpoint(output_dir, args)

    traj_manifest = load_trajectory_manifest(
        manifest,
        require_all_splits=False,
        require_production_eligible=True,
        require_hashes=not args.allow_unhashed_data,
        verify_hashes=False,
    )
    present_splits = {r.split for r in traj_manifest.records}
    if "train" not in present_splits:
        raise ValueError("manifest has no train trajectories")
    if selection_split not in present_splits:
        raise ValueError(
            f"selection_split={selection_split!r} missing from manifest "
            f"(have {sorted(present_splits)})"
        )
    sample_speed = next(
        (r.speed_m_per_s for r in traj_manifest.records if r.speed_m_per_s is not None),
        1.0,
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    l_ref_raw = raw.get("dataset_metadata", {}).get("l_ref_m", 1.0)
    try:
        l_ref = float(l_ref_raw)
    except (TypeError, ValueError):
        l_ref = 1.0

    normalizer_path: Path | None = None
    if args.normalizer_path is not None:
        normalizer_path = Path(args.normalizer_path).resolve()
    elif paths.get("normalizer"):
        normalizer_path = Path(str(paths["normalizer"])).resolve()

    pin_memory = bool(train_cfg.get("pin_memory", False)) and device.type == "cuda"
    architecture = make_model_args(model_cfg, args)
    train_cfg = dict(train_cfg)
    train_cfg["initialization"] = initialization
    train_cfg["stream_tf_clip"] = stream_tf_clip
    train_cfg["ar_stream"] = ar_stream
    train_cfg["ss_prob"] = ss_prob_max
    train_cfg["pushforward_last_only"] = pushforward_last_only
    train_cfg["mixed_clean_pushforward"] = mixed_clean_pushforward
    train_cfg["mixed_parallel_noise"] = mixed_parallel_noise
    train_cfg["parallel_noise_eps"] = parallel_noise_eps
    train_cfg["parallel_noise_a"] = parallel_noise_a
    train_cfg["clip_replay"] = clip_replay
    train_cfg["replay_prob"] = replay_prob
    train_cfg["replay_prob_schedule"] = [
        {"until_epoch": u, "prob": p, "mode": mode}
        for u, p, mode in replay_prob_schedule
    ]
    train_cfg["clip_replay_horizon"] = clip_replay_horizon
    train_cfg["pf_loss_weight"] = pf_loss_weight
    train_cfg["selection_split"] = selection_split
    if init_checkpoint is not None:
        train_cfg["init_checkpoint"] = str(init_checkpoint)
    if normalizer_path is not None:
        train_cfg["normalizer_path"] = str(normalizer_path)
    experiment = build_experiment_snapshot(
        cfg=cfg,
        config_path=Path(args.config),
        manifest=manifest,
        output_dir=output_dir,
        seed=seed,
        points=points,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        early_stop_min_epoch=early_stop_min_epoch,
        max_grad_norm=max_grad_norm,
        history_frames=history_frames,
        subset_seed=subset_seed,
        architecture=architecture,
        train_cfg=train_cfg,
        pin_memory=pin_memory,
    )
    # Filesystem copy of the exact yaml used for this run (audit / resume).
    (output_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    write_json(output_dir / "experiment.json", experiment)

    model = Model(architecture)
    geopt_load_info: dict[str, Any] | None = None
    finetune_load_info: dict[str, Any] | None = None
    if initialization == "geopt_pretrained":
        pretrained = Path(
            args.pretrained_checkpoint
            or paths.get("geopt_checkpoint")
            or GEOPT_CHECKPOINT
        ).resolve()
        expected_sha = args.pretrained_sha256
        if expected_sha == "":
            expected_sha = None
        geopt_load_info = load_geopt_backbone(
            model, pretrained, expected_sha256=expected_sha
        )
        print(
            json.dumps(
                {
                    "status": "geopt_pretrained_loaded",
                    "checkpoint": str(pretrained),
                    "sha256": geopt_load_info.get("sha256"),
                    "loaded_tensor_count": geopt_load_info.get("loaded_tensor_count"),
                    "loaded_parameter_elements": geopt_load_info.get(
                        "loaded_parameter_elements"
                    ),
                },
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
    elif initialization == "from_checkpoint":
        assert init_checkpoint is not None
        finetune_load_info = load_finetune_checkpoint_weights(model, init_checkpoint)
        print(
            json.dumps(
                {
                    "status": "from_checkpoint_loaded",
                    **finetune_load_info,
                },
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
    model = model.to(device)
    spatial_mult0 = lr_mult_for_epoch(1, lr_spatial_schedule, default=1.0)
    temporal_mult0 = lr_mult_for_epoch(1, lr_temporal_schedule, default=1.0)
    optimizer = build_adamw_param_groups(
        model,
        lr=lr,
        weight_decay=weight_decay,
        spatial_mult=spatial_mult0,
        temporal_mult=temporal_mult0,
    )
    print(
        json.dumps(
            {
                "status": "optimizer_param_groups",
                "initialization": initialization,
                "base_lr": lr,
                "lr_spatial_mult_ep1": spatial_mult0,
                "lr_temporal_mult_ep1": temporal_mult0,
                "groups": [
                    {
                        "name": g.get("name"),
                        "lr": g["lr"],
                        "n_params": sum(p.numel() for p in g["params"]),
                    }
                    for g in optimizer.param_groups
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    history: list[dict] = []
    best = float("inf")
    best_epoch = 0
    best_selection_metric = selection_metric_name
    best_step = float("inf")
    best_step_epoch = 0
    best_rollout = float("inf")
    best_rollout_epoch = 0
    start_epoch = 1
    normalizer: DynamicStateNormalizer

    if resume_path is not None:
        resume_probe = torch.load(resume_path, map_location="cpu", weights_only=False)
        has_optimizer = isinstance(resume_probe, dict) and resume_probe.get("optimizer") is not None
        normalizer, history, best, best_epoch, start_epoch = load_training_resume(
            resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            manifest=manifest,
            seed=seed,
            points=points,
            experiment=experiment,
            allow_partial=bool(args.resume_allow_partial),
        )
        step_best = best_metric_from_history(history, "validation_loss")
        if step_best is not None:
            best_step, best_step_epoch = step_best
        roll_best = best_metric_from_history(history, "validation_rollout_relative_l2")
        if roll_best is not None:
            best_rollout, best_rollout_epoch = roll_best
        write_json(output_dir / "normalizer.json", normalizer.to_dict())
        print(
            json.dumps(
                {
                    "status": "resume_start",
                    "checkpoint": str(resume_path),
                    "start_epoch": start_epoch,
                    "best_epoch": best_epoch,
                    "best_selection_value": best,
                    "selection_metric": selection_metric_name,
                    "best_step_epoch": best_step_epoch,
                    "best_step_validation_loss": (
                        None if best_step == float("inf") else best_step
                    ),
                    "best_rollout_epoch": best_rollout_epoch,
                    "best_rollout_relative_l2": (
                        None if best_rollout == float("inf") else best_rollout
                    ),
                    "history_epochs": len(history),
                    "has_optimizer": has_optimizer,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        if normalizer_path is not None:
            if not normalizer_path.is_file():
                raise FileNotFoundError(f"normalizer not found: {normalizer_path}")
            normalizer = DynamicStateNormalizer.from_dict(
                json.loads(normalizer_path.read_text(encoding="utf-8"))
            )
            l_ref = float(normalizer.l_ref_m)
            print(
                json.dumps(
                    {
                        "status": "normalizer_loaded",
                        "path": str(normalizer_path),
                        "l_ref_m": l_ref,
                        "u_ref_m_per_s": float(normalizer.u_ref_m_per_s),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            normalizer = DynamicStateNormalizer.fit_from_manifest(
                traj_manifest,
                u_ref_m_per_s=float(sample_speed),
                l_ref_m=l_ref,
                p_farfield_kinematic=0.0,
                subset_points=points,
                subset_seed=subset_seed,
            )
        write_json(output_dir / "normalizer.json", normalizer.to_dict())

    frame_cache_max_bytes = int(train_cfg.get("frame_cache_max_bytes", 0))
    if frame_cache_max_bytes < 0:
        raise ValueError("frame_cache_max_bytes must be >= 0")

    def _make_split_dataset(split: str, train_unroll: int) -> TemporalCFDWindowDataset:
        if parallel_causal_tf:
            # Train may use a longer clip (horizon=W+1) for slide L5; val/test stay W.
            if (
                split == "train"
                and clip_replay_horizon is not None
                and clip_replay_horizon > parallel_window_frames
            ):
                fixed_w: int | None = int(clip_replay_horizon)
            else:
                fixed_w = parallel_window_frames
        else:
            fixed_w = None
        return TemporalCFDWindowDataset(
            traj_manifest,
            split,
            normalizer,
            history_frames=history_frames,
            subset_points=points,
            subset_seed=subset_seed,
            unroll_steps=train_unroll if split == "train" else 1,
            sample_stride=sample_stride if split == "train" else 1,
            frame_cache_max_bytes=frame_cache_max_bytes,
            fixed_window_frames=fixed_w,
            # Dataset always returns full causal windows; start_singleton is
            # applied inside run_epoch (PF slice after L_clean).
            start_singleton=False,
        )

    def _make_loaders(train_unroll: int) -> tuple[dict[str, TemporalCFDWindowDataset], dict[str, DataLoader]]:
        splits_to_build = ["train", selection_split]
        if "test" in present_splits and "test" not in splits_to_build:
            splits_to_build.append("test")
        ds = {
            split: _make_split_dataset(split, train_unroll)
            for split in splits_to_build
        }
        ld = {
            split: build_dataloader(
                ds[split],
                batch_size=batch_size,
                shuffle=(
                    bool(train_cfg.get("shuffle", True))
                    if split == "train"
                    else False
                ),
                train_cfg=train_cfg,
                device=device,
                seed=seed + (0 if split == "train" else 1),
            )
            for split in ds
        }
        return ds, ld

    # Curriculum / closed-loop: rebuild dataset when train unroll K changes.
    def train_unroll_for_epoch(epoch: int) -> int:
        if use_pushforward_curriculum:
            return unroll_steps_from_k_max(k_max_for_epoch(epoch, pushforward_k_schedule))
        if unroll_steps <= 1:
            return 1
        if closed_loop_start_epoch <= 1:
            return unroll_steps
        return unroll_steps if int(epoch) >= closed_loop_start_epoch else 1

    def k_max_for_logging(epoch: int) -> int:
        if use_pushforward_curriculum:
            return k_max_for_epoch(epoch, pushforward_k_schedule)
        # Legacy: unroll_steps==1 → k_max 0 (TF); else k_max == unroll_steps.
        eu = train_unroll_for_epoch(epoch)
        return 0 if eu <= 1 else eu

    active_train_unroll = train_unroll_for_epoch(start_epoch)
    datasets, loaders = _make_loaders(active_train_unroll)

    run_manifest = {
        "config": str(Path(args.config).resolve()),
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "seed": seed,
        "points": points,
        "model": vars(architecture),
        "training": {
            "lr": lr,
            "weight_decay": float(train_cfg.get("weight_decay", 1.0e-5)),
            "epochs": epochs,
            "early_stop_patience": patience,
            "early_stop_min_epoch": early_stop_min_epoch,
            "max_grad_norm": max_grad_norm,
            "history_frames": history_frames,
            "subset_seed": subset_seed,
            "batch_size": batch_size,
            "initialization": initialization,
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "detach_memory_each_step": bool(train_cfg.get("detach_memory_each_step", True)),
            "hist_noise_std": hist_noise_std,
            "unroll_steps": unroll_steps,
            "sample_stride": sample_stride,
            "closed_loop_start_epoch": closed_loop_start_epoch,
            "pushforward_k_schedule": [
                {"until_epoch": until, "k_max": k} for until, k in pushforward_k_schedule
            ],
            "pushforward_last_only": pushforward_last_only,
            "stream_tf_clip": stream_tf_clip,
            "ar_stream": ar_stream,
            "mixed_clean_pushforward": mixed_clean_pushforward,
            "mixed_parallel_noise": mixed_parallel_noise,
            "parallel_noise_eps": parallel_noise_eps,
            "parallel_noise_a": parallel_noise_a,
            "clip_replay": clip_replay,
            "replay_prob": replay_prob,
            "replay_prob_schedule": [
                {"until_epoch": u, "prob": p, "mode": mode}
                for u, p, mode in replay_prob_schedule
            ],
            "clip_replay_horizon": clip_replay_horizon,
            "pf_loss_weight": pf_loss_weight,
            "train_loss_mode": train_loss_mode,
            "channel_loss_weights": channel_loss_weights,
            "rel_vp_alpha": rel_vp_alpha,
            "rel_vp_beta": rel_vp_beta,
            "lr_spatial_mult_schedule": [
                {"until_epoch": u, "mult": m, "mode": mode}
                for u, m, mode in lr_spatial_schedule
            ],
            "lr_temporal_mult_schedule": [
                {"until_epoch": u, "mult": m, "mode": mode}
                for u, m, mode in lr_temporal_schedule
            ],
            "temporal_bptt_mode": temporal_bptt_mode,
            "trainable_chunk": trainable_chunk,
            "ss_prob": ss_prob_max,
            "ss_warmup_epochs": ss_warmup_epochs,
            "detach_unroll": detach_unroll,
            "validation_rollout_steps": validation_rollout_steps,
            "validation_rollout_max_trajs": validation_rollout_max_trajs,
            "validation_rollout_every": validation_rollout_every,
            "selection_metric": selection_metric_name,
            "selection_split": selection_split,
        },
        "experiment": experiment,
        "split_counts": {split: len(datasets[split]) for split in datasets},
        "dataloader": experiment["resolved"]["dataloader"],
        "status": "resumed" if resume_path is not None else "running",
    }
    if resume_path is not None:
        run_manifest["resume_checkpoint"] = str(resume_path)
        run_manifest["resumed_from_epoch"] = int(history[-1]["epoch"])
    write_json(output_dir / "run_manifest.json", run_manifest)

    print(
        json.dumps(
            {
                "status": "train_config",
                "selection_metric": selection_metric_name,
                "selection_split": selection_split,
                "hist_noise_std": hist_noise_std,
                "unroll_steps": unroll_steps,
                "sample_stride": sample_stride,
                "pushforward_k_schedule": [
                    {"until_epoch": until, "k_max": k} for until, k in pushforward_k_schedule
                ],
                "pushforward_last_only": pushforward_last_only,
                "stream_tf_clip": stream_tf_clip,
                "ar_stream": ar_stream,
                "mixed_clean_pushforward": mixed_clean_pushforward,
                "mixed_parallel_noise": mixed_parallel_noise,
                "parallel_noise_eps": parallel_noise_eps,
                "parallel_noise_a": parallel_noise_a,
                "clip_replay": clip_replay,
                "replay_prob": replay_prob,
                "replay_prob_schedule": [
                    {"until_epoch": u, "prob": p, "mode": mode}
                    for u, p, mode in replay_prob_schedule
                ],
                "clip_replay_horizon": clip_replay_horizon,
                "pf_loss_weight": pf_loss_weight,
                "train_loss_mode": train_loss_mode,
                "channel_loss_weights": channel_loss_weights,
                "rel_vp_alpha": rel_vp_alpha,
                "rel_vp_beta": rel_vp_beta,
                "initialization": initialization,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                "temporal_bptt_mode": temporal_bptt_mode,
                "trainable_chunk": trainable_chunk,
                "ss_prob": ss_prob_max,
                "ss_warmup_epochs": ss_warmup_epochs,
                "detach_unroll": detach_unroll,
                "use_hist_state_embed": bool(
                    getattr(architecture, "use_hist_state_embed", True)
                ),
                "validation_rollout_steps": validation_rollout_steps,
                "validation_rollout_every": validation_rollout_every,
                "selection_metric": selection_metric_name,
                "selection_split": selection_split,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if start_epoch > epochs:
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "start_epoch": start_epoch,
                    "epochs": epochs,
                    "best_epoch": best_epoch,
                    "best_selection_value": best,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        replay_model: Model | None = None
        if clip_replay and start_epoch > 1:
            # Resume into epoch>=2: lagged weights == checkpoint (end of prior epoch).
            replay_model = sync_clip_replay_model(None, model)
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            spatial_mult = lr_mult_for_epoch(epoch, lr_spatial_schedule, default=1.0)
            temporal_mult = lr_mult_for_epoch(epoch, lr_temporal_schedule, default=1.0)
            applied_lrs = apply_param_group_lr(
                optimizer,
                base_lr=lr,
                spatial_mult=spatial_mult,
                temporal_mult=temporal_mult,
            )
            epoch_unroll = train_unroll_for_epoch(epoch)
            epoch_k_max = k_max_for_logging(epoch)
            if epoch_unroll != active_train_unroll:
                datasets, loaders = _make_loaders(epoch_unroll)
                active_train_unroll = epoch_unroll
                print(
                    json.dumps(
                        {
                            "status": "phase_switch",
                            "epoch": epoch,
                            "train_unroll_steps": epoch_unroll,
                            "k_max": epoch_k_max,
                            "closed_loop_start_epoch": closed_loop_start_epoch,
                            "pushforward_last_only": pushforward_last_only,
                            "train_samples": len(datasets["train"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            in_closed_loop = epoch_unroll > 1 and not use_pushforward_curriculum
            in_pushforward = use_pushforward_curriculum and epoch_k_max >= 1
            if in_closed_loop and closed_loop_start_epoch > 1:
                phase_epoch = epoch - closed_loop_start_epoch + 1
            elif in_closed_loop:
                phase_epoch = epoch
            else:
                phase_epoch = 0
            ss_now = (
                1.0
                if ar_stream
                else (
                    scheduled_sampling_prob(
                        epoch,
                        ss_prob=ss_prob_max,
                        ss_warmup_epochs=ss_warmup_epochs,
                        phase_epoch=phase_epoch if in_closed_loop else None,
                    )
                    if in_closed_loop and not pushforward_last_only
                    else 0.0
                )
            )
            epoch_replay_prob = replay_prob_for_epoch(
                epoch,
                clip_replay=clip_replay,
                replay_prob=replay_prob,
                schedule=replay_prob_schedule,
            )
            train_timer = _phase_timing_start(device)
            train_metrics = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                max_grad_norm=max_grad_norm,
                detach_memory=bool(train_cfg.get("detach_memory_each_step", True)),
                pin_memory=pin_memory,
                hist_noise_std=hist_noise_std,
                unroll_steps=epoch_unroll,
                ss_prob=ss_now if epoch_unroll > 1 else 0.0,
                detach_unroll=detach_unroll,
                pushforward_last_only=pushforward_last_only and epoch_unroll > 1 and not ar_stream,
                ar_stream=bool(ar_stream and epoch_unroll > 1),
                mixed_clean_pushforward=bool(
                    mixed_clean_pushforward and epoch_unroll > 1
                ),
                pf_loss_weight=pf_loss_weight,
                start_singleton=bool(train_cfg.get("start_singleton", False)),
                mixed_parallel_noise=bool(mixed_parallel_noise),
                parallel_noise_eps=parallel_noise_eps,
                parallel_noise_a=parallel_noise_a,
                clip_replay=bool(clip_replay),
                replay_prob=epoch_replay_prob,
                replay_model=replay_model,
                clip_replay_horizon=clip_replay_horizon,
                trainable_chunk=trainable_chunk,
                history_frames=history_frames,
                parallel_causal_tf=parallel_causal_tf,
                position_loss_weights=position_loss_weights,
                train_loss_mode=train_loss_mode,
                channel_loss_weights=channel_loss_weights,
                rel_vp_alpha=rel_vp_alpha,
                rel_vp_beta=rel_vp_beta,
            )
            train_timing = _phase_timing_finish(train_timer)
            if clip_replay:
                # Snapshot θ_epoch for epoch+1 lagged replay generation.
                replay_model = sync_clip_replay_model(replay_model, model)
            eval_timer = _phase_timing_start(device)
            val_metrics = run_epoch(
                model,
                loaders[selection_split],
                device,
                optimizer=None,
                detach_memory=True,
                pin_memory=pin_memory,
                hist_noise_std=0.0,
                unroll_steps=1,
                ss_prob=0.0,
                detach_unroll=True,
                pushforward_last_only=False,
                ar_stream=False,
                mixed_clean_pushforward=False,
                mixed_parallel_noise=False,
                clip_replay=False,
                replay_prob=0.0,
                replay_model=None,
                clip_replay_horizon=None,
                trainable_chunk=trainable_chunk,
                history_frames=history_frames,
                parallel_causal_tf=parallel_causal_tf,
                position_loss_weights=position_loss_weights,
                train_loss_mode=train_loss_mode,
                channel_loss_weights=channel_loss_weights,
                rel_vp_alpha=rel_vp_alpha,
                rel_vp_beta=rel_vp_beta,
            )
            eval_timing = _phase_timing_finish(eval_timer)
            row: dict[str, Any] = {
                "epoch": epoch,
                **prefix_metrics("train_", train_metrics),
                **prefix_metrics("validation_", val_metrics),
                "selection_split": selection_split,
                "train_wall_s": float(train_timing["wall_s"]),
                "train_cuda_ms": float(train_timing["cuda_ms"]),
                "train_peak_alloc_mb": float(train_timing["peak_alloc_mb"]),
                "train_peak_reserved_mb": float(train_timing["peak_reserved_mb"]),
                "eval_wall_s": float(eval_timing["wall_s"]),
                "eval_cuda_ms": float(eval_timing["cuda_ms"]),
                "eval_peak_alloc_mb": float(eval_timing["peak_alloc_mb"]),
                "eval_peak_reserved_mb": float(eval_timing["peak_reserved_mb"]),
                "ss_prob_effective": ss_now,
                "train_unroll_steps": float(epoch_unroll),
                "train_k_max": float(epoch_k_max),
                "closed_loop_active": float(1.0 if in_closed_loop else 0.0),
                "pushforward_active": float(1.0 if in_pushforward else 0.0),
                "pushforward_last_only": float(
                    1.0 if (pushforward_last_only and epoch_unroll > 1) else 0.0
                ),
                "ar_stream": float(1.0 if (ar_stream and epoch_unroll > 1) else 0.0),
                "lr_spatial": float(applied_lrs.get("spatial", lr * spatial_mult)),
                "lr_temporal": float(applied_lrs.get("temporal", lr * temporal_mult)),
                "lr_spatial_mult": float(spatial_mult),
                "lr_temporal_mult": float(temporal_mult),
            }
            # Back-compat aliases used by older plot scripts.
            row["train_loss"] = float(train_metrics["loss"])
            row["validation_loss"] = float(val_metrics["loss"])

            do_rollout = validation_rollout_steps > 0 and (
                epoch == start_epoch or epoch % validation_rollout_every == 0
            )
            if do_rollout:
                roll_timer = _phase_timing_start(device)
                rollout_metrics = rollout_validation_split(
                    model,
                    traj_manifest,
                    normalizer,
                    history_frames=history_frames,
                    points=points,
                    subset_seed=subset_seed,
                    device=device,
                    max_steps=validation_rollout_steps,
                    max_trajs=validation_rollout_max_trajs,
                    parallel_causal_tf=parallel_causal_tf,
                    split=selection_split,
                )
                roll_timing = _phase_timing_finish(roll_timer)
                row.update(rollout_metrics)
                row["roll_wall_s"] = float(roll_timing["wall_s"])
                row["roll_cuda_ms"] = float(roll_timing["cuda_ms"])
                row["roll_peak_alloc_mb"] = float(roll_timing["peak_alloc_mb"])
                row["roll_peak_reserved_mb"] = float(roll_timing["peak_reserved_mb"])
            # Intentionally no carry-forward: sparse rollout must not fake a flat curve.

            row["epoch_seconds"] = time.perf_counter() - started
            if selection_metric_name not in row:
                raise RuntimeError(
                    f"selection metric {selection_metric_name!r} missing from epoch row; "
                    "enable validation_rollout_steps if selecting on rollout "
                    "(and use validation_rollout_every=1 when selecting on rollout)"
                )
            selection_value = float(row[selection_metric_name])
            row["selection_metric"] = selection_metric_name
            row["selection_value"] = selection_value
            history.append(row)
            write_json(output_dir / "history.json", history)
            write_history_csv(output_dir / "epoch_history.csv", history)
            print(json.dumps(row, sort_keys=True), flush=True)

            is_best = selection_value < best
            if is_best:
                best = selection_value
                best_epoch = epoch

            step_value = float(row["validation_loss"])
            is_best_step = step_value < best_step
            if is_best_step:
                best_step = step_value
                best_step_epoch = epoch

            rollout_raw = row.get("validation_rollout_relative_l2")
            is_best_rollout = False
            if rollout_raw is not None:
                rollout_value = float(rollout_raw)
                is_best_rollout = rollout_value < best_rollout
                if is_best_rollout:
                    best_rollout = rollout_value
                    best_rollout_epoch = epoch

            payload = build_training_checkpoint(
                model=model,
                optimizer=optimizer,
                normalizer=normalizer,
                run_manifest=run_manifest,
                epoch=epoch,
                best_validation_loss=float(row["validation_loss"]),
                best_epoch=best_epoch,
                history=history,
                experiment=experiment,
            )
            payload["best_selection_metric"] = best_selection_metric
            payload["best_selection_value"] = float(best)
            payload["selection_metric"] = selection_metric_name
            payload["selection_value"] = selection_value
            payload["validation_rollout_relative_l2"] = row.get(
                "validation_rollout_relative_l2"
            )
            payload["best_step_epoch"] = int(best_step_epoch)
            payload["best_step_validation_loss"] = (
                None if best_step == float("inf") else float(best_step)
            )
            payload["best_rollout_epoch"] = int(best_rollout_epoch)
            payload["best_rollout_relative_l2"] = (
                None if best_rollout == float("inf") else float(best_rollout)
            )
            save_training_checkpoint(output_dir / "checkpoint_last.pt", payload)
            # Primary best follows selection_metric (early-stop / default eval).
            if is_best:
                save_training_checkpoint(output_dir / "checkpoint_best.pt", payload)
            # Always keep dedicated step / rollout bests when those metrics improve.
            if is_best_step:
                save_training_checkpoint(output_dir / "checkpoint_best_step.pt", payload)
            if is_best_rollout:
                save_training_checkpoint(
                    output_dir / "checkpoint_best_rollout.pt", payload
                )
            if (
                epoch >= early_stop_min_epoch
                and epoch - best_epoch >= patience
            ):
                print(
                    json.dumps(
                        {
                            "status": "early_stopped",
                            "best_epoch": best_epoch,
                            "patience": patience,
                            "early_stop_min_epoch": early_stop_min_epoch,
                            "selection_metric": selection_metric_name,
                            "best_step_epoch": best_step_epoch,
                            "best_rollout_epoch": best_rollout_epoch,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break

    checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(
        model,
        loaders["test"],
        device,
        optimizer=None,
        detach_memory=True,
        pin_memory=pin_memory,
        hist_noise_std=0.0,
        unroll_steps=1,
        ss_prob=0.0,
        detach_unroll=True,
        pushforward_last_only=False,
        trainable_chunk=trainable_chunk,
        history_frames=history_frames,
        parallel_causal_tf=parallel_causal_tf,
        position_loss_weights=position_loss_weights,
        train_loss_mode=train_loss_mode,
        channel_loss_weights=channel_loss_weights,
        rel_vp_alpha=rel_vp_alpha,
        rel_vp_beta=rel_vp_beta,
    )
    summary = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_selection_metric": selection_metric_name,
        "best_selection_value": best,
        "best_validation_loss": best
        if selection_metric_name == "validation_loss"
        else float(
            min(history, key=lambda r: r["validation_loss"])["validation_loss"]
            if history
            else float("nan")
        ),
        "best_step_epoch": best_step_epoch,
        "best_step_validation_loss": (
            None if best_step == float("inf") else float(best_step)
        ),
        "best_rollout_epoch": best_rollout_epoch,
        "best_rollout_relative_l2": (
            None if best_rollout == float("inf") else float(best_rollout)
        ),
        "checkpoints": {
            "best": "checkpoint_best.pt",
            "best_step": "checkpoint_best_step.pt",
            "best_rollout": (
                "checkpoint_best_rollout.pt"
                if best_rollout != float("inf")
                else None
            ),
            "last": "checkpoint_last.pt",
        },
        "test_loss": test_metrics["loss"],
        "test_metrics": test_metrics,
        "epochs_completed": len(history),
    }
    write_json(output_dir / "summary.json", summary)
    for dataset in datasets.values():
        dataset.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
