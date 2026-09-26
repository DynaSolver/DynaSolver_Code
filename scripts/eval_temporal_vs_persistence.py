#!/usr/bin/env python3
"""Evaluate Temporal Transolver vs Persistence: one-step + full AR + keyframe vis.

Persistence baseline: predict next state = previous state (no dynamics).
For one-step: MSE(state_t, state_{t+1}).
For AR rollout: hold the initial GT state for all future frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "dynasolver"
for _p in (REPO_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_provider.dynamic_cfd import (
    DynamicStateNormalizer,
    load_trajectory_manifest,
    resolve_point_indices,
    validate_trajectory_hdf5,
)
from data_provider.temporal_cfd import TemporalCFDWindowDataset, causal_window_frame_ids
from dynamic_cfd.training import weighted_masked_mse, weighted_relative_l2
from models.TemporalTransolver import Model


CHANNEL_NAMES = ("Ux", "Uy", "Uz", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Temporal seed_* run directory")
    parser.add_argument("--config", type=Path, help="yaml config; default from run_manifest")
    parser.add_argument("--checkpoint", type=Path, help="default: checkpoint_best.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--key-frames",
        type=int,
        nargs="+",
        default=[20, 60, 100, 140, 180],
        help="target absolute frame indices to visualize (clamped to available)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <run-dir>/analysis_vs_persistence",
    )
    parser.add_argument("--max-test-trajectories", type=int, default=0, help="0 = all test trajs")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_model_args(cfg_model: dict) -> Namespace:
    use_long = bool(cfg_model.get("use_long_memory", True))
    placement = str(
        cfg_model.get(
            "long_memory_placement",
            "per_layer" if use_long else "none",
        )
    )
    return Namespace(
        model="TemporalTransolver",
        fun_dim=int(cfg_model.get("fun_dim", 11)),
        space_dim=int(cfg_model.get("space_dim", 3)),
        state_dim=int(cfg_model["state_dim"]),
        boundary_dim=int(cfg_model.get("boundary_dim", 8)),
        n_hidden=int(cfg_model.get("n_hidden", 256)),
        n_heads=int(cfg_model.get("n_heads", 8)),
        n_layers=int(cfg_model.get("n_layers", 8)),
        mlp_ratio=int(cfg_model.get("mlp_ratio", 2)),
        out_dim=int(cfg_model["state_dim"]),
        slice_num=int(cfg_model.get("slice_num", 32)),
        dropout=float(cfg_model.get("dropout", 0.0)),
        act=str(cfg_model.get("act", "gelu")),
        geotype=str(cfg_model.get("geotype", "unstructured")),
        shapelist=None,
        checkpoint=False,
        unified_pos=False,
        window_frames=int(cfg_model.get("window_frames", 4)),
        use_long_memory=use_long,
        long_memory_placement=placement,
        mid_long_after_layer=int(cfg_model.get("mid_long_after_layer", 4)),
        use_short_window=bool(cfg_model.get("use_short_window", True)),
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


def encode_state(
    normalizer: DynamicStateNormalizer,
    raw: np.ndarray,
    mask: np.ndarray,
    speed: float | None = None,
) -> np.ndarray:
    """Match TemporalCFDWindowDataset: per-traj U_inf via u_ref_m_per_s=speed."""
    out = np.zeros((raw.shape[0], normalizer.state_dim), dtype=np.float32)
    if np.any(mask):
        out[mask] = normalizer.encode(raw[mask], u_ref_m_per_s=speed)
    return out


@torch.no_grad()
def evaluate_one_step(
    model: torch.nn.Module | None,
    dataset: TemporalCFDWindowDataset,
    device: torch.device,
    *,
    mode: str,
) -> dict[str, Any]:
    """mode: 'model' or 'persistence' (pred = last window state)."""
    mse_sum = 0.0
    rel_sum = 0.0
    pers_mse_sum = 0.0
    pers_rel_sum = 0.0
    n = 0
    if model is not None:
        model.eval()
    for index in range(len(dataset)):
        sample = dataset[index]
        state = torch.as_tensor(sample["state"], device=device).unsqueeze(0)
        query = torch.as_tensor(sample["query_xyz"], device=device).unsqueeze(0)
        static = torch.as_tensor(sample["static_fx"], device=device).unsqueeze(0)
        boundary = torch.as_tensor(sample["boundary_feat"], device=device).unsqueeze(0)
        target = torch.as_tensor(sample["target_state"], device=device).unsqueeze(0)
        mask = torch.as_tensor(sample["loss_mask"], device=device).unsqueeze(0)
        weight = torch.as_tensor(sample["loss_weight"], device=device).unsqueeze(0)

        if mode == "persistence":
            pred = state[:, -1]
        else:
            assert model is not None
            pred = model(query, static, state, boundary)

        mse = float(weighted_masked_mse(pred, target, mask, weight))
        rel = float(weighted_relative_l2(pred, target, mask, weight).item())
        pers_mse = float(weighted_masked_mse(state[:, -1], target, mask, weight))
        pers_rel = float(weighted_relative_l2(state[:, -1], target, mask, weight).item())

        mse_sum += mse
        rel_sum += rel
        pers_mse_sum += pers_mse
        pers_rel_sum += pers_rel
        n += 1

    return {
        "samples": n,
        "one_step_mse": mse_sum / n,
        "one_step_relative_l2": rel_sum / n,
        "persistence_one_step_mse": pers_mse_sum / n,
        "persistence_one_step_relative_l2": pers_rel_sum / n,
        "mse_vs_persistence": (pers_mse_sum / n) / max(mse_sum / n, 1e-30),
        "rel_l2_vs_persistence": (pers_rel_sum / n) / max(rel_sum / n, 1e-30),
    }


def load_trajectory_arrays(
    hdf5_path: Path,
    *,
    point_indices: np.ndarray,
    normalizer: DynamicStateNormalizer,
    speed: float | None,
) -> dict[str, Any]:
    # h5py fancy indexing requires strictly increasing indices.
    point_indices = np.asarray(point_indices, dtype=np.int64)
    order = np.argsort(point_indices)
    sorted_indices = point_indices[order]
    with h5py.File(hdf5_path, "r") as handle:
        points = np.asarray(handle["points_world_m"], dtype=np.float32)[sorted_indices]
        query = np.asarray(handle["query_xyz_normalized"], dtype=np.float32)[sorted_indices]
        base_weight = np.asarray(handle["loss_weight"], dtype=np.float32)[sorted_indices]
        frames = int(handle["state_raw"].shape[0])
        static = np.asarray(handle["static_fx"][:, sorted_indices], dtype=np.float32)
        b_name = "boundary_feat_kf20" if "boundary_feat_kf20" in handle else "boundary_feat"
        boundary = np.asarray(handle[b_name][:, sorted_indices], dtype=np.float32)
        valid = np.asarray(handle["valid_mask"][:, sorted_indices], dtype=bool)
        fluid = np.asarray(handle["fluid_mask"][:, sorted_indices], dtype=bool)
        raw = np.asarray(handle["state_raw"][:, sorted_indices], dtype=np.float32)
        times = np.asarray(handle["times"], dtype=np.float64)
        surface_mask = None
        if "surface_mask" in handle:
            surface_mask = np.asarray(
                handle["surface_mask"][:, sorted_indices], dtype=bool
            )

    # Restore original subset order so it matches training/eval subsets.
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    points = points[inv]
    query = query[inv]
    base_weight = base_weight[inv]
    static = static[:, inv]
    boundary = boundary[:, inv]
    valid = valid[:, inv]
    fluid = fluid[:, inv]
    raw = raw[:, inv]
    if surface_mask is not None:
        surface_mask = surface_mask[:, inv]

    state = np.stack(
        [encode_state(normalizer, raw[t], valid[t] & fluid[t], speed) for t in range(frames)],
        axis=0,
    )
    out = {
        "points": points,
        "query_xyz": query,
        "base_weight": base_weight,
        "static_fx": static,
        "boundary_feat": boundary,
        "valid": valid,
        "fluid": fluid,
        "state": state,
        "raw": raw,
        "times": times,
        "frames": frames,
    }
    if surface_mask is not None:
        out["surface_mask"] = surface_mask
    return out


@torch.no_grad()
def rollout_temporal_and_persistence(
    model: Model,
    traj: dict[str, Any],
    *,
    history_frames: int,
    device: torch.device,
) -> dict[str, Any]:
    """AR from frame 0 with growing-prefix then slide; persistence holds GT[0]."""
    model.eval()
    window = history_frames + 1
    frames = traj["frames"]
    first_t = 0
    last_t = frames - 2
    if last_t < first_t:
        raise ValueError("trajectory too short for temporal window")

    gt_states = torch.as_tensor(traj["state"], device=device)
    query_base = torch.as_tensor(traj["query_xyz"], device=device)
    static = torch.as_tensor(traj["static_fx"], device=device)
    boundary = torch.as_tensor(traj["boundary_feat"], device=device)
    base_weight = torch.as_tensor(traj["base_weight"], device=device)

    # predicted state buffer for AR (normalized)
    ar_state = gt_states.clone()
    pers_state = gt_states.clone()
    held = gt_states[first_t].clone()

    model_mse: list[float] = []
    model_rel: list[float] = []
    pers_mse: list[float] = []
    pers_rel: list[float] = []
    target_frames: list[int] = []

    memory = None
    for current_t in range(first_t, last_t + 1):
        target_t = current_t + 1
        times = causal_window_frame_ids(current_t, history_frames)
        static_times = [t + 1 for t in times]
        t_len = len(times)
        state_win = torch.stack([ar_state[t] for t in times], dim=0).unsqueeze(0)
        query_win = query_base.unsqueeze(0).unsqueeze(0).expand(1, t_len, -1, -1).contiguous()
        static_win = torch.stack([static[t] for t in static_times], dim=0).unsqueeze(0)
        boundary_win = torch.stack([boundary[t] for t in times], dim=0).unsqueeze(0)

        out = model.forward_window(query_win, static_win, state_win, boundary_win, memory_prev=memory)
        pred = out["prediction"][0]
        memory = out["memory_new"]
        ar_state[target_t] = pred

        # persistence: hold initial GT at first_t for all future targets
        pers_state[target_t] = held

        mask = torch.as_tensor(
            traj["valid"][target_t] & traj["fluid"][target_t], device=device
        )
        weight = torch.where(mask, base_weight, torch.zeros_like(base_weight))
        target = gt_states[target_t]
        mse_m = float(
            weighted_masked_mse(pred.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0), weight.unsqueeze(0))
        )
        rel_m = float(
            weighted_relative_l2(pred.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0), weight.unsqueeze(0)).item()
        )
        mse_p = float(
            weighted_masked_mse(
                held.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0), weight.unsqueeze(0)
            )
        )
        rel_p = float(
            weighted_relative_l2(
                held.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0), weight.unsqueeze(0)
            ).item()
        )
        model_mse.append(mse_m)
        model_rel.append(rel_m)
        pers_mse.append(mse_p)
        pers_rel.append(rel_p)
        target_frames.append(target_t)

    return {
        "target_frames": target_frames,
        "model_per_step_mse": model_mse,
        "model_per_step_relative_l2": model_rel,
        "persistence_per_step_mse": pers_mse,
        "persistence_per_step_relative_l2": pers_rel,
        "model_relative_l2_mean": float(np.mean(model_rel)),
        "model_relative_l2_final": float(model_rel[-1]),
        "persistence_relative_l2_mean": float(np.mean(pers_rel)),
        "persistence_relative_l2_final": float(pers_rel[-1]),
        "ar_state": ar_state.detach().cpu().numpy(),
        "pers_state": pers_state.detach().cpu().numpy(),
        "gt_state": traj["state"],
    }


def speed_field(state: np.ndarray) -> np.ndarray:
    return np.linalg.norm(state[..., :3], axis=-1)


def pick_key_frames(available: list[int], requested: list[int]) -> list[int]:
    available_arr = np.asarray(available, dtype=np.int64)
    chosen: list[int] = []
    for req in requested:
        idx = int(np.argmin(np.abs(available_arr - req)))
        frame = int(available_arr[idx])
        if frame not in chosen:
            chosen.append(frame)
    return chosen


def plot_rollout_curves(
    result: dict[str, Any],
    times: np.ndarray,
    output: Path,
    title: str,
) -> None:
    frames = result["target_frames"]
    t = times[frames]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    axes[0].plot(t, result["model_per_step_mse"], label="Temporal", color="#1f4e79", linewidth=1.6)
    axes[0].plot(
        t,
        result["persistence_per_step_mse"],
        label="Persistence",
        color="#9a6500",
        linestyle="--",
        linewidth=1.4,
    )
    axes[0].set_title(f"{title} — AR MSE")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("weighted masked MSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(t, result["model_per_step_relative_l2"], label="Temporal", color="#1f4e79", linewidth=1.6)
    axes[1].plot(
        t,
        result["persistence_per_step_relative_l2"],
        label="Persistence",
        color="#9a6500",
        linestyle="--",
        linewidth=1.4,
    )
    axes[1].set_title(f"{title} — AR relative L2")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("relative L2")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_keyframe_panels(
    *,
    points: np.ndarray,
    gt_state: np.ndarray,
    pred_state: np.ndarray,
    pers_state: np.ndarray,
    valid: np.ndarray,
    fluid: np.ndarray,
    frame: int,
    output: Path,
    title: str,
) -> None:
    mask = valid[frame] & fluid[frame]
    xy = points[mask]
    gt = speed_field(gt_state[frame])[mask]
    pred = speed_field(pred_state[frame])[mask]
    pers = speed_field(pers_state[frame])[mask]
    err_m = np.abs(pred - gt)
    err_p = np.abs(pers - gt)

    common = float(np.percentile(np.concatenate([gt, pred, pers]), 99)) if gt.size else 1.0
    err_max = float(np.percentile(np.concatenate([err_m, err_p]), 99)) if err_m.size else 1.0
    common = max(common, 1e-8)
    err_max = max(err_max, 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True, sharex=True, sharey=True)
    panels = (
        (axes[0, 0], gt, "GT |U|", common, "viridis"),
        (axes[0, 1], pred, "Temporal |U|", common, "viridis"),
        (axes[0, 2], err_m, "Temporal |error|", err_max, "magma"),
        (axes[1, 0], gt, "GT |U|", common, "viridis"),
        (axes[1, 1], pers, "Persistence |U|", common, "viridis"),
        (axes[1, 2], err_p, "Persistence |error|", err_max, "magma"),
    )
    for axis, values, label, vmax, cmap in panels:
        sc = axis.scatter(
            xy[:, 0],
            xy[:, 2],
            c=values,
            s=3.0,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        axis.set_title(label)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("z [m]")
        fig.colorbar(sc, ax=axis, shrink=0.82)
    fig.suptitle(title, fontsize=12)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_one_step_bars(metrics: dict[str, Any], output: Path, run_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    labels = ["Temporal", "Persistence"]
    mse = [metrics["one_step_mse"], metrics["persistence_one_step_mse"]]
    rel = [metrics["one_step_relative_l2"], metrics["persistence_one_step_relative_l2"]]
    axes[0].bar(labels, mse, color=["#1f4e79", "#9a6500"])
    axes[0].set_title(f"{run_label} — one-step MSE")
    axes[0].set_ylabel("weighted masked MSE")
    axes[1].bar(labels, rel, color=["#1f4e79", "#9a6500"])
    axes[1].set_title(f"{run_label} — one-step relative L2")
    axes[1].set_ylabel("relative L2")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config_path = Path(args.config or manifest_run["config"]).resolve()
    cfg = load_yaml(config_path)
    training = cfg["training"]
    model_cfg = dict(cfg["model"])
    # Prefer recorded model settings from the run
    if isinstance(manifest_run.get("model"), dict):
        model_cfg.update(manifest_run["model"])

    checkpoint_path = (args.checkpoint or (run_dir / "checkpoint_best.pt")).resolve()
    output_dir = (args.output_dir or (run_dir / "analysis_vs_persistence")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    history_frames = int(manifest_run.get("history_frames", model_cfg.get("window_frames", 4) - 1))
    points = int(manifest_run.get("points", training.get("points", 3200)))
    subset_seed = int(manifest_run.get("subset_seed", training.get("subset_seed", 20260803)))
    manifest_path = Path(manifest_run["manifest"]).resolve()

    normalizer = DynamicStateNormalizer.from_json(run_dir / "normalizer.json")
    model = Model(make_model_args(model_cfg)).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    test_ds = TemporalCFDWindowDataset(
        manifest_path,
        "test",
        normalizer,
        history_frames=history_frames,
        subset_points=points,
        subset_seed=subset_seed,
    )
    one_step = evaluate_one_step(model, test_ds, device, mode="model")
    # persistence metrics already embedded in evaluate_one_step
    plot_one_step_bars(one_step, figures / "one_step_vs_persistence.png", run_dir.parent.name)
    test_ds.close()

    traj_manifest = load_trajectory_manifest(
        manifest_path,
        require_production_eligible=False,
        require_hashes=False,
        verify_hashes=False,
    )
    test_records = traj_manifest.records_for_split("test", require_production_eligible=False)
    if args.max_test_trajectories > 0:
        test_records = test_records[: args.max_test_trajectories]

    rollout_summaries: list[dict[str, Any]] = []
    for record in test_records:
        shape = validate_trajectory_hdf5(record.hdf5_path, validate_values=False)
        indices = resolve_point_indices(
            shape.points,
            points,
            seed=subset_seed,
            point_bank_id=record.point_bank_id,
        )
        traj = load_trajectory_arrays(
            record.hdf5_path,
            point_indices=indices,
            normalizer=normalizer,
            speed=record.speed_m_per_s,
        )
        result = rollout_temporal_and_persistence(
            model,
            traj,
            history_frames=history_frames,
            device=device,
        )
        plot_rollout_curves(
            result,
            traj["times"],
            figures / f"{record.trajectory_id}_ar_curves.png",
            record.trajectory_id,
        )
        key_frames = pick_key_frames(result["target_frames"], list(args.key_frames))
        for frame in key_frames:
            plot_keyframe_panels(
                points=traj["points"],
                gt_state=result["gt_state"],
                pred_state=result["ar_state"],
                pers_state=result["pers_state"],
                valid=traj["valid"],
                fluid=traj["fluid"],
                frame=frame,
                output=figures / f"{record.trajectory_id}_frame{frame:03d}_gt_pred_error.png",
                title=f"{record.trajectory_id} · frame {frame} · t={traj['times'][frame]:.3f}s",
            )
        rollout_summaries.append(
            {
                "trajectory_id": record.trajectory_id,
                "steps": len(result["target_frames"]),
                "model_relative_l2_mean": result["model_relative_l2_mean"],
                "model_relative_l2_final": result["model_relative_l2_final"],
                "persistence_relative_l2_mean": result["persistence_relative_l2_mean"],
                "persistence_relative_l2_final": result["persistence_relative_l2_final"],
                "key_frames": key_frames,
                "model_mse_mean": float(np.mean(result["model_per_step_mse"])),
                "persistence_mse_mean": float(np.mean(result["persistence_per_step_mse"])),
            }
        )
        print(
            json.dumps(
                {
                    "trajectory_id": record.trajectory_id,
                    "model_relL2_mean": result["model_relative_l2_mean"],
                    "pers_relL2_mean": result["persistence_relative_l2_mean"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "one_step": one_step,
        "rollout_test": {
            "trajectories": rollout_summaries,
            "model_relative_l2_mean": float(
                np.mean([row["model_relative_l2_mean"] for row in rollout_summaries])
            ),
            "persistence_relative_l2_mean": float(
                np.mean([row["persistence_relative_l2_mean"] for row in rollout_summaries])
            ),
        },
        "figures_dir": str(figures),
        "note": (
            "Persistence predicts next=previous. One-step uses teacher-forced windows. "
            "AR rollout feeds predicted states into the Temporal window; "
            "persistence AR holds the GT state at the first prediction start frame."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
