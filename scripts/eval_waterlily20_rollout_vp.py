#!/usr/bin/env python3
"""AR rollout eval: average + per-frame velocity/pressure relative L2.

Uses checkpoint_best.pt from --run-dir (stage rollbest as checkpoint_best.pt).
Pass --manifest to force the shared main-data manifest so a kinematic-wall
checkpoint can be compared with baselines on the same HDF5s.

Example:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python -u scripts/eval_waterlily20_rollout_vp.py \\
    --run-dir .../eval_ckpts/ours_kinematic_rollbest/seed_17 \\
    --manifest .../waterlily20_grid18x18x36_n11664_f30_temporal_manifest.json \\
    --output-dir .../compare/comprehensive_waterlily20/ours_kinematic_main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "dynasolver"
for _p in (REPO_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_provider.dynamic_cfd import (  # noqa: E402
    load_trajectory_manifest,
    resolve_point_indices,
    validate_trajectory_hdf5,
)
from data_provider.temporal_cfd import (  # noqa: E402
    causal_window_frame_ids,
    resolve_keyframe_timeline,
)
from dynamic_cfd.training import weighted_relative_l2  # noqa: E402
from scripts.eval_temporal_vs_persistence import load_trajectory_arrays  # noqa: E402
from scripts.load_checkpoint import load_temporal_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_best.pt",
        help="checkpoint filename inside --run-dir (default checkpoint_best.pt)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="override run_manifest.json; use main-data temporal manifest for fair compare",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--label",
        type=str,
        default="ours_kinematic_wall",
        help="label written into rollout_vp.json / fields index",
    )
    p.add_argument(
        "--points",
        type=int,
        default=0,
        help="override point count (0 = use run_manifest / config). "
        "Use H5 N (e.g. 93312) for full-grid no-downsample eval.",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-test-trajs", type=int, default=0, help="0 = all test")
    p.add_argument("--max-steps", type=int, default=0, help="0 = full trajectory")
    p.add_argument("--decimals", type=int, default=6)
    p.add_argument(
        "--save-fields",
        action="store_true",
        help=(
            "also dump per-traj NPZ under output-dir/fields/ with pred/gt states "
            "(normalized), xyz, masks, and the same per-step / mean relL2 metrics"
        ),
    )
    p.add_argument(
        "--robots",
        type=str,
        default="",
        help="comma-separated robot id prefixes (e.g. h1_2,h1,solo,a1). "
        "Empty = all test. h1_2 is matched before h1.",
    )
    p.add_argument(
        "--traj-ids",
        type=str,
        default="",
        help="comma-separated trajectory_ids to keep (exact match). Empty = all.",
    )
    return p.parse_args()


def _robot_of(tid: str) -> str:
    tid = str(tid)
    if tid.startswith("h1_2"):
        return "h1_2"
    if tid.startswith("h1_"):
        return "h1"
    if tid.startswith("solo"):
        return "solo"
    if tid.startswith("a1_"):
        return "a1"
    return tid.split("_", 1)[0]


def _rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    channel_slice: slice | None = None,
) -> float:
    if channel_slice is not None:
        pred = pred[..., channel_slice]
        target = target[..., channel_slice]
    return float(
        weighted_relative_l2(
            pred.unsqueeze(0),
            target.unsqueeze(0),
            mask.unsqueeze(0),
            weight.unsqueeze(0),
        ).item()
    )


@torch.no_grad()
def rollout_vp(
    model,
    traj: dict[str, Any],
    *,
    history_frames: int,
    device: torch.device,
    max_steps: int | None,
    keyframe_timeline: Sequence[int] | None,
    parallel_causal_tf: bool = False,
) -> dict[str, Any]:
    frames = int(traj["frames"])
    if keyframe_timeline is not None:
        timeline = tuple(int(t) for t in keyframe_timeline)
    else:
        timeline = resolve_keyframe_timeline(n_frames=frames)
    n_steps = len(timeline) - 1
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))
    if n_steps < 1:
        raise ValueError("no AR steps")

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
    surf_all = None
    if asym:
        if "surface_mask" not in traj:
            raise RuntimeError(f"{getattr(model, 'branch_mode', '?')} AR requires surface_mask")
        surf_all = torch.as_tensor(traj["surface_mask"], device=device, dtype=torch.bool)

    model.eval()
    all_list: list[float] = []
    vel_list: list[float] = []
    p_list: list[float] = []
    for seq_pos in range(n_steps):
        local_ids = causal_window_frame_ids(seq_pos, history_frames)
        times = [int(timeline[i]) for i in local_ids]
        static_times = [int(timeline[i + 1]) for i in local_ids]
        target_t = int(timeline[seq_pos + 1])
        t_len = len(times)
        state_win = torch.stack([ar[t] for t in times], dim=0).unsqueeze(0)
        query_win = query.unsqueeze(0).unsqueeze(0).expand(1, t_len, -1, -1).contiguous()
        static_win = torch.stack([static[t] for t in static_times], dim=0).unsqueeze(0)
        boundary_win = torch.stack([boundary[t] for t in times], dim=0).unsqueeze(0)
        surf_win = None
        if surf_all is not None:
            surf_win = torch.stack([surf_all[t] for t in times], dim=0).unsqueeze(0)
        # Match train rollout: parallel_causal models refine full window (not last-only).
        out = model.forward_window(
            query_win,
            static_win,
            state_win,
            boundary_win,
            memory_prev=memory,
            frame_ids=None if parallel_causal_tf else local_ids,
            parallel_causal_tf=parallel_causal_tf,
            surface_mask=surf_win,
        )
        pred = out["prediction"][0]
        memory = out["memory_new"]
        # Match train writeback: keep full pred (incl. surface / non-fluid).
        ar[target_t] = pred
        mask = valid[target_t] & fluid[target_t]
        weight = torch.where(mask, base_weight, torch.zeros_like(base_weight))
        target = gt[target_t]
        all_list.append(_rel_l2(pred, target, mask, weight))
        vel_list.append(_rel_l2(pred, target, mask, weight, slice(0, 3)))
        p_list.append(_rel_l2(pred, target, mask, weight, slice(3, 4)))

    return {
        "steps": n_steps,
        "per_step_relative_l2_all": all_list,
        "per_step_relative_l2_velocity": vel_list,
        "per_step_relative_l2_pressure": p_list,
        "relative_l2_all_mean": float(np.mean(all_list)),
        "relative_l2_velocity_mean": float(np.mean(vel_list)),
        "relative_l2_pressure_mean": float(np.mean(p_list)),
        "keyframe_timeline": list(timeline),
        # Full AR fields (normalized space) for qualitative / 3D vis.
        "pred_state": ar.detach().cpu().numpy().astype(np.float32),
        "gt_state": gt.detach().cpu().numpy().astype(np.float32),
        "query_xyz": query.detach().cpu().numpy().astype(np.float32),
        "valid": valid.detach().cpu().numpy().astype(np.bool_),
        "fluid": fluid.detach().cpu().numpy().astype(np.bool_),
        "base_weight": base_weight.detach().cpu().numpy().astype(np.float32),
    }


def _mean_pad(curves: list[list[float]]) -> list[float]:
    if not curves:
        return []
    n = max(len(c) for c in curves)
    out = []
    for i in range(n):
        vals = [c[i] for c in curves if i < len(c)]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device(args.device)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg, normalizer, run_manifest = load_temporal_model(
        args.run_dir, device, checkpoint=str(args.checkpoint)
    )
    ckpt_name = str(args.checkpoint)
    ckpt_path = args.run_dir / ckpt_name
    print(f"checkpoint={ckpt_path}", flush=True)
    history_frames = int(cfg["model"].get("history_frames", 3))
    points = int(run_manifest.get("points") or cfg["training"].get("points", 11664))
    if int(args.points) > 0:
        points = int(args.points)
    subset_seed = int(
        run_manifest.get("subset_seed") or cfg["training"].get("subset_seed", 20260803)
    )
    parallel_causal_tf = bool(cfg.get("training", {}).get("parallel_causal_tf", False))

    man_path = args.manifest
    if man_path is None:
        man_path = Path(str(run_manifest.get("manifest") or ""))
    if not man_path.is_file():
        raise SystemExit(f"manifest not found: {man_path}")
    print(f"manifest={man_path}", flush=True)
    print(
        f"run_dir={args.run_dir}  history_frames={history_frames}  points={points}"
        f"  parallel_causal_tf={parallel_causal_tf}",
        flush=True,
    )

    traj_manifest = load_trajectory_manifest(
        man_path,
        require_all_splits=False,  # waterlily20 is train/test only
        require_production_eligible=True,
        require_hashes=False,
        verify_hashes=False,
    )
    test_records = [r for r in traj_manifest.records if r.split == "test"]
    if str(args.robots).strip():
        robots = {x.strip() for x in str(args.robots).split(",") if x.strip()}

        def _keep(r) -> bool:
            return _robot_of(r.trajectory_id) in robots

        test_records = [r for r in test_records if _keep(r)]
        print(f"robot filter={sorted(robots)} → {len(test_records)} trajs", flush=True)
    if str(args.traj_ids).strip():
        keep = {x.strip() for x in str(args.traj_ids).split(",") if x.strip()}
        # targeted viz may include train-split cases (e.g. a1_u2/u8_yaw_p0)
        test_records = [
            r for r in traj_manifest.records if str(r.trajectory_id) in keep
        ]
        print(
            f"traj-ids filter={sorted(keep)} → {len(test_records)} trajs "
            f"(any split)",
            flush=True,
        )
    if args.max_test_trajs > 0:
        test_records = test_records[: args.max_test_trajs]
    print(f"test_trajs={len(test_records)}", flush=True)

    max_steps = None if args.max_steps <= 0 else int(args.max_steps)
    vel_curves: list[list[float]] = []
    p_curves: list[list[float]] = []
    all_curves: list[list[float]] = []
    per_traj: list[dict[str, Any]] = []
    infer_cuda_ms = 0.0
    use_cuda_time = device.type == "cuda"
    fields_dir: Path | None = None
    if args.save_fields:
        fields_dir = out_dir / "fields"
        fields_dir.mkdir(parents=True, exist_ok=True)
        print(f"save_fields -> {fields_dir}", flush=True)

    for ti, record in enumerate(test_records):
        print(f"  AR {ti+1}/{len(test_records)} {record.trajectory_id}", flush=True)
        shape = validate_trajectory_hdf5(record.hdf5_path, validate_values=False)
        # Full-grid eval (e.g. N=93312): keep native H5 order for high-res viz.
        # Subset eval keeps the stable nested permutation.
        if int(points) >= int(shape.points):
            indices = np.arange(int(shape.points), dtype=np.int64)
            if int(points) > int(shape.points):
                print(
                    f"  warn: --points={points} > H5 N={shape.points}; using full H5",
                    flush=True,
                )
        else:
            indices = resolve_point_indices(
                shape.points, points, seed=subset_seed, point_bank_id=record.point_bank_id
            )
        print(f"  points_used={len(indices)} (H5 N={shape.points})", flush=True)
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
        if use_cuda_time:
            torch.cuda.synchronize(device)
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        result = rollout_vp(
            model,
            traj,
            history_frames=history_frames,
            device=device,
            max_steps=max_steps,
            keyframe_timeline=timeline,
            parallel_causal_tf=parallel_causal_tf,
        )
        if use_cuda_time:
            ev1.record()
            torch.cuda.synchronize(device)
            infer_cuda_ms += float(ev0.elapsed_time(ev1))
        vel_curves.append(result["per_step_relative_l2_velocity"])
        p_curves.append(result["per_step_relative_l2_pressure"])
        all_curves.append(result["per_step_relative_l2_all"])
        traj_entry: dict[str, Any] = {
            "trajectory_id": record.trajectory_id,
            "hdf5_path": str(record.hdf5_path),
            "steps": result["steps"],
            "keyframe_timeline": result["keyframe_timeline"],
            "relative_l2_all_mean": result["relative_l2_all_mean"],
            "relative_l2_velocity_mean": result["relative_l2_velocity_mean"],
            "relative_l2_pressure_mean": result["relative_l2_pressure_mean"],
            "per_step_relative_l2_velocity": result["per_step_relative_l2_velocity"],
            "per_step_relative_l2_pressure": result["per_step_relative_l2_pressure"],
            "per_step_relative_l2_all": result["per_step_relative_l2_all"],
        }
        if fields_dir is not None:
            safe_id = str(record.trajectory_id).replace("/", "_")
            npz_path = fields_dir / f"{safe_id}.npz"
            np.savez_compressed(
                npz_path,
                pred_state=result["pred_state"],
                gt_state=result["gt_state"],
                query_xyz=result["query_xyz"],
                valid=result["valid"],
                fluid=result["fluid"],
                base_weight=result["base_weight"],
                keyframe_timeline=np.asarray(result["keyframe_timeline"], dtype=np.int32),
                per_step_relative_l2_all=np.asarray(
                    result["per_step_relative_l2_all"], dtype=np.float64
                ),
                per_step_relative_l2_velocity=np.asarray(
                    result["per_step_relative_l2_velocity"], dtype=np.float64
                ),
                per_step_relative_l2_pressure=np.asarray(
                    result["per_step_relative_l2_pressure"], dtype=np.float64
                ),
                relative_l2_all_mean=np.float64(result["relative_l2_all_mean"]),
                relative_l2_velocity_mean=np.float64(result["relative_l2_velocity_mean"]),
                relative_l2_pressure_mean=np.float64(result["relative_l2_pressure_mean"]),
            )
            traj_entry["fields_npz"] = str(npz_path)
        per_traj.append(traj_entry)

    nd = int(args.decimals)
    vel_mean_per_frame = _mean_pad(vel_curves)
    p_mean_per_frame = _mean_pad(p_curves)
    all_mean_per_frame = _mean_pad(all_curves)
    summary = {
        "label": str(args.label),
        "run_dir": str(args.run_dir),
        "manifest": str(man_path),
        "checkpoint": ckpt_name,
        "n_trajs": len(per_traj),
        "n_steps": len(vel_mean_per_frame),
        "points": int(points),
        "save_fields": bool(args.save_fields),
        "fields_dir": str(fields_dir) if fields_dir is not None else None,
        "average": {
            "relative_l2_all": float(np.mean([t["relative_l2_all_mean"] for t in per_traj])),
            "relative_l2_velocity": float(
                np.mean([t["relative_l2_velocity_mean"] for t in per_traj])
            ),
            "relative_l2_pressure": float(
                np.mean([t["relative_l2_pressure_mean"] for t in per_traj])
            ),
        },
        "per_frame": {
            "relative_l2_all": all_mean_per_frame,
            "relative_l2_velocity": vel_mean_per_frame,
            "relative_l2_pressure": p_mean_per_frame,
        },
        "trajectories": per_traj,
        "inference": {
            "cuda_ms": float(infer_cuda_ms),
            "cuda_s": float(infer_cuda_ms) / 1000.0,
            "scope": "sum of AR rollout forwards over all test trajectories",
        },
    }

    dest = out_dir / "rollout_vp.json"
    dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if fields_dir is not None:
        index_path = fields_dir / "index.json"
        index_path.write_text(
            json.dumps(
                {
                    "label": summary["label"],
                    "run_dir": summary["run_dir"],
                    "manifest": summary["manifest"],
                    "checkpoint": summary["checkpoint"],
                    "n_trajs": summary["n_trajs"],
                    "points": summary["points"],
                    "average": summary["average"],
                    "per_frame": summary["per_frame"],
                    "trajectories": [
                        {
                            "trajectory_id": t["trajectory_id"],
                            "fields_npz": t.get("fields_npz"),
                            "relative_l2_all_mean": t["relative_l2_all_mean"],
                            "relative_l2_velocity_mean": t["relative_l2_velocity_mean"],
                            "relative_l2_pressure_mean": t["relative_l2_pressure_mean"],
                            "per_step_relative_l2_all": t["per_step_relative_l2_all"],
                            "per_step_relative_l2_velocity": t["per_step_relative_l2_velocity"],
                            "per_step_relative_l2_pressure": t["per_step_relative_l2_pressure"],
                        }
                        for t in per_traj
                    ],
                    "note": (
                        "pred_state/gt_state are in normalizer space [T,N,4]=uvw+p; "
                        "frame 0 of pred is GT (AR starts after first keyframe)."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {index_path}", flush=True)

    avg = summary["average"]
    print("\n=== average (mean over traj means) ===", flush=True)
    print(f"relative_l2_all      = {avg['relative_l2_all']:.{nd}f}", flush=True)
    print(f"relative_l2_velocity = {avg['relative_l2_velocity']:.{nd}f}", flush=True)
    print(f"relative_l2_pressure = {avg['relative_l2_pressure']:.{nd}f}", flush=True)
    print(f"inference_cuda_s     = {infer_cuda_ms / 1000.0:.4f}", flush=True)
    print(f"inference_cuda_ms    = {infer_cuda_ms:.1f}", flush=True)
    print("\n=== per-frame (mean over trajs) velocity / pressure ===", flush=True)
    print(f"{'frame':>5}  {'vel_relL2':>12}  {'p_relL2':>12}  {'all_relL2':>12}", flush=True)
    for i, (v, pr, a) in enumerate(
        zip(vel_mean_per_frame, p_mean_per_frame, all_mean_per_frame), start=1
    ):
        print(f"{i:5d}  {v:.{nd}f}  {pr:.{nd}f}  {a:.{nd}f}", flush=True)
    print(f"\nwrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
