#!/usr/bin/env python3
"""Paper scatter: train wall (h) vs overall AR relative-L2 (%).

Task-specific: method list, train-time paths, json loading.
Style and export live in ``utils.paper_curve_style``.

Outputs under <root>/figures/paper/train_vs_overall/:
  labeled/   PNG@500dpi + PDF with Times-like labels and multi-col legend
  vector/    text-free PDF for draw.io
"""

from __future__ import annotations

import sys
import os

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "dynasolver"
for _p in (REPO_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.paper_curve_style import (
    METHOD_COLORS,
    PAPER_DPI,
    add_split_legend,
    apply_axis_text,
    ghost_style,
    make_panels,
    save_axes_only,
    save_figure,
    strip_axis_text,
    style_scatter,
    use_times,
)

DEFAULT_ROOT = Path(os.path.expandvars('${DYNASOLVER_DATA}/runs/compare/comprehensive_waterlily20/methods_kinematic'))
BASELINE_RUNS = Path(
    "${DYNASOLVER_DATA}/external_baselines/"
    "waterlily20_grid18x18x36_train_v1/runs/final_baselines"
)

# Ours = s16 + loss [1,1,1,2] (hero). Persistence = horizontal reference (no train wall).
# Name map matches paper k6 / attached run-id list (22 baselines + 1 Ours).
HERO = "ours_slice16_w1122"
REF = "persistence"

# (eval_dir, display_label, train_arm_or_None)
# train_arm=None → Ours path table or no train (persistence).
METHODS: tuple[tuple[str, str, str | None], ...] = (
    (HERO, "Ours", None),
    ("linearno", "LinearNO", "linearno"),
    ("mspt", "MSPT", "mspt"),
    ("physgto", "PhysGTO", "physgto"),
    ("gino_v2", "GINO", "gino_v2"),
    ("gaot3d_v2", "GAOT3D", "gaot3d_v2"),
    ("abupt", "AB-UPT", "abupt"),
    ("gk_transformer", "GK-Transformer", "gk_transformer"),
    ("deeponet_official", "DeepONet", "deeponet_official"),
    ("transolver", "Transolver", "transolver"),
    ("bsms", "BSMS", "bsms"),
    ("fno", "FNO", "fno"),
    ("dpot", "DPOT", "dpot"),
    ("mgn_radius", "MGN", "mgn_radius"),
    ("gns_radius", "GNS", "gns_radius"),
    ("unet_res", "U-Net", "unet_res"),
    ("cno", "CNO", "cno"),
    ("uno", "UNO", "uno"),
    ("mwt_res", "MWT", "mwt_res"),
    ("p3d", "P3D", "p3d"),
    ("dilresnet", "DilResNet", "dilresnet"),
    ("gnot", "GNOT", "gnot"),
    (REF, "Persistence", None),
)

OURS_TRAIN_CSV = Path(
    "${DYNASOLVER_DATA}/runs/"
    "ablations_loss_s16_kinematic/mse_ch_w1122/"
    "20260923_061942_from_scratch_n11664/seed_17/epoch_history.csv"
)

# Headerless MGN CSV column layout (matches dsb trainers).
_MGN_COLS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_rollout_relative_l2",
    "validation_rollout_relative_l2_final",
    "validation_rollout_relL2_at_1",
    "validation_rollout_relL2_at_4",
    "validation_rollout_relL2_at_10",
    "validation_rollout_relL2_at_15",
    "validation_rollout_relL2_at_25",
    "rollout_steps",
    "rollout_trajs",
    "train_seconds",
    "val_seconds",
    "epoch_seconds",
    "gpu_peak_MiB",
    "gpu_reserved_MiB",
    "samples",
)

# Legend: hero + 21 ghosts + persistence = 23 → (7, 8, 8).
LEGEND_COLUMNS = (7, 8, 8)
STEM = "train_wall_vs_overall_relL2"
DASH = (0, (6, 3))


def _load_overall_pct(root: Path, name: str) -> float:
    payload = json.loads((root / name / "rollout_vp.json").read_text(encoding="utf-8"))
    return 100.0 * float(payload["average"]["relative_l2_all"])


def _read_baseline_rows(arm: str) -> list[dict[str, str]]:
    csvp = BASELINE_RUNS / arm / "epoch_history.csv"
    text = csvp.read_text(encoding="utf-8").splitlines()
    if not text:
        raise FileNotFoundError(csvp)
    if text[0].startswith("epoch"):
        return list(csv.DictReader(csvp.open(encoding="utf-8")))
    rows: list[dict[str, str]] = []
    for line in text:
        parts = line.split(",")
        if len(parts) < 13:
            continue
        while len(parts) < len(_MGN_COLS):
            parts.append("")
        rows.append(dict(zip(_MGN_COLS, parts)))
    return rows


def _train_wall_hours(eval_name: str, train_arm: str | None) -> float | None:
    if eval_name == REF:
        return None
    if eval_name == HERO:
        rows = list(csv.DictReader(OURS_TRAIN_CSV.open(encoding="utf-8")))
        return sum(float(r["train_wall_s"]) for r in rows) / 3600.0
    assert train_arm is not None
    rows = _read_baseline_rows(train_arm)
    total = sum(
        float(r["train_seconds"]) for r in rows if r.get("train_seconds") not in (None, "")
    )
    # mgn_radius CSV is ep99–200 only → scale to 200 epochs.
    if train_arm == "mgn_radius" and rows:
        total = total / len(rows) * 200.0
    return total / 3600.0


def _collect(root: Path) -> list[dict]:
    points: list[dict] = []
    for eval_name, label, train_arm in METHODS:
        overall = _load_overall_pct(root, eval_name)
        hours = _train_wall_hours(eval_name, train_arm)
        points.append(
            {
                "eval": eval_name,
                "label": label,
                "hours": hours,
                "overall": overall,
            }
        )
    return points


def _plot_points(ax, points: list[dict], *, labeled: bool) -> None:
    ghost_i = 0
    for pt in points:
        name = pt["eval"]
        label = pt["label"] if labeled else None
        if name == REF:
            # Horizontal reference: no train wall.
            ax.axhline(
                pt["overall"],
                color=METHOD_COLORS["transolver"],
                linestyle=DASH,
                linewidth=1.35,
                alpha=0.85,
                zorder=2.5,
                label=label,
            )
            continue
        assert pt["hours"] is not None
        if name == HERO:
            style_scatter(ax, [pt["hours"]], [pt["overall"]], "hero", label=label)
        else:
            style_scatter(
                ax,
                [pt["hours"]],
                [pt["overall"]],
                "ghost",
                label=label,
                **ghost_style(ghost_i, alpha=0.55),
            )
            ghost_i += 1


def _new_axes(xlim: tuple[float, float], ylim: tuple[float, float]):
    fig, axes = make_panels(1, width=7.6, height=4.6)
    use_times()
    ax = axes[0]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return fig, ax


def _limits(points: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [p["hours"] for p in points if p["hours"] is not None]
    ys = [p["overall"] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = max(x1 - x0, 1.0)
    dy = max(y1 - y0, 1.0)
    return (x0 - 0.05 * dx, x1 + 0.08 * dx), (y0 - 0.08 * dy, y1 + 0.08 * dy)


def _draw_labeled(points: list[dict], stem: Path) -> None:
    """Labeled panel with legend ordered exactly as METHODS."""
    xlim, ylim = _limits(points)
    fig, ax = _new_axes(xlim, ylim)
    apply_axis_text(ax, "Training time (h)", "Overall relative L2 (%)")
    _plot_points(ax, points, labeled=True)

    raw_h, raw_l = ax.get_legend_handles_labels()
    by_label = {lab: hand for hand, lab in zip(raw_h, raw_l)}
    if "Persistence" not in by_label:
        by_label["Persistence"] = Line2D(
            [0],
            [0],
            color=METHOD_COLORS["transolver"],
            linestyle=DASH,
            linewidth=1.35,
        )
    ordered_labels = [lab for _, lab, _ in METHODS]
    missing = [lab for lab in ordered_labels if lab not in by_label]
    if missing:
        raise KeyError(f"missing legend labels: {missing}")
    handles = [by_label[lab] for lab in ordered_labels]
    labels = list(ordered_labels)

    def _ordered(_ax=ax):
        return handles, labels

    ax.get_legend_handles_labels = _ordered  # type: ignore[method-assign]
    if sum(LEGEND_COLUMNS) != len(handles):
        raise ValueError(
            f"column_sizes sum {sum(LEGEND_COLUMNS)} != {len(handles)} entries"
        )
    add_split_legend(ax, LEGEND_COLUMNS)
    pdf_path, png_path = save_figure(fig, stem, dpi=PAPER_DPI)
    plt.close(fig)
    print(f"wrote {pdf_path}", flush=True)
    print(f"wrote {png_path}", flush=True)


def _draw_vector(points: list[dict], stem: Path) -> None:
    xlim, ylim = _limits(points)
    fig, ax = _new_axes(xlim, ylim)
    strip_axis_text(ax)
    _plot_points(ax, points, labeled=False)
    path = save_axes_only(fig, stem)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    points = _collect(root)
    for pt in points:
        h = "—" if pt["hours"] is None else f"{pt['hours']:.2f}h"
        print(f"  {pt['label']:16s}  train={h:>8s}  overall={pt['overall']:.2f}%", flush=True)

    paper = root / "figures" / "paper" / "train_vs_overall"
    labeled_dir = paper / "labeled"
    vector_dir = paper / "vector"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    _draw_labeled(points, labeled_dir / STEM)
    _draw_vector(points, vector_dir / STEM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
