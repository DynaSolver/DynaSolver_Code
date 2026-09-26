#!/usr/bin/env python3
"""Paper rollout relative-L2 curves at 6 keyframes (%% scale).

Matches the main RoboCFD-4D Base rollout table method set and keyframes
1 / 6 / 12 / 18 / 24 / 29. Y-axis is relative L2 in percent (×100).

Name map (paper display ← run id), 21 learned + Persistence + DynaSolver:
  DynaSolver ← ours_slice16_w1122
  GAOT ← gaot3d_v2  ·  GraphViT ← graphvit  ·  …
  (excludes GINO and older ours / deeponet / gaot3d / gino variants)

Outputs under <root>/figures/paper/k6/:
  labeled/   PNG@500dpi + PDF with Times-like labels and 4-col legend
  vector/    text-free PDF for draw.io
"""

from __future__ import annotations

import sys
import os

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "dynasolver"
for _p in (REPO_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.paper_curve_style import (
    PAPER_DPI,
    add_split_legend,
    apply_axis_text,
    ghost_style,
    make_panels,
    save_axes_only,
    save_figure,
    strip_axis_text,
    style_line,
    use_times,
)

DEFAULT_ROOT = Path(os.path.expandvars('${DYNASOLVER_DATA}/runs/compare/comprehensive_waterlily20/methods_kinematic'))

# Run-id order: hero first, then table baselines, persistence last as REF.
# Aligned with tab:rollout_comparison (GINO commented out there).
METHOD_ORDER = (
    "ours_slice16_w1122",
    "gnot",
    "gaot3d_v2",
    "cno",
    "p3d",
    "mgn_radius",
    "dpot",
    "abupt",
    "uno",
    "unet_res",
    "mspt",
    "gk_transformer",
    "dilresnet",
    "physgto",
    "gns_radius",
    "deeponet_official",
    "graphvit",
    "bsms",
    "fno",
    "linearno",
    "transolver",
    "mwt_res",
    "persistence",
)

DISPLAY = {
    "ours_slice16_w1122": "DynaSolver",
    "gnot": "GNOT",
    "gaot3d_v2": "GAOT",
    "cno": "CNO",
    "p3d": "P3D",
    "mgn_radius": "MGN",
    "dpot": "DPOT",
    "abupt": "AB-UPT",
    "uno": "UNO",
    "unet_res": "U-Net",
    "mspt": "MSPT",
    "gk_transformer": r"Transformer$_{gal}$",
    "dilresnet": "DilResNet",
    "physgto": "PhysGTO",
    "gns_radius": "GNS",
    "deeponet_official": "DeepONet",
    "graphvit": "GraphViT",
    "bsms": "BSMS",
    "fno": "FNO",
    "linearno": "LinearNO",
    "transolver": "Transolver",
    "mwt_res": "MWT",
    "persistence": "Persistence",
}

# 23 entries = hero + 21 ghosts + persistence; 4-col legend.
LEGEND_COLUMNS = (5, 6, 6, 6)

METRICS = (
    ("relative_l2_all", r"Relative $L_2$ (%)", "rollout_all_relL2"),
    ("relative_l2_velocity", r"Velocity relative $L_2$ (%)", "rollout_velocity_relL2"),
    ("relative_l2_pressure", r"Pressure relative $L_2$ (%)", "rollout_pressure_relL2"),
)

N_FRAMES = 29
# Match table keyframes (1-indexed AR steps).
KEYFRAMES = (1, 6, 12, 18, 24, 29)
HERO = "ours_slice16_w1122"
REF = "persistence"
DASH = (0, (6, 3))
# Table reports percent; json stores fractions in [0, 1+].
PCT = 100.0


def _legend(name: str) -> str:
    return DISPLAY.get(name, name)


def _load(root: Path, name: str) -> dict[str, list[float]]:
    payload = json.loads((root / name / "rollout_vp.json").read_text(encoding="utf-8"))
    out = {}
    for key, _ylabel, _stem in METRICS:
        curve = payload["per_frame"][key]
        if len(curve) != N_FRAMES:
            raise ValueError(f"{name} {key} has {len(curve)} frames, expected {N_FRAMES}")
        out[key] = [float(v) * PCT for v in curve]
    return out


def _plot_series(ax, curves, metric: str, frames: list[int], *, labeled: bool) -> None:
    ghost_i = 0
    for name in METHOD_ORDER:
        ys = [curves[name][metric][frame - 1] for frame in frames]
        label = _legend(name) if labeled else None
        if name == HERO:
            style_line(ax, frames, ys, "hero", label=label)
        elif name == REF:
            style_line(ax, frames, ys, "baseline", label=label, linestyle=DASH)
        else:
            style_line(ax, frames, ys, "ghost", label=label, **ghost_style(ghost_i))
            ghost_i += 1


def _new_axes(xlim=(0, 30)):
    fig, axes = make_panels(1, width=7.6, height=4.6)
    use_times()
    ax = axes[0]
    ax.set_xlim(*xlim)
    return fig, ax


def _draw_labeled(curves, metric: str, ylabel: str, frames: list[int], stem: Path) -> None:
    fig, ax = _new_axes()
    apply_axis_text(ax, "AR frame", ylabel)
    _plot_series(ax, curves, metric, frames, labeled=True)
    add_split_legend(ax, LEGEND_COLUMNS)
    pdf_path, png_path = save_figure(fig, stem, dpi=PAPER_DPI)
    plt.close(fig)
    print(f"wrote {pdf_path}", flush=True)
    print(f"wrote {png_path}", flush=True)


def _draw_vector(curves, metric: str, frames: list[int], stem: Path) -> None:
    fig, ax = _new_axes()
    strip_axis_text(ax)
    _plot_series(ax, curves, metric, frames, labeled=False)
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
    missing = [m for m in METHOD_ORDER if not (root / m / "rollout_vp.json").is_file()]
    if missing:
        raise SystemExit(f"missing rollout_vp.json: {missing}")
    curves = {name: _load(root, name) for name in METHOD_ORDER}
    frames = list(KEYFRAMES)

    paper = root / "figures" / "paper"
    labeled_dir = paper / "k6" / "labeled"
    vector_dir = paper / "k6" / "vector"

    for metric, ylabel, stem in METRICS:
        print(f"{metric} frames={frames} n_methods={len(METHOD_ORDER)}", flush=True)
        _draw_labeled(curves, metric, ylabel, frames, labeled_dir / stem)
        _draw_vector(curves, metric, frames, vector_dir / stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
