#!/usr/bin/env python3
"""Paper curve: slice size S∈{16,32,64,128} selection (mini8 full-test48).

Shows how S=16 (slice16) is chosen: All/Vel/P relL2 (%) vs slice_num.
Style matches ``plot_paper_window_select.py`` / ``utils.paper_curve_style``.

Outputs under figures/paper/slice_select/:
  labeled/   PNG@500dpi + PDF with Times-like labels
  vector/    text-free PDF for draw.io
"""

from __future__ import annotations

import sys
import os

import argparse
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
    add_legend,
    apply_axis_text,
    make_panels,
    save_axes_only,
    save_figure,
    set_categorical_x,
    strip_axis_text,
    style_line,
    style_scatter,
    use_times,
)

# mini8 kinematic full-test48; relL2 ×100.
# (S, arm, Vel%, P%, All%, infer_s)
SLICE_ROWS: tuple[tuple[int, str, float, float, float, float], ...] = (
    (16, "slice16", 59.30, 43.30, 56.39, 102.4),
    (32, "ours_s32", 59.34, 43.65, 56.47, 56.6),
    (64, "slice64", 59.95, 44.11, 57.06, 88.1),
    (128, "slice128", 59.83, 45.18, 57.09, 70.7),
)

SELECTED_S = 16
STEM = "slice_S_relL2_select"

DEFAULT_OUT = Path(os.path.expandvars('./figures/paper/slice_select'))


def _new_axes(ylim: tuple[float, float], n_cats: int):
    fig, axes = make_panels(1, width=7.6, height=4.6)
    use_times()
    ax = axes[0]
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.35, n_cats - 0.65)
    return fig, ax


def _ylim(rows: list[tuple[int, str, float, float, float, float]]):
    ys = [v for r in rows for v in (r[2], r[3], r[4])]
    y0, y1 = min(ys), max(ys)
    dy = max(y1 - y0, 1.0)
    return (y0 - 0.10 * dy, y1 + 0.12 * dy)


def _plot(ax, rows: list[tuple[int, str, float, float, float, float]], *, labeled: bool) -> None:
    labels = [str(r[0]) for r in rows]
    xs = set_categorical_x(ax, labels)
    vel = [r[2] for r in rows]
    p = [r[3] for r in rows]
    all_ = [r[4] for r in rows]
    sel_i = next(i for i, r in enumerate(rows) if r[0] == SELECTED_S)

    style_line(
        ax,
        xs,
        all_,
        "hero",
        label="Overall" if labeled else None,
        marker="o",
        markersize=5.5,
    )
    style_line(
        ax,
        xs,
        vel,
        "ablation",
        label="Velocity" if labeled else None,
        marker="^",
        markersize=5.2,
    )
    style_line(
        ax,
        xs,
        p,
        "baseline",
        label="Pressure" if labeled else None,
        marker="s",
        markersize=4.8,
        linestyle=(0, (6, 3)),
    )
    style_scatter(
        ax,
        [xs[sel_i]],
        [all_[sel_i]],
        "hero",
        label=f"Selected ($S$={SELECTED_S})" if labeled else None,
        markersize=14.0,
        zorder=5,
    )


def _draw_labeled(rows, stem: Path) -> None:
    ylim = _ylim(rows)
    fig, ax = _new_axes(ylim, len(rows))
    apply_axis_text(ax, "Slice size $S$", "Relative L2 (%)")
    _plot(ax, rows, labeled=True)
    # Right spine; higher, just under Overall into the gap.
    add_legend(
        ax,
        loc="upper right",
        fontsize=14.0,
        bbox_to_anchor=(0.990, 0.70),
        bbox_transform=ax.transAxes,
        labelspacing=0.30,
        handletextpad=0.5,
    )
    pdf_path, png_path = save_figure(fig, stem, dpi=PAPER_DPI)
    plt.close(fig)
    print(f"wrote {pdf_path}", flush=True)
    print(f"wrote {png_path}", flush=True)


def _draw_vector(rows, stem: Path) -> None:
    ylim = _ylim(rows)
    fig, ax = _new_axes(ylim, len(rows))
    strip_axis_text(ax)
    _plot(ax, rows, labeled=False)
    path = save_axes_only(fig, stem)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(SLICE_ROWS)
    print(
        f"{'S':>4} {'arm':12s} {'Vel%':>7s} {'P%':>7s} {'All%':>7s}  note",
        flush=True,
    )
    for s, arm, vel, p, all_, _inf in rows:
        note = " ← selected" if s == SELECTED_S else ""
        print(f"{s:4d} {arm:12s} {vel:7.2f} {p:7.2f} {all_:7.2f}{note}", flush=True)

    labeled_dir = args.out / "labeled"
    vector_dir = args.out / "vector"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    _draw_labeled(rows, labeled_dir / STEM)
    _draw_vector(rows, vector_dir / STEM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
