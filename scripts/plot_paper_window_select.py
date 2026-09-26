#!/usr/bin/env python3
"""Paper curve: window size W=1–7 selection (mini8 full-test48).

Shows how W=4 (ours_s32) is chosen: All/Vel/P relL2 (%) vs window length.
Style matches ``plot_paper_train_vs_overall.py`` / ``utils.paper_curve_style``.

Outputs under figures/paper/window_select/:
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
    strip_axis_text,
    style_line,
    style_scatter,
    use_times,
)

# mini8 kinematic full-test48; relL2 ×100. W=8 omitted from selection curve.
# (W, arm, Vel%, P%, All%, infer_s)
WINDOW_ROWS: tuple[tuple[int, str, float, float, float, float], ...] = (
    (1, "window1", 63.93, 59.47, 62.74, 55.4),
    (2, "window2", 60.63, 49.63, 58.48, 56.3),
    (3, "window3", 59.69, 45.62, 57.05, 65.4),
    (4, "ours_s32", 59.34, 43.65, 56.47, 56.6),
    (5, "window5", 59.08, 44.16, 56.38, 64.6),
    (6, "window6", 59.35, 43.46, 56.45, 63.3),
    (7, "window7", 60.61, 44.02, 57.57, 45.9),
)

SELECTED_W = 4
STEM = "window_W_relL2_select"

DEFAULT_OUT = Path(os.path.expandvars('./figures/paper/window_select'))


def _rows_w17() -> list[tuple[int, str, float, float, float, float]]:
    return [r for r in WINDOW_ROWS if 1 <= r[0] <= 7]


def _new_axes(xlim: tuple[float, float], ylim: tuple[float, float]):
    fig, axes = make_panels(1, width=7.6, height=4.6)
    use_times()
    ax = axes[0]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(list(range(1, 8)))
    return fig, ax


def _limits(rows: list[tuple[int, str, float, float, float, float]]):
    ys = [v for r in rows for v in (r[2], r[3], r[4])]
    y0, y1 = min(ys), max(ys)
    dy = max(y1 - y0, 1.0)
    return (0.6, 7.4), (y0 - 0.08 * dy, y1 + 0.10 * dy)


def _plot(ax, rows: list[tuple[int, str, float, float, float, float]], *, labeled: bool) -> None:
    ws = [r[0] for r in rows]
    vel = [r[2] for r in rows]
    p = [r[3] for r in rows]
    all_ = [r[4] for r in rows]
    sel = next(r for r in rows if r[0] == SELECTED_W)

    style_line(
        ax,
        ws,
        all_,
        "hero",
        label="Overall" if labeled else None,
        marker="o",
        markersize=5.5,
    )
    style_line(
        ax,
        ws,
        vel,
        "ablation",
        label="Velocity" if labeled else None,
        marker="^",
        markersize=5.2,
    )
    style_line(
        ax,
        ws,
        p,
        "baseline",
        label="Pressure" if labeled else None,
        marker="s",
        markersize=4.8,
        linestyle=(0, (6, 3)),
    )
    # Selected operating point W=4 (ours_s32).
    style_scatter(
        ax,
        [sel[0]],
        [sel[4]],
        "hero",
        label=f"Selected (W={SELECTED_W})" if labeled else None,
        markersize=14.0,
        zorder=5,
    )


def _draw_labeled(rows, stem: Path) -> None:
    xlim, ylim = _limits(rows)
    fig, ax = _new_axes(xlim, ylim)
    apply_axis_text(ax, "Window size $W$", "Relative L2 (%)")
    _plot(ax, rows, labeled=True)
    # Right spine; top under Overall (~56%), deep in Overall–Pressure gap.
    add_legend(
        ax,
        loc="upper right",
        fontsize=14.0,
        bbox_to_anchor=(0.990, 0.48),
        bbox_transform=ax.transAxes,
        labelspacing=0.30,
        handletextpad=0.5,
    )
    pdf_path, png_path = save_figure(fig, stem, dpi=PAPER_DPI)
    plt.close(fig)
    print(f"wrote {pdf_path}", flush=True)
    print(f"wrote {png_path}", flush=True)


def _draw_vector(rows, stem: Path) -> None:
    xlim, ylim = _limits(rows)
    fig, ax = _new_axes(xlim, ylim)
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
    rows = _rows_w17()
    print(
        f"{'W':>3} {'arm':12s} {'Vel%':>7s} {'P%':>7s} {'All%':>7s}  note",
        flush=True,
    )
    for w, arm, vel, p, all_, _inf in rows:
        note = " ← selected" if w == SELECTED_W else ""
        print(f"{w:3d} {arm:12s} {vel:7.2f} {p:7.2f} {all_:7.2f}{note}", flush=True)

    labeled_dir = args.out / "labeled"
    vector_dir = args.out / "vector"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    _draw_labeled(rows, labeled_dir / STEM)
    _draw_vector(rows, vector_dir / STEM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
