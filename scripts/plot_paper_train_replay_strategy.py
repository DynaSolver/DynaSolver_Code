#!/usr/bin/env python3
"""Paper training curves: Pure Replay vs Replay-augmented Parallel (Gradual).

Two arms from Table-2 (fulltest48). Metric: validation rollout relative-L2 (%)
vs epoch — matches Overall L2 selection used in the table.

Main line: only key epochs (every 25, plus first/last). At each key, take a
local mean of the raw series; connect keys with straight segments (polyline).
Light vibration band = rolling ±1 std sampled at the same keys.

Task-specific: arm list + epoch_history loading.
Style/export: ``utils.paper_curve_style``.

Outputs under <out>/figures/paper/train_replay_strategy/:
  labeled/   PNG@500dpi + PDF (Times labels, legend)
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
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "dynasolver"
for _p in (REPO_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.paper_curve_style import (
    PAPER_DPI,
    ROLE_STYLES,
    add_legend,
    apply_axis_text,
    make_panels,
    save_axes_only,
    save_figure,
    strip_axis_text,
    style_line,
    use_times,
)

RUNS_ROOT = Path(
    "${DYNASOLVER_DATA}/runs/ablations_mini8_kinematic"
)
EVAL_ROOT = RUNS_ROOT / "_eval_fulltest48_rollout"
DEFAULT_OUT = RUNS_ROOT

# (eval_arm, display, role) — Gradual mixed = replay-augmented parallel (hero).
SERIES: tuple[tuple[str, str, str], ...] = (
    ("ours_s32", "Replay-augmented Parallel", "hero"),
    ("pure_replay", "Pure Replay", "baseline"),
)

STEM = "train_rollout_relL2_vs_epoch"
SMOOTH_WINDOW = 11
BAND_WINDOW = 25  # rolling window for vibration (±1 std)
KEY_EVERY = 25
BAND_ALPHA = 0.18


def _run_dir_from_eval(arm: str) -> Path:
    payload = json.loads((EVAL_ROOT / arm / "rollout_vp.json").read_text(encoding="utf-8"))
    run_dir = Path(str(payload["run_dir"]))
    hist = run_dir / "epoch_history.csv"
    if not hist.is_file():
        raise FileNotFoundError(hist)
    return run_dir


def _load_curve(arm: str) -> tuple[np.ndarray, np.ndarray]:
    hist = _run_dir_from_eval(arm) / "epoch_history.csv"
    rows = list(csv.DictReader(hist.open(encoding="utf-8")))
    xs = np.asarray([int(float(r["epoch"])) for r in rows], dtype=np.int32)
    ys = np.asarray(
        [100.0 * float(r["validation_rollout_relative_l2"]) for r in rows],
        dtype=np.float64,
    )
    return xs, ys


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average; window forced odd."""
    if window <= 1 or values.size <= 1:
        return values.copy()
    w = int(window)
    if w % 2 == 0:
        w += 1
    if values.size < w:
        return values.copy()
    half = w // 2
    out = np.empty_like(values)
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        out[i] = values[lo:hi].mean()
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling std (population); used for vibration band width."""
    if window <= 1 or values.size <= 1:
        return np.zeros_like(values)
    w = int(window)
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = np.empty_like(values)
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        chunk = values[lo:hi]
        out[i] = float(chunk.std(ddof=0)) if chunk.size > 1 else 0.0
    return out


def _key_indices(xs: np.ndarray, every: int) -> np.ndarray:
    """Epochs every ``every`` (incl. first and last)."""
    keys = {int(xs[0]), int(xs[-1])}
    for e in range(every, int(xs[-1]) + 1, every):
        keys.add(e)
    # map epoch -> index (prefer exact match)
    idx_by_ep = {int(e): i for i, e in enumerate(xs)}
    idxs: list[int] = []
    for e in sorted(keys):
        if e in idx_by_ep:
            idxs.append(idx_by_ep[e])
        else:
            # nearest
            idxs.append(int(np.argmin(np.abs(xs.astype(np.int64) - e))))
    # unique, sorted
    return np.asarray(sorted(set(idxs)), dtype=np.int64)


def _key_local_mean(values: np.ndarray, key_i: np.ndarray, window: int) -> np.ndarray:
    """Local mean of ``values`` centered at each key index."""
    smooth = _smooth(values, window)
    return smooth[key_i]


def _collect() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for arm, label, _role in SERIES:
        xs, raw = _load_curve(arm)
        std_full = _rolling_std(raw, BAND_WINDOW)
        key_i = _key_indices(xs, KEY_EVERY)
        x_key = xs[key_i]
        y_key = _key_local_mean(raw, key_i, SMOOTH_WINDOW)
        std_key = std_full[key_i]
        print(
            f"  {label:24s}  n={len(xs)}  last={raw[-1]:.2f}%  "
            f"key_last={y_key[-1]:.2f}%  keys={x_key.tolist()}",
            flush=True,
        )
        out[arm] = {
            "x_key": x_key,
            "y_key": y_key,
            "std_key": std_key,
            "key_i": key_i,
        }
    return out


def _plot_series(ax, curves: dict, *, labeled: bool) -> None:
    # Bands first (under lines) — polyline band at key frames only.
    for arm, _label, role in SERIES:
        d = curves[arm]
        color = ROLE_STYLES[role]["color"]
        lo = d["y_key"] - d["std_key"]
        hi = d["y_key"] + d["std_key"]
        ax.fill_between(
            d["x_key"],
            lo,
            hi,
            color=color,
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=ROLE_STYLES[role].get("zorder", 2) - 0.5,
        )

    for arm, label, role in SERIES:
        d = curves[arm]
        # Straight segments between key means only (no dense curve).
        style_line(
            ax,
            d["x_key"],
            d["y_key"],
            role,
            label=label if labeled else None,
        )


def _new_axes():
    fig, axes = make_panels(1, width=7.6, height=4.6)
    use_times()
    ax = axes[0]
    ax.set_xlim(0, 205)
    return fig, ax


def _draw_labeled(curves: dict, stem: Path) -> None:
    fig, ax = _new_axes()
    apply_axis_text(ax, "Epoch", "Overall relative L2 (%)")
    _plot_series(ax, curves, labeled=True)
    add_legend(ax, loc="upper right", fontsize=10.5)
    pdf_path, png_path = save_figure(fig, stem, dpi=PAPER_DPI)
    plt.close(fig)
    print(f"wrote {pdf_path}", flush=True)
    print(f"wrote {png_path}", flush=True)


def _draw_vector(curves: dict, stem: Path) -> None:
    fig, ax = _new_axes()
    strip_axis_text(ax)
    _plot_series(ax, curves, labeled=False)
    path = save_axes_only(fig, stem)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    curves = _collect()

    paper = args.out_root / "figures" / "paper" / "train_replay_strategy"
    labeled_dir = paper / "labeled"
    vector_dir = paper / "vector"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    _draw_labeled(curves, labeled_dir / STEM)
    _draw_vector(curves, vector_dir / STEM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
