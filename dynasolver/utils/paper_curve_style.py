"""Reusable publication curve style for paper figures.

Roles
  hero      opaque red star for the proposed method
  baseline  opaque black diamond for a strong reference (e.g. persistence)
  ablation  green triangle for controlled variants
  ghost     low-alpha pastel for crowded baselines

Typical flow
  apply_style() / use_times()
  fig, axes = make_panels(...)
  style_line(ax, x, y, "hero", label="Ours")
  # or style_scatter(ax, xs, ys, "hero", label="Ours") for dot panels
  add_split_legend(ax, column_sizes=(5, 6, 6))
  save_figure(fig, stem, dpi=500)
  save_axes_only(fig, stem)   # draw.io overlay PDF
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Dataset identity when color encodes the dataset.
DATASET_COLORS = {
    "DrivAerML": "#A3C94A",
    "NASA-CRM": "#C56B6B",
    "AirCraft": "#7A3480",
}

METHOD_COLORS = {
    "ours": "#D62728",
    "transolver": "#1A1A1A",
    "ablation": "#2E7D32",
    "ghost": "#9AA0A6",
}

CONDITION_STYLES = {
    "with_geopt": {"linestyle": "-", "marker": "*", "markersize": 9.0, "linewidth": 1.7},
    "without_geopt": {
        "linestyle": "--",
        "marker": "d",
        "markersize": 5.2,
        "linewidth": 1.05,
        "alpha": 0.55,
    },
    "unique_geometry": {"linestyle": "-", "marker": "s", "markersize": 5.4, "linewidth": 1.6},
    "unique_dynamics": {"linestyle": "-", "marker": "^", "markersize": 6.2, "linewidth": 1.6},
    "method": {"linestyle": "-", "marker": "*", "markersize": 9.0, "linewidth": 1.8},
    "strong_baseline": {"linestyle": "-", "marker": "d", "markersize": 5.6, "linewidth": 1.6},
    "variant": {"linestyle": "-", "marker": "v", "markersize": 5.6, "linewidth": 1.45},
    "other": {"linestyle": "-", "marker": "o", "markersize": 5.0, "linewidth": 1.35},
}

ROLE_STYLES = {
    "hero": {
        "color": METHOD_COLORS["ours"],
        "marker": "*",
        "linestyle": "-",
        "linewidth": 1.8,
        "markersize": 9.0,
        "alpha": 1.0,
        "zorder": 4,
    },
    "baseline": {
        "color": METHOD_COLORS["transolver"],
        "marker": "d",
        "linestyle": "-",
        "linewidth": 1.6,
        "markersize": 5.6,
        "alpha": 1.0,
        "zorder": 3,
    },
    "ablation": {
        "color": METHOD_COLORS["ablation"],
        "marker": "v",
        "linestyle": "-",
        "linewidth": 1.45,
        "markersize": 5.6,
        "alpha": 0.9,
        "zorder": 2,
    },
    "ghost": {
        "color": METHOD_COLORS["ghost"],
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.15,
        "markersize": 4.8,
        "alpha": 0.35,
        "zorder": 1,
    },
}

# Fixed low-saturation palette for many ghost baselines.
GHOST_COLORS = (
    "#7BA3C9",
    "#E0A15A",
    "#7DAA7A",
    "#C98B8B",
    "#9A84C7",
    "#B08968",
    "#D4A0B0",
    "#8E8E8E",
    "#A8B85A",
    "#6AA8B5",
    "#5C7A99",
    "#C47A3A",
    "#5E8A6A",
    "#A35D5D",
    "#6B4C7A",
)
GHOST_MARKERS = ("o", "s", "v", "^", "D", "P", "X")

# Paper defaults settled on the waterlily rollout figures.
PAPER_DPI = 500
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 13
LEGEND_SIZE = 10.5

_SANS = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
_SERIF = [
    "Times New Roman",
    "Liberation Serif",
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "TeX Gyre Termes",
    "DejaVu Serif",
]


def apply_style() -> None:
    """White canvas, four black spines, inward ticks, no grid."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _SANS,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "regular",
            "axes.grid": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def use_times() -> None:
    """Prefer Times New Roman; Fall back to Liberation Serif on this cluster."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": list(_SERIF),
            "mathtext.fontset": "stix",
        }
    )


def setup_panel(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
        length=3.5,
        width=0.8,
        color="black",
    )


def set_panel_title(ax, title: str) -> None:
    ax.set_title(title, loc="center", pad=8, fontsize=11)


def set_categorical_x(ax, labels, xlabel: str | None = None) -> list[int]:
    """Equal spacing even when labels look numeric."""
    xs = list(range(len(labels)))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    if labels:
        ax.set_xlim(-0.35, len(labels) - 0.65)
    if xlabel:
        ax.set_xlabel(xlabel)
    return xs


def sample_indices(n_total: int, n_keep: int, *, one_indexed: bool = True) -> list[int]:
    """Evenly sample ``n_keep`` frames, always including the first and last.

    Returns 1-based frame indices when ``one_indexed`` is True (AR frame labels).
    """
    if n_keep < 2:
        raise ValueError("need at least the first and last sample")
    if n_total < n_keep:
        raise ValueError(f"cannot keep {n_keep} of {n_total}")
    last = n_total - 1
    idxs = [round(i * last / (n_keep - 1)) for i in range(n_keep)]
    idxs[0] = 0
    idxs[-1] = last
    if len(set(idxs)) != n_keep or any(idxs[i] >= idxs[i + 1] for i in range(n_keep - 1)):
        raise ValueError(f"sample of {n_keep} collapsed to {idxs}")
    if one_indexed:
        return [i + 1 for i in idxs]
    return idxs


def ghost_style(index: int, *, alpha: float = 0.4) -> dict:
    """Color and marker for the ``index``-th ghost baseline."""
    return {
        "color": GHOST_COLORS[index % len(GHOST_COLORS)],
        "marker": GHOST_MARKERS[index % len(GHOST_MARKERS)],
        "alpha": alpha,
    }


def add_legend(ax, loc: str = "upper right", **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return None
    legend = ax.legend(frameon=False, loc=loc, fontsize=kwargs.pop("fontsize", 7.5), **kwargs)
    if legend is not None:
        legend.get_frame().set_facecolor("none")
        legend.get_frame().set_edgecolor("none")
    return legend


def add_split_legend(
    ax,
    column_sizes: tuple[int, ...],
    *,
    fontsize: float = LEGEND_SIZE,
    right: float = 0.995,
    bottom: float = 0.02,
    gap: float = 0.012,
    family: str = "serif",
) -> None:
    """Bottom-right multi-column legend. Column heights follow ``column_sizes`` left to right."""
    handles, labels = ax.get_legend_handles_labels()
    if sum(column_sizes) != len(handles):
        raise ValueError(
            f"column_sizes sum {sum(column_sizes)} != {len(handles)} legend entries"
        )
    columns: list[tuple[list, list]] = []
    start = 0
    for size in column_sizes:
        columns.append((handles[start : start + size], labels[start : start + size]))
        start += size

    legend_kw = dict(
        frameon=False,
        loc="lower right",
        prop={"family": family, "size": fontsize},
        handlelength=1.6,
        handletextpad=0.4,
        borderaxespad=0.3,
        labelspacing=0.28,
        markerscale=1.0,
    )
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cursor = right
    for col_handles, col_labels in reversed(columns):
        legend = ax.legend(
            col_handles,
            col_labels,
            bbox_to_anchor=(cursor, bottom),
            bbox_transform=ax.transAxes,
            **legend_kw,
        )
        legend.get_frame().set_facecolor("none")
        legend.get_frame().set_edgecolor("none")
        ax.add_artist(legend)
        fig.canvas.draw()
        box = legend.get_window_extent(renderer)
        (x0, _y0), (_x1, _y1) = ax.transAxes.inverted().transform(
            [(box.x0, box.y0), (box.x1, box.y1)]
        )
        cursor = x0 - gap


def style_line(ax, x, y, role: str = "hero", **kwargs):
    """Draw one series. ``role`` is hero / baseline / ablation / ghost."""
    if role not in ROLE_STYLES:
        known = ", ".join(ROLE_STYLES)
        raise KeyError(f"unknown role {role!r}; expected one of {known}")

    spec = dict(ROLE_STYLES[role])
    condition = kwargs.pop("condition", None)
    if condition is not None:
        if condition not in CONDITION_STYLES:
            known = ", ".join(CONDITION_STYLES)
            raise KeyError(f"unknown condition {condition!r}; expected one of {known}")
        spec.update(CONDITION_STYLES[condition])
    spec.update(kwargs)

    color = spec.pop("color")
    line, = ax.plot(
        x,
        y,
        color=color,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.6,
        clip_on=True,
        **spec,
    )
    return line


def style_scatter(ax, x, y, role: str = "hero", **kwargs):
    """Draw one or more markers with no connecting line (dot / scatter panels).

    ``role`` matches ``style_line`` (hero / baseline / ablation / ghost).
    Pass ``ghost_style(i)`` kwargs for crowded baselines.
    """
    if role not in ROLE_STYLES:
        known = ", ".join(ROLE_STYLES)
        raise KeyError(f"unknown role {role!r}; expected one of {known}")

    spec = dict(ROLE_STYLES[role])
    condition = kwargs.pop("condition", None)
    if condition is not None:
        if condition not in CONDITION_STYLES:
            known = ", ".join(CONDITION_STYLES)
            raise KeyError(f"unknown condition {condition!r}; expected one of {known}")
        spec.update(CONDITION_STYLES[condition])
    spec.update(kwargs)
    # Scatter panels: marker only.
    spec["linestyle"] = "None"
    spec.pop("linewidth", None)
    # Slightly larger markers than curve ghosts for readability.
    if role == "ghost" and "markersize" not in kwargs:
        spec["markersize"] = 7.0
    elif role == "hero" and "markersize" not in kwargs:
        spec["markersize"] = 14.0
    elif role == "baseline" and "markersize" not in kwargs:
        spec["markersize"] = 8.0

    color = spec.pop("color")
    (artist,) = ax.plot(
        x,
        y,
        color=color,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.6,
        clip_on=True,
        **spec,
    )
    return artist


def make_panels(
    ncols: int,
    *,
    sharey: bool = True,
    width: float = 3.35,
    height: float = 2.75,
    ylabel: str | None = None,
):
    if ncols < 1:
        raise ValueError("ncols must be >= 1")
    apply_style()
    fig, axes = plt.subplots(
        1,
        ncols,
        sharey=sharey,
        figsize=(width * ncols + 0.18 * (ncols - 1), height),
        squeeze=False,
    )
    panels = list(axes[0])
    for ax in panels:
        setup_panel(ax)
    if ylabel:
        panels[0].set_ylabel(ylabel)
    fig.subplots_adjust(wspace=0.16)
    return fig, panels


def apply_axis_text(ax, xlabel: str, ylabel: str) -> None:
    """Large serif axis labels and tick numbers for paper panels."""
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, fontfamily="serif")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, fontfamily="serif")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("serif")


def strip_axis_text(ax) -> None:
    """Keep the frame and ticks; drop every word for draw.io overlays."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        axis="both",
        labelbottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
    )


def save_figure(fig, stem: str | Path, dpi: int = PAPER_DPI) -> tuple[Path, Path]:
    """Write a vector PDF and a PNG next to each other."""
    path = Path(stem)
    if path.suffix.lower() in {".pdf", ".png", ".svg"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    return pdf_path, png_path


def save_axes_only(fig, stem: str | Path) -> Path:
    """Write a text-free vector PDF for external captioning."""
    path = Path(stem)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    return path
