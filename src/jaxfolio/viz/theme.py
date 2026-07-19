"""Dark matplotlib theme for jaxfolio.

Colors come from a validated, colorblind-considered palette (the reference
instance of the data-viz method): a fixed categorical order applied in-sequence
(never cycled arbitrarily), a single-hue blue sequential ramp for magnitude, and
neutral ink/grid tokens tuned for the dark chart surface ``#1a1a19``.

Import side effect: calling :func:`use_dark_theme` (also invoked by
``jaxfolio.viz`` on import) registers the rcParams globally.
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Surfaces & ink -------------------------------------------------------- #
SURFACE = "#1a1a19"  # chart surface
PLANE = "#0d0d0d"  # page/figure background
CARD = "#141413"  # card fill (slightly darker than the plot surface)
BORDER = "#33332f"  # card border (subtle, reads as a panel edge on the dark plane)
HIGHLIGHT = "#232322"  # emphasized table row / selected card fill
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"  # axis labels / ticks
GRID = "#2c2c2a"  # hairline gridline
BASELINE = "#383835"  # axis spine / baseline

# --- Categorical palette (dark steps, applied in fixed order) -------------- #
CATEGORICAL = [
    "#3987e5",  # blue
    "#008300",  # green
    "#d55181",  # magenta
    "#c98500",  # yellow
    "#199e70",  # aqua
    "#d95926",  # orange
    "#9085e9",  # violet
    "#e66767",  # red
]

# --- Status accents (reserved; never reused as a series color) ------------- #
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

# --- Sequential blue ramp (light -> dark) for heatmaps --------------------- #
_SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#104281",
    "#0d366b",
]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("jaxfolio_blue", _SEQUENTIAL_BLUE)

# --- Diverging blue<->red ramp for correlation / signed magnitude ---------- #
_DIVERGING = ["#104281", "#3987e5", "#9ec5f4", "#383835", "#eda1a1", "#e66767", "#d03b3b"]
DIVERGING_CMAP = LinearSegmentedColormap.from_list("jaxfolio_div", _DIVERGING)


def color(i: int) -> str:
    """Return the categorical color for series index ``i`` (fixed order).

    A 9th+ series intentionally wraps — callers with many series should fold to
    an "Other" bucket or use small multiples rather than rely on this wrap.
    """
    return CATEGORICAL[i % len(CATEGORICAL)]


def use_dark_theme() -> None:
    """Register the jaxfolio dark theme as the active matplotlib rcParams."""
    plt.rcParams.update(
        {
            "figure.facecolor": PLANE,
            "figure.edgecolor": PLANE,
            "savefig.facecolor": PLANE,
            "savefig.edgecolor": PLANE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.facecolor": SURFACE,
            "legend.edgecolor": BASELINE,
            "legend.framealpha": 0.9,
            "legend.labelcolor": INK_SECONDARY,
            "figure.titlesize": "x-large",
            "figure.titleweight": "bold",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "patch.edgecolor": SURFACE,
            "patch.linewidth": 0.5,
        }
    )
