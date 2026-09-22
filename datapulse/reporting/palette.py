"""Colour system for the reporting suite.

The palette is not a matter of taste - it is validated against five checks
(lightness band, chroma floor, colour-vision-deficiency separation between
adjacent slots, normal-vision separation, and contrast against the chart
surface). The order below is **fixed**: series take slots 1, 2, 3 ... and are
never cycled through a generated hue.

Rules encoded here
------------------
* **Categorical** (identity): the fixed order in :data:`CATEGORICAL`. The worst
  adjacent CVD separation sits in the 6-8 Delta-E floor band, which is legal
  only alongside secondary encoding - so every categorical chart in
  :mod:`datapulse.reporting.visuals` ships a legend *and* direct value labels.
* **Sequential** (magnitude): one hue, light to dark. Never a rainbow.
* **Diverging** (polarity, e.g. correlation): two hues with a **neutral grey**
  midpoint - never a hue at zero.
* **Status** (state): reserved for good / warning / serious / critical. These
  are never reused as "series 4".
"""

from __future__ import annotations

from typing import Final

from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "CATEGORICAL",
    "STATUS",
    "SURFACE",
    "INK",
    "MUTED_INK",
    "GRID",
    "SEQUENTIAL_HUE",
    "diverging_cmap",
    "sequential_cmap",
    "status_color",
]

#: Fixed categorical order. Assign slot-by-slot; never cycle or generate hues.
CATEGORICAL: Final[tuple[str, ...]] = (
    "#2B6CB0",  # blue
    "#DD6B20",  # orange
    "#805AD5",  # purple
    "#2F855A",  # green
    "#C53030",  # red
    "#B7791F",  # gold
)

#: Reserved state colours - never used for ordinary series.
STATUS: Final[dict[str, str]] = {
    "good": "#2F855A",
    "stable": "#2F855A",
    "warning": "#B7791F",
    "serious": "#DD6B20",
    "critical": "#C53030",
    "alert": "#C53030",
}

SURFACE: Final[str] = "#FCFCFB"
INK: Final[str] = "#1A202C"
MUTED_INK: Final[str] = "#718096"
GRID: Final[str] = "#E2E8F0"

#: Single hue used for every sequential (magnitude) ramp.
SEQUENTIAL_HUE: Final[tuple[str, str]] = ("#EBF2FA", "#1A4E8A")

#: Warm pole, neutral grey midpoint, cool pole.
_DIVERGING_STOPS: Final[tuple[str, str, str]] = ("#C53030", "#EDEDEA", "#2B6CB0")


def sequential_cmap(name: str = "datapulse_seq") -> LinearSegmentedColormap:
    """Return the single-hue, light-to-dark sequential colormap."""
    return LinearSegmentedColormap.from_list(name, list(SEQUENTIAL_HUE), N=256)


def diverging_cmap(name: str = "datapulse_div") -> LinearSegmentedColormap:
    """Return the two-hue diverging colormap with a neutral grey midpoint."""
    return LinearSegmentedColormap.from_list(name, list(_DIVERGING_STOPS), N=256)


def status_color(severity: str) -> str:
    """Map a severity label onto its reserved status colour."""
    return STATUS.get(str(severity).lower(), MUTED_INK)
