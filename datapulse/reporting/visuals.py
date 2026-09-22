"""Module 9 - Automated visual reporting suite.

Every plot answers one question and carries its own legend or direct labels, so
none of them depends on colour alone:

=============================  =================================================
Plot                           Question it answers
=============================  =================================================
Correlation heatmap            Which features are redundant with each other?
Nullity matrix                 Is missingness random, or structured by row?
Class balance (pre/post)       What exactly did SMOTE do to my label prior?
Skew before/after              Did the power transforms actually work?
Drift (PSI)                    Which columns moved between train and test?
Feature relevance              Which features survived selection, and why?
=============================  =================================================

Design rules applied throughout: one axis per chart (never a secondary y-axis),
a fixed categorical colour order, a single-hue sequential ramp for magnitude, a
two-hue diverging ramp with a **neutral grey** midpoint for correlation,
reserved status colours for drift severity, recessive grids, and direct value
labels wherever a reader would otherwise have to measure a bar against an axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from datapulse.config import ReportingConfig  # noqa: E402
from datapulse.exceptions import ReportingError  # noqa: E402
from datapulse.logger import get_logger  # noqa: E402
from datapulse.reporting.palette import (  # noqa: E402
    CATEGORICAL,
    GRID,
    INK,
    MUTED_INK,
    SURFACE,
    diverging_cmap,
    sequential_cmap,
    status_color,
)

__all__ = ["VisualReporter"]

_logger = get_logger(__name__)


class VisualReporter:
    """Render and persist the diagnostic plot suite.

    Parameters
    ----------
    config:
        A :class:`~datapulse.config.ReportingConfig`.

    Attributes
    ----------
    generated_ : dict[str, pathlib.Path]
        Plot name -> file path for everything produced so far.

    Examples
    --------
    >>> import pandas as pd, tempfile
    >>> from datapulse.config import ReportingConfig
    >>> df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     rep = VisualReporter(ReportingConfig(output_dir=tmp))
    ...     _ = rep.correlation_heatmap(df)
    ...     sorted(rep.generated_)
    ['correlation_heatmap']

    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()
        self.generated_: dict[str, Path] = {}
        self._apply_style()

    # ------------------------------------------------------------------ #
    def _apply_style(self) -> None:
        """Set a recessive, consistent look for every figure."""
        sns.set_theme(style=self.config.style)
        plt.rcParams.update(
            {
                "figure.facecolor": SURFACE,
                "axes.facecolor": SURFACE,
                "savefig.facecolor": SURFACE,
                "axes.edgecolor": GRID,
                "axes.labelcolor": INK,
                "axes.titlesize": 13,
                "axes.titleweight": "semibold",
                "axes.titlecolor": INK,
                "axes.grid": True,
                "grid.color": GRID,
                "grid.linewidth": 0.7,
                "text.color": INK,
                "xtick.color": MUTED_INK,
                "ytick.color": MUTED_INK,
                "font.size": 10,
                "legend.frameon": False,
                "figure.autolayout": False,
            }
        )

    def _path(self, name: str) -> Path:
        directory = Path(self.config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{name}.{self.config.figure_format}"

    def _save(self, fig: plt.Figure, name: str) -> Path:
        """Persist ``fig`` under ``name`` and register it in ``generated_``."""
        path = self._path(name)
        try:
            fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        except OSError as exc:  # pragma: no cover - disk failures
            raise ReportingError(f"Could not write report {path}: {exc}") from exc
        finally:
            plt.close(fig)
        self.generated_[name] = path
        _logger.info("Report written: %s", path)
        return path

    @staticmethod
    def _despine(ax: plt.Axes, *, keep_x: bool = True) -> None:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if not keep_x:
            ax.spines["bottom"].set_visible(False)

    # ------------------------------------------------------------------ #
    # 1. Correlation heatmap
    # ------------------------------------------------------------------ #
    def correlation_heatmap(
        self,
        frame: pd.DataFrame,
        *,
        name: str = "correlation_heatmap",
        title: str = "Feature correlation (Pearson)",
    ) -> Optional[Path]:
        """Plot a lower-triangle correlation heatmap.

        Uses a **diverging** ramp (warm / neutral grey / cool) because
        correlation has a meaningful zero - a sequential ramp would hide the
        sign, and a rainbow would invent structure that is not there.
        """
        if not self.config.enabled or not self.config.generate_correlation_heatmap:
            return None

        numeric = frame.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            _logger.warning("Correlation heatmap skipped: fewer than 2 numeric columns.")
            return None

        limit = self.config.max_heatmap_features
        if numeric.shape[1] > limit:
            # Keep the most variable columns: they carry the structure worth seeing.
            keep = numeric.var(axis=0, ddof=0).nlargest(limit).index
            numeric = numeric[keep]
            title = f"{title} - top {limit} by variance"

        correlation = numeric.corr(numeric_only=True)
        mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)

        size = max(6.0, min(0.42 * correlation.shape[1] + 3.0, 16.0))
        fig, ax = plt.subplots(figsize=(size, size * 0.86))
        annotate = correlation.shape[1] <= 15
        sns.heatmap(
            correlation,
            mask=mask,
            cmap=diverging_cmap(),
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            square=True,
            linewidths=0.8,
            linecolor=SURFACE,
            annot=annotate,
            fmt=".2f",
            annot_kws={"size": 8, "color": INK},
            cbar_kws={"shrink": 0.6, "label": "Pearson r"},
            ax=ax,
        )
        ax.set_title(title, pad=14, loc="left")
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(False)
        return self._save(fig, name)

    # ------------------------------------------------------------------ #
    # 2. Nullity matrix
    # ------------------------------------------------------------------ #
    def nullity_matrix(
        self,
        frame: pd.DataFrame,
        *,
        name: str = "nullity_matrix",
        max_rows: int = 800,
    ) -> Optional[Path]:
        """Plot a missingness matrix plus a per-column missing-rate bar chart.

        Two panels share one figure but **not** one axis: the left panel is a
        row x column presence map, the right is a magnitude ranking with direct
        labels. Stripes in the left panel mean missingness is structured (a
        failed join, a feature added mid-history), which changes how you impute.
        """
        if not self.config.enabled or not self.config.generate_nullity_matrix:
            return None
        if frame.empty:
            return None

        sample = frame if len(frame) <= max_rows else frame.sample(max_rows, random_state=0)
        sample = sample.sort_index()
        missing_map = sample.isna().to_numpy(dtype=float)
        all_rates = frame.isna().mean().sort_values(ascending=True)
        # Listing thirty columns at 0.0% buries the handful that matter.
        rates = all_rates[all_rates > 0]
        n_clean = int((all_rates == 0).sum())
        if rates.empty:
            rates = all_rates.tail(12)
            n_clean = 0

        fig, (ax_map, ax_bar) = plt.subplots(
            1, 2, figsize=(15, max(5.0, min(0.22 * frame.shape[1] + 3.5, 12.0))),
            gridspec_kw={"width_ratios": [1.45, 1.0]},
        )

        present_colour, missing_colour = "#DCE3EC", status_color("critical")
        ax_map.imshow(
            missing_map,
            aspect="auto",
            interpolation="nearest",
            cmap=matplotlib.colors.ListedColormap([present_colour, missing_colour]),
            vmin=0,
            vmax=1,
        )
        ax_map.set_xticks(range(frame.shape[1]))
        ax_map.set_xticklabels(frame.columns, rotation=90, fontsize=7)
        ax_map.set_ylabel(f"Rows (sample of {len(sample):,})")
        ax_map.set_title("Missingness by row and column", loc="left", pad=12)
        ax_map.grid(False)
        ax_map.legend(
            handles=[
                Patch(facecolor=present_colour, label="present"),
                Patch(facecolor=missing_colour, label="missing"),
            ],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.12),
            ncol=2,
            fontsize=8,
        )

        positions = np.arange(len(rates))
        ramp = sequential_cmap()
        maximum = float(rates.max()) or 1.0
        colours = [ramp(0.25 + 0.7 * (v / maximum)) for v in rates.to_numpy()]
        ax_bar.barh(positions, rates.to_numpy() * 100, color=colours, height=0.62)
        ax_bar.set_yticks(positions)
        ax_bar.set_yticklabels(rates.index, fontsize=8)
        ax_bar.set_xlabel("Missing (%)")
        subtitle = (
            f"{len(rates)} column(s) affected · {n_clean} complete"
            if n_clean
            else f"{len(rates)} column(s)"
        )
        ax_bar.set_title(f"Missing rate per column\n{subtitle}", loc="left", pad=12)
        ax_bar.set_xlim(0, max(rates.max() * 100 * 1.22, 1.0))
        for y, value in zip(positions, rates.to_numpy()):
            ax_bar.text(
                value * 100 + maximum * 100 * 0.015,
                y,
                f"{value * 100:.1f}%",
                va="center",
                fontsize=7.5,
                color=MUTED_INK,
            )
        self._despine(ax_bar)
        ax_bar.grid(axis="y", visible=False)

        fig.suptitle("")
        fig.tight_layout()
        return self._save(fig, name)

    # ------------------------------------------------------------------ #
    # 3. Class balance before / after resampling
    # ------------------------------------------------------------------ #
    def class_balance(
        self,
        before: dict,
        after: dict,
        *,
        name: str = "class_balance_smote",
        strategy: str = "SMOTE",
        applied: bool = True,
    ) -> Optional[Path]:
        """Plot the label distribution before and after synthetic oversampling.

        Two series, so a legend is always present, and every bar is directly
        labelled - identity is never carried by colour alone.
        """
        if not self.config.enabled or not self.config.generate_class_balance:
            return None
        if not before:
            return None

        labels = sorted(set(before) | set(after or before), key=str)
        before_values = np.array([before.get(k, 0) for k in labels], dtype=float)
        after_values = np.array([(after or before).get(k, 0) for k in labels], dtype=float)

        positions = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(labels) + 4.0), 5.0))

        bars_before = ax.bar(
            positions - width / 2 - 0.01,
            before_values,
            width,
            label="Before resampling",
            color=CATEGORICAL[0],
        )
        bars_after = ax.bar(
            positions + width / 2 + 0.01,
            after_values,
            width,
            label=f"After {strategy.replace('_', '-').upper()}"
            if applied
            else "After (unchanged)",
            color=CATEGORICAL[1],
        )

        for group in (bars_before, bars_after):
            for bar in group:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max(after_values.max(), 1) * 0.015,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=MUTED_INK,
                )

        imbalance_before = before_values.max() / max(before_values.min(), 1)
        imbalance_after = after_values.max() / max(after_values.min(), 1)
        subtitle = (
            f"imbalance ratio {imbalance_before:.2f}:1 -> {imbalance_after:.2f}:1"
            if applied
            else "resampling vetoed by safeguard - distribution unchanged"
        )

        ax.set_xticks(positions)
        ax.set_xticklabels([str(label) for label in labels])
        ax.set_xlabel("Class")
        ax.set_ylabel("Training rows")
        ax.set_title(f"Class distribution shift\n{subtitle}", loc="left", pad=14)
        ax.set_ylim(0, max(after_values.max(), before_values.max()) * 1.12)
        # Below the axes: a legend box inside the plot would collide with the
        # tallest bar's direct label.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
        ax.grid(axis="x", visible=False)
        self._despine(ax)
        fig.tight_layout()
        return self._save(fig, name)

    # ------------------------------------------------------------------ #
    # 4. Skewness before / after
    # ------------------------------------------------------------------ #
    def skew_report(
        self,
        decisions: pd.DataFrame,
        *,
        name: str = "skew_transformations",
        threshold: float = 0.75,
        top_n: int = 25,
    ) -> Optional[Path]:
        """Plot \\|skew\\| before and after the adaptive transform.

        The dashed reference line is the configured skew threshold: bars that
        start right of it and end left of it are the transforms that earned
        their keep.
        """
        if not self.config.enabled or not self.config.generate_skew_report:
            return None
        if decisions is None or decisions.empty:
            return None

        table = decisions.copy()
        table["abs_before"] = table["skew_before"].abs()
        table["abs_after"] = table["skew_after"].abs()
        table = table.nlargest(min(top_n, len(table)), "abs_before").iloc[::-1]

        positions = np.arange(len(table))
        height = 0.38
        fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.42 * len(table) + 2.4)))

        ax.barh(
            positions - height / 2 - 0.01,
            table["abs_before"],
            height,
            label="|skew| before",
            color=CATEGORICAL[0],
        )
        ax.barh(
            positions + height / 2 + 0.01,
            table["abs_after"],
            height,
            label="|skew| after",
            color=CATEGORICAL[3],
        )

        limit = float(max(table["abs_before"].max(), table["abs_after"].max(), 1.0))
        for y, (before, after) in enumerate(
            zip(table["abs_before"], table["abs_after"])
        ):
            ax.text(before + limit * 0.012, y - height / 2, f"{before:.2f}",
                    va="center", fontsize=7.5, color=MUTED_INK)
            ax.text(after + limit * 0.012, y + height / 2, f"{after:.2f}",
                    va="center", fontsize=7.5, color=MUTED_INK)

        ax.axvline(threshold, color=MUTED_INK, linestyle="--", linewidth=1.2, zorder=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(
            [f"{row.column}  ({row.treatment})" for row in table.itertuples()],
            fontsize=8,
        )
        ax.set_xlabel("|skewness|")
        ax.set_xlim(0, limit * 1.2)
        ax.set_title(
            "Skew correction by column\ndashed line = configured skew threshold",
            loc="left",
            pad=14,
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color=MUTED_INK, linestyle="--", linewidth=1.2))
        labels.append(f"threshold = {threshold}")
        ax.legend(handles, labels, loc="lower right", fontsize=9)
        ax.grid(axis="y", visible=False)
        self._despine(ax)
        fig.tight_layout()
        return self._save(fig, name)

    # ------------------------------------------------------------------ #
    # 5. Drift
    # ------------------------------------------------------------------ #
    def drift_report(
        self,
        table: pd.DataFrame,
        *,
        name: str = "drift_psi",
        warn_threshold: float = 0.10,
        alert_threshold: float = 0.25,
        top_n: int = 25,
    ) -> Optional[Path]:
        """Plot per-column PSI, coloured by reserved status colours.

        Status colour never travels alone: the severity is also spelled out in
        the legend and the two reference lines make the bands readable in
        greyscale.
        """
        if not self.config.enabled or not self.config.generate_drift_report:
            return None
        if table is None or table.empty:
            return None

        subset = table.nlargest(min(top_n, len(table)), "psi").iloc[::-1]
        positions = np.arange(len(subset))
        colours = [status_color(s) for s in subset["severity"]]

        fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.34 * len(subset) + 2.4)))
        ax.barh(positions, subset["psi"], height=0.62, color=colours)

        limit = float(max(subset["psi"].max(), alert_threshold * 1.4, 0.05))
        for y, value in zip(positions, subset["psi"]):
            ax.text(value + limit * 0.012, y, f"{value:.3f}", va="center",
                    fontsize=8, color=MUTED_INK)

        ax.axvline(warn_threshold, color=MUTED_INK, linestyle="--", linewidth=1.1, zorder=0)
        ax.axvline(alert_threshold, color=MUTED_INK, linestyle=":", linewidth=1.4, zorder=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(subset["column"], fontsize=8)
        ax.set_xlabel("Population Stability Index")
        ax.set_xlim(0, limit * 1.22)
        ax.set_title(
            "Train vs test drift by feature\nPSI < 0.10 stable · 0.10-0.25 moderate · > 0.25 significant",
            loc="left",
            pad=14,
        )
        ax.legend(
            handles=[
                Patch(facecolor=status_color("stable"), label="stable"),
                Patch(facecolor=status_color("warning"), label="warning"),
                Patch(facecolor=status_color("alert"), label="alert"),
                Line2D([0], [0], color=MUTED_INK, linestyle="--", label=f"warn = {warn_threshold}"),
                Line2D([0], [0], color=MUTED_INK, linestyle=":", label=f"alert = {alert_threshold}"),
            ],
            loc="lower right",
            fontsize=8,
        )
        ax.grid(axis="y", visible=False)
        self._despine(ax)
        fig.tight_layout()
        return self._save(fig, name)

    # ------------------------------------------------------------------ #
    # 6. Feature relevance
    # ------------------------------------------------------------------ #
    def feature_relevance(
        self,
        selection_report: pd.DataFrame,
        *,
        name: str = "feature_relevance",
        top_n: int = 25,
    ) -> Optional[Path]:
        """Plot mutual information for the top features, flagging the dropped ones.

        A single measure, so a single hue: the sequential ramp encodes magnitude
        and nothing else. Selection status is carried by the hatch and the
        legend, never by hue.
        """
        if not self.config.enabled or not self.config.generate_feature_importance:
            return None
        if selection_report is None or selection_report.empty:
            return None
        if "mutual_info" not in selection_report.columns:
            return None

        subset = (
            selection_report.dropna(subset=["mutual_info"])
            .nlargest(min(top_n, len(selection_report)), "mutual_info")
            .iloc[::-1]
        )
        if subset.empty:
            return None

        positions = np.arange(len(subset))
        ramp = sequential_cmap()
        maximum = float(subset["mutual_info"].max()) or 1.0
        colours = [ramp(0.25 + 0.7 * (v / maximum)) for v in subset["mutual_info"]]
        hatches = ["" if keep else "///" for keep in subset["selected"]]

        fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.34 * len(subset) + 2.4)))
        bars = ax.barh(positions, subset["mutual_info"], height=0.62, color=colours)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            bar.set_edgecolor(SURFACE)

        for y, value in zip(positions, subset["mutual_info"]):
            ax.text(value + maximum * 0.015, y, f"{value:.3f}", va="center",
                    fontsize=8, color=MUTED_INK)

        ax.set_yticks(positions)
        ax.set_yticklabels(subset["feature"], fontsize=8)
        ax.set_xlabel("Mutual information with target")
        ax.set_xlim(0, maximum * 1.2)
        ax.set_title("Feature relevance and selection outcome", loc="left", pad=14)
        ax.legend(
            handles=[
                Patch(facecolor=ramp(0.8), label="selected"),
                Patch(facecolor=ramp(0.8), hatch="///", edgecolor=SURFACE, label="dropped"),
            ],
            loc="lower right",
            fontsize=9,
        )
        ax.grid(axis="y", visible=False)
        self._despine(ax)
        fig.tight_layout()
        return self._save(fig, name)

    # ------------------------------------------------------------------ #
    def manifest(self) -> pd.DataFrame:
        """Return a table of everything generated during this run."""
        if not self.generated_:
            return pd.DataFrame(columns=["report", "path"])
        return pd.DataFrame(
            {"report": list(self.generated_), "path": [str(p) for p in self.generated_.values()]}
        )
