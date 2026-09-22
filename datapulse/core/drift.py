"""Module 8 - Data drift and quality auditing.

Preprocessing parameters (imputation fills, winsorization fences, target
encodings, scaler centres) are all fitted on the training distribution. If the
serving distribution moves, those parameters silently become wrong - which is
why a drift audit belongs *inside* the preprocessing framework rather than in a
separate monitoring project bolted on later.

Three complementary statistics are computed per column.

Population Stability Index (PSI)
    .. math:: \\mathrm{PSI} = \\sum_i (a_i - e_i)\\,\\ln(a_i / e_i)

    where :math:`e_i` and :math:`a_i` are the expected (reference) and actual
    (current) proportions in bin *i*. Bin edges come from the **reference**
    quantiles so the reference is uniform by construction. Industry convention:
    ``< 0.10`` stable, ``0.10-0.25`` moderate shift, ``> 0.25`` significant
    shift. PSI is symmetric-ish, bounded in practice, and works for categorical
    columns too (each level is its own bin).

Wasserstein distance (Earth Mover's Distance)
    The minimum "work" to morph one distribution into the other. Unlike PSI it
    is binning-free and respects the metric on the axis: a 1-unit shift of the
    whole distribution costs exactly 1. It is computed on the **z-scored**
    scale (using reference mean/std) so thresholds are comparable across
    columns with different units.

Kolmogorov-Smirnov test
    Supplies a p-value for "are these the same distribution?". Useful as a
    tie-breaker, but note that on large samples it flags differences far too
    small to matter - which is exactly why PSI and Wasserstein carry the
    decision and KS is advisory.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from datapulse.config import DriftConfig
from datapulse.exceptions import DataPulseError
from datapulse.logger import get_logger

__all__ = ["DriftSeverity", "DriftReport", "DriftAuditor"]

_logger = get_logger(__name__)


class DriftSeverity(str, Enum):
    """Traffic-light verdict for a single column."""

    STABLE = "stable"
    WARNING = "warning"
    ALERT = "alert"


class DriftReport:
    """Container for a completed drift audit.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per column with ``psi``, ``wasserstein``, ``ks_pvalue``,
        ``severity`` and supporting statistics.
    n_reference_rows, n_current_rows : int

    """

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        n_reference_rows: int,
        n_current_rows: int,
        config: DriftConfig,
    ) -> None:
        self.table = table
        self.n_reference_rows = n_reference_rows
        self.n_current_rows = n_current_rows
        self.config = config

    # ------------------------------------------------------------------ #
    @property
    def alerts(self) -> pd.DataFrame:
        """Columns whose drift exceeded the alert threshold."""
        if self.table.empty:
            return self.table
        return self.table[self.table["severity"] == DriftSeverity.ALERT.value]

    @property
    def warnings(self) -> pd.DataFrame:
        """Columns in the moderate-shift band."""
        if self.table.empty:
            return self.table
        return self.table[self.table["severity"] == DriftSeverity.WARNING.value]

    @property
    def is_stable(self) -> bool:
        """``True`` when no column breached the alert threshold."""
        return self.alerts.empty

    def to_frame(self) -> pd.DataFrame:
        """Return the audit table (a copy)."""
        return self.table.copy()

    def to_dict(self) -> dict:
        """JSON-friendly representation for logging or artifact metadata."""
        return {
            "n_reference_rows": self.n_reference_rows,
            "n_current_rows": self.n_current_rows,
            "n_columns_audited": int(len(self.table)),
            "n_alerts": int(len(self.alerts)),
            "n_warnings": int(len(self.warnings)),
            "is_stable": self.is_stable,
            "max_psi": float(self.table["psi"].max()) if not self.table.empty else 0.0,
            "top_drifted": self.table.head(5)[["column", "psi", "wasserstein", "severity"]]
            .to_dict(orient="records")
            if not self.table.empty
            else [],
        }

    def summary(self) -> str:
        """One-line human summary."""
        if self.table.empty:
            return "Drift audit: no comparable columns."
        return (
            f"Drift audit over {len(self.table)} columns "
            f"({self.n_reference_rows} ref vs {self.n_current_rows} current rows): "
            f"{len(self.alerts)} alert(s), {len(self.warnings)} warning(s), "
            f"max PSI={self.table['psi'].max():.4f}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<DriftReport {self.summary()}>"


class DriftAuditor:
    """Compute PSI, Wasserstein distance and KS statistics between two frames.

    Parameters
    ----------
    config:
        A :class:`~datapulse.config.DriftConfig`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> ref = pd.DataFrame({"x": rng.normal(0, 1, 2000)})
    >>> cur = pd.DataFrame({"x": rng.normal(2, 1, 2000)})
    >>> report = DriftAuditor().audit(ref, cur)
    >>> report.is_stable
    False

    """

    def __init__(self, config: Optional[DriftConfig] = None) -> None:
        self.config = config or DriftConfig()

    # ------------------------------------------------------------------ #
    def audit(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        columns: Optional[list[str]] = None,
    ) -> DriftReport:
        """Compare ``current`` against ``reference`` column by column.

        Parameters
        ----------
        reference:
            The training (baseline) frame.
        current:
            The test / production frame.
        columns:
            Optional explicit subset; defaults to the intersection.

        Returns
        -------
        DriftReport
            Sorted by PSI, most-drifted first.

        Raises
        ------
        DataPulseError
            If the two frames share no columns.

        """
        cfg = self.config
        shared = columns or [c for c in reference.columns if c in current.columns]
        if not shared:
            raise DataPulseError(
                "Drift audit requires at least one shared column between the frames."
            )
        shared = shared[: cfg.max_columns_reported]

        records: list[dict] = []
        for column in shared:
            ref_series = reference[column]
            cur_series = current[column]
            numeric = pd.api.types.is_numeric_dtype(
                ref_series
            ) and pd.api.types.is_numeric_dtype(cur_series)
            try:
                record = (
                    self._audit_numeric(column, ref_series, cur_series)
                    if numeric
                    else self._audit_categorical(column, ref_series, cur_series)
                )
            except (ValueError, ZeroDivisionError) as exc:  # pragma: no cover
                _logger.warning("Drift computation failed for %r: %s", column, exc)
                continue
            records.append(record)

        table = pd.DataFrame(records)
        if not table.empty:
            table["severity"] = table.apply(self._severity, axis=1)
            table = table.sort_values("psi", ascending=False).reset_index(drop=True)

        report = DriftReport(
            table,
            n_reference_rows=int(len(reference)),
            n_current_rows=int(len(current)),
            config=cfg,
        )
        _logger.info(report.summary())
        return report

    # ------------------------------------------------------------------ #
    def _audit_numeric(
        self, column: str, reference: pd.Series, current: pd.Series
    ) -> dict:
        """PSI + Wasserstein + KS for a numeric column."""
        ref = pd.to_numeric(reference, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna().to_numpy(dtype="float64")
        cur = pd.to_numeric(current, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna().to_numpy(dtype="float64")

        if ref.size == 0 or cur.size == 0:
            return self._empty_record(column, "numeric", reference, current)

        psi = self.population_stability_index(ref, cur)

        # z-score using the REFERENCE moments so the distance is unit-free and
        # a shift of the current distribution is not hidden by its own spread.
        scale = float(ref.std()) or 1.0
        centre = float(ref.mean())
        wasserstein = float(
            stats.wasserstein_distance((ref - centre) / scale, (cur - centre) / scale)
        )

        ks_stat, ks_p = (np.nan, np.nan)
        if self.config.ks_test_enabled and ref.size > 1 and cur.size > 1:
            result = stats.ks_2samp(ref, cur)
            ks_stat, ks_p = float(result.statistic), float(result.pvalue)

        return {
            "column": column,
            "dtype": "numeric",
            "psi": round(psi, 6),
            "wasserstein": round(wasserstein, 6),
            "ks_statistic": None if np.isnan(ks_stat) else round(ks_stat, 6),
            "ks_pvalue": None if np.isnan(ks_p) else round(ks_p, 6),
            "ref_mean": round(float(ref.mean()), 6),
            "cur_mean": round(float(cur.mean()), 6),
            "mean_shift": round(float(cur.mean() - ref.mean()), 6),
            "ref_null_rate": round(float(reference.isna().mean()), 6),
            "cur_null_rate": round(float(current.isna().mean()), 6),
            "new_categories": 0,
        }

    def _audit_categorical(
        self, column: str, reference: pd.Series, current: pd.Series
    ) -> dict:
        """PSI over category shares, plus an unseen-level count."""
        cfg = self.config
        ref_counts = reference.astype("object").fillna("__NA__").value_counts(normalize=True)
        cur_counts = current.astype("object").fillna("__NA__").value_counts(normalize=True)
        levels = ref_counts.index.union(cur_counts.index)

        expected = ref_counts.reindex(levels).fillna(0.0).to_numpy() + cfg.epsilon
        actual = cur_counts.reindex(levels).fillna(0.0).to_numpy() + cfg.epsilon
        expected = expected / expected.sum()
        actual = actual / actual.sum()
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))

        # Total-variation distance doubles as a bounded "distribution distance"
        # for categorical columns, where Wasserstein has no natural ordering.
        total_variation = float(0.5 * np.abs(actual - expected).sum())
        new_levels = int(len(set(cur_counts.index) - set(ref_counts.index)))

        return {
            "column": column,
            "dtype": "categorical",
            "psi": round(psi, 6),
            "wasserstein": round(total_variation, 6),
            "ks_statistic": None,
            "ks_pvalue": None,
            "ref_mean": None,
            "cur_mean": None,
            "mean_shift": None,
            "ref_null_rate": round(float(reference.isna().mean()), 6),
            "cur_null_rate": round(float(current.isna().mean()), 6),
            "new_categories": new_levels,
        }

    @staticmethod
    def _empty_record(
        column: str, dtype: str, reference: pd.Series, current: pd.Series
    ) -> dict:
        return {
            "column": column,
            "dtype": dtype,
            "psi": 0.0,
            "wasserstein": 0.0,
            "ks_statistic": None,
            "ks_pvalue": None,
            "ref_mean": None,
            "cur_mean": None,
            "mean_shift": None,
            "ref_null_rate": round(float(reference.isna().mean()), 6),
            "cur_null_rate": round(float(current.isna().mean()), 6),
            "new_categories": 0,
        }

    # ------------------------------------------------------------------ #
    def population_stability_index(
        self, reference: np.ndarray, current: np.ndarray
    ) -> float:
        """Compute PSI between two numeric samples.

        Parameters
        ----------
        reference, current:
            1-D numeric arrays with NaNs already removed.

        Returns
        -------
        float
            The PSI value; ``0.0`` for identical distributions.

        """
        cfg = self.config
        n_bins = cfg.psi_bins

        # Discrete numerics (binary flags, one-hot dummies, small rating
        # scales) have fewer distinct values than bins. Quantile edges would
        # collapse to a single bin and report a meaningless PSI of 0, so treat
        # each distinct value as its own bin instead.
        reference_levels = np.unique(reference)
        if reference_levels.size <= max(2, n_bins):
            levels = np.union1d(reference_levels, np.unique(current))
            expected = np.array([(reference == v).mean() for v in levels], dtype=float)
            actual = np.array([(current == v).mean() for v in levels], dtype=float)
            expected = expected + cfg.epsilon
            actual = actual + cfg.epsilon
            expected /= expected.sum()
            actual /= actual.sum()
            return float(np.sum((actual - expected) * np.log(actual / expected)))

        if cfg.psi_binning == "quantile":
            quantiles = np.linspace(0, 1, n_bins + 1)
            edges = np.unique(np.quantile(reference, quantiles))
        else:
            edges = np.unique(np.linspace(reference.min(), reference.max(), n_bins + 1))

        if edges.size < 2:
            # Constant reference column: PSI is defined as 0 unless the current
            # sample moved off that constant value entirely.
            return 0.0 if np.allclose(current, reference[0]) else 1.0

        edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        expected, _ = np.histogram(reference, bins=edges)
        actual, _ = np.histogram(current, bins=edges)

        expected_pct = expected / max(expected.sum(), 1) + cfg.epsilon
        actual_pct = actual / max(actual.sum(), 1) + cfg.epsilon
        expected_pct = expected_pct / expected_pct.sum()
        actual_pct = actual_pct / actual_pct.sum()

        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    # ------------------------------------------------------------------ #
    def _severity(self, row: pd.Series) -> str:
        """Assign a traffic-light verdict from PSI and Wasserstein."""
        cfg = self.config
        psi = float(row.get("psi") or 0.0)
        wasserstein = float(row.get("wasserstein") or 0.0)
        if psi >= cfg.psi_alert_threshold or wasserstein >= cfg.wasserstein_alert_threshold:
            return DriftSeverity.ALERT.value
        if psi >= cfg.psi_warn_threshold or wasserstein >= cfg.wasserstein_warn_threshold:
            return DriftSeverity.WARNING.value
        return DriftSeverity.STABLE.value
