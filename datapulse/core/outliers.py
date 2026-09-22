"""Module 3 - Outlier detection and treatment.

Two complementary views of "outlier" are supported:

* **Univariate (IQR / quantile winsorization).** A value is extreme relative to
  its own column's distribution. Cheap, interpretable, and the right tool for
  sensor spikes and fat-fingered data entry.
* **Multivariate (Isolation Forest).** A *row* is extreme as a combination even
  though no single field is. A 22-year-old with a 30-year mortgage history has
  no individually anomalous field.

The **action** is decoupled from detection:

``clip``
    Winsorise to the fitted fences. Inference-safe: it is a pure function of
    parameters learnt on the training set.
``mask``
    Replace with ``NaN`` and let a downstream imputer decide.
``flag``
    Leave values untouched but append ``__outlier_score`` / ``__is_outlier``
    features, letting the model decide how much to trust the row.
``drop``
    Remove the row entirely. This is a **training-only** operation - you can
    never refuse to score a production record - so it is exposed through
    :meth:`OutlierEngine.fit_resample`, not through ``transform``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from datapulse.base import AbstractTransformer
from datapulse.config import OutlierAction, OutlierConfig, OutlierMethod

__all__ = ["OutlierEngine"]


class OutlierEngine(AbstractTransformer):
    """Detect and treat univariate and multivariate outliers.

    Parameters
    ----------
    config:
        An :class:`~datapulse.config.OutlierConfig`.
    random_state:
        Seed for the Isolation Forest.

    Attributes
    ----------
    fences_ : dict[str, tuple[float, float]]
        Per-column ``(lower, upper)`` winsorization bounds.
    isolation_forest_ : IsolationForest | None
        Fitted multivariate detector, when the method requires one.
    train_outlier_rate_ : float
        Share of training rows flagged - a useful sanity metric.

    Examples
    --------
    >>> import pandas as pd
    >>> from datapulse.config import OutlierConfig, OutlierMethod, OutlierAction
    >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
    >>> cfg = OutlierConfig(method=OutlierMethod.IQR, action=OutlierAction.CLIP)
    >>> float(OutlierEngine(cfg).fit_transform(df)["x"].max()) < 1000.0
    True

    """

    stage_name = "outliers"

    def __init__(
        self,
        config: Optional[OutlierConfig] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.config = config or OutlierConfig()
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ #
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        cfg = self.config
        self.numerical_columns_ = [
            c
            for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and c not in cfg.exclude_columns
        ]
        # Binary indicator columns must not be winsorised.
        self.numerical_columns_ = [
            c for c in self.numerical_columns_ if X[c].nunique(dropna=True) > 2
        ]

        self.fences_: dict[str, tuple[float, float]] = {}
        if cfg.enabled and cfg.method in (OutlierMethod.IQR, OutlierMethod.HYBRID):
            for column in self.numerical_columns_:
                self.fences_[column] = self._compute_fence(X[column])

        self.isolation_forest_ = None
        if (
            cfg.enabled
            and cfg.method in (OutlierMethod.ISOLATION_FOREST, OutlierMethod.HYBRID)
            and self.numerical_columns_
        ):
            self.isolation_forest_ = IsolationForest(
                n_estimators=cfg.n_estimators,
                contamination=cfg.contamination,
                max_samples=cfg.max_samples,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            block = self._numeric_block(X)
            self.isolation_forest_.fit(block)
            flags = self.isolation_forest_.predict(block) == -1
            self.train_outlier_rate_ = float(flags.mean())
        else:
            self.train_outlier_rate_ = 0.0

        self.fit_report_.update(
            {
                "method": cfg.method.value,
                "action": cfg.action.value,
                "n_columns_fenced": len(self.fences_),
                "train_outlier_rate": round(self.train_outlier_rate_, 4),
            }
        )

    # ------------------------------------------------------------------ #
    def _compute_fence(self, series: pd.Series) -> tuple[float, float]:
        """Return ``(lower, upper)`` using the wider of IQR and quantile fences."""
        cfg = self.config
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return (-np.inf, np.inf)

        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        iqr = q3 - q1
        iqr_low, iqr_high = q1 - cfg.iqr_multiplier * iqr, q3 + cfg.iqr_multiplier * iqr

        q_low, q_high = cfg.winsorize_quantiles
        quantile_low = float(values.quantile(q_low))
        quantile_high = float(values.quantile(q_high))

        # Taking the max/min keeps the *tighter* of the two fences, which is
        # what winsorization is for; degenerate (zero-IQR) columns fall back to
        # the empirical quantiles.
        lower = max(iqr_low, quantile_low) if iqr > 0 else quantile_low
        upper = min(iqr_high, quantile_high) if iqr > 0 else quantile_high
        if lower > upper:
            lower, upper = quantile_low, quantile_high
        return (lower, upper)

    def _numeric_block(self, X: pd.DataFrame) -> pd.DataFrame:
        """Numeric sub-frame with non-finite values neutralised."""
        block = X.reindex(columns=self.numerical_columns_).astype("float64")
        block = block.replace([np.inf, -np.inf], np.nan)
        return block.fillna(block.median(numeric_only=True)).fillna(0.0)

    # ------------------------------------------------------------------ #
    def detect(self, X: pd.DataFrame) -> pd.Series:
        """Return a boolean Series flagging multivariate outlier rows.

        Parameters
        ----------
        X:
            Frame with the fitted columns.

        Returns
        -------
        pandas.Series
            ``True`` where the row is anomalous. All ``False`` when no
            multivariate detector was fitted.

        """
        self._check_is_fitted()
        frame = self._as_frame(X)
        if self.isolation_forest_ is None:
            return pd.Series(False, index=frame.index)
        predictions = self.isolation_forest_.predict(self._numeric_block(frame))
        return pd.Series(predictions == -1, index=frame.index)

    def score_samples(self, X: pd.DataFrame) -> pd.Series:
        """Return the Isolation Forest anomaly score (higher = more normal)."""
        self._check_is_fitted()
        frame = self._as_frame(X)
        if self.isolation_forest_ is None:
            return pd.Series(0.0, index=frame.index)
        scores = self.isolation_forest_.score_samples(self._numeric_block(frame))
        return pd.Series(scores, index=frame.index)

    # ------------------------------------------------------------------ #
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if not cfg.enabled or cfg.method is OutlierMethod.NONE:
            return X

        out = X.copy()
        action = cfg.action

        if action is OutlierAction.DROP:
            # Row removal cannot happen at inference time: a scoring service
            # must return a prediction for every record it is given. Degrade
            # to the closest inference-safe behaviour.
            self._logger.debug(
                "action='drop' is training-only; clipping instead during transform. "
                "Use fit_resample(X, y) to actually remove training rows."
            )
            action = OutlierAction.CLIP

        if action in (OutlierAction.CLIP, OutlierAction.MASK):
            for column, (lower, upper) in self.fences_.items():
                if column not in out.columns:
                    continue
                values = pd.to_numeric(out[column], errors="coerce")
                if action is OutlierAction.CLIP:
                    out[column] = values.clip(lower=lower, upper=upper)
                else:
                    out[column] = values.where(values.between(lower, upper))

        if action is OutlierAction.FLAG or cfg.method is OutlierMethod.HYBRID:
            if self.isolation_forest_ is not None:
                out["__outlier_score"] = self.score_samples(X).to_numpy()
                out["__is_outlier"] = self.detect(X).astype("float64").to_numpy()

        return out

    # ------------------------------------------------------------------ #
    def fit_resample(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit, then *remove* outlier rows. Training-time use only.

        Mirrors the imbalanced-learn sampler API so the orchestrator can treat
        row-count-changing steps uniformly. When ``action`` is not ``drop`` this
        simply returns ``(transform(X), y)``.

        Parameters
        ----------
        X, y:
            Training features and labels.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.Series | None]
            Filtered, transformed features and the aligned target.

        """
        frame = self._as_frame(X)
        target = self._as_series(y, index=frame.index)
        self.fit(frame, target)

        if self.config.action is not OutlierAction.DROP or not self.config.enabled:
            return self.transform(frame), target

        flagged = self.detect(frame)
        for column, (lower, upper) in self.fences_.items():
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                flagged |= ~values.between(lower, upper) & values.notna()

        drop_fraction = float(flagged.mean())
        if drop_fraction > self.config.max_drop_fraction:
            # Refuse to amputate the training set: keep only the most extreme
            # rows up to the configured budget.
            self._logger.warning(
                "Outlier drop would remove %.1f%% of rows (> max_drop_fraction=%.1f%%); "
                "restricting to the most anomalous rows.",
                drop_fraction * 100,
                self.config.max_drop_fraction * 100,
            )
            budget = int(len(frame) * self.config.max_drop_fraction)
            scores = self.score_samples(frame)
            worst = scores.nsmallest(budget).index
            flagged = pd.Series(False, index=frame.index)
            flagged.loc[worst] = True

        keep = ~flagged
        self.fit_report_["n_rows_dropped"] = int((~keep).sum())
        self._logger.info(
            "Outlier engine dropped %d of %d training rows (%.2f%%).",
            int((~keep).sum()),
            len(frame),
            100 * float((~keep).mean()),
        )
        filtered = frame.loc[keep]
        filtered_target = target.loc[keep] if target is not None else None
        return self.transform(filtered), filtered_target
