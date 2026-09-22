"""Module 4 - Smart scaling and skew correction.

Blindly applying ``StandardScaler`` to everything is the most common
preprocessing mistake in production ML: it assumes each column is roughly
Gaussian, which monetary amounts, counts and durations never are.

:class:`SkewAwareScaler` makes a *per-column* decision:

1. Test normality - D'Agostino-Pearson (``scipy.stats.normaltest``) for
   n >= 20, Shapiro-Wilk below that (``AUTO`` mode picks for you).
2. Measure skewness.
3. Route the column:

   ==========================================  ===============================
   Condition                                   Treatment
   ==========================================  ===============================
   Passes normality and \\|skew\\| <= threshold   ``StandardScaler``
   \\|skew\\| > threshold and strictly positive   Box-Cox + optional standardise
   \\|skew\\| > threshold, has zeros/negatives    Yeo-Johnson + standardise
   Otherwise (heavy tails, non-normal)         ``RobustScaler`` (median/IQR)
   ==========================================  ===============================

Every decision, with its p-value and before/after skew, is recorded in
:attr:`SkewAwareScaler.decisions_` and rendered by the reporting suite.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from datapulse.base import AbstractTransformer
from datapulse.config import NormalityTest, ScalerKind, ScalingConfig
from datapulse.utils.validation import safe_sample

__all__ = ["SkewAwareScaler"]

_MIN_SAMPLES_FOR_NORMALTEST = 20
_SHAPIRO_MAX_SAMPLES = 5000


class SkewAwareScaler(AbstractTransformer):
    """Per-column normality testing followed by an adaptive transform.

    Parameters
    ----------
    config:
        A :class:`~datapulse.config.ScalingConfig`.
    random_state:
        Seed used when sub-sampling large columns for the normality test.

    Attributes
    ----------
    transformers_ : dict[str, sklearn.base.TransformerMixin]
        Fitted per-column transformer.
    decisions_ : pandas.DataFrame
        One row per column: test used, p-value, skew before/after, treatment.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"income": rng.lognormal(3, 1, 500)})
    >>> scaler = SkewAwareScaler().fit(df)
    >>> scaler.decisions_.loc[0, "treatment"]
    'box_cox'

    """

    stage_name = "scaling"

    def __init__(
        self,
        config: Optional[ScalingConfig] = None,
        random_state: int = 42,
    ) -> None:
        self.config = config or ScalingConfig()
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        cfg = self.config
        self.numerical_columns_ = [
            c
            for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c])
            and c not in cfg.exclude_columns
            and X[c].nunique(dropna=True) > 2
        ]
        self.passthrough_columns_ = [c for c in X.columns if c not in self.numerical_columns_]

        self.transformers_: dict[str, object] = {}
        records: list[dict[str, object]] = []

        if not cfg.enabled:
            self.decisions_ = pd.DataFrame(records)
            return

        for column in self.numerical_columns_:
            values = pd.to_numeric(X[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            clean = values.dropna()
            if clean.empty:
                self.transformers_[column] = None
                continue

            skew_before = float(stats.skew(clean.to_numpy())) if len(clean) > 2 else 0.0
            p_value, test_name = self._normality_p_value(clean.to_numpy())
            is_normal = p_value is not None and p_value > cfg.alpha
            is_skewed = abs(skew_before) > cfg.skew_threshold
            positive_only = bool((clean > 0).all())

            treatment, transformer = self._select_transformer(
                is_normal=is_normal, is_skewed=is_skewed, positive_only=positive_only
            )

            skew_after = skew_before
            if transformer is not None:
                reshaped = values.to_frame()
                filled = reshaped.fillna(clean.median())
                try:
                    transformer.fit(filled)
                    transformed = transformer.transform(filled).ravel()
                    if len(transformed) > 2:
                        skew_after = float(stats.skew(transformed))
                except (ValueError, FloatingPointError) as exc:
                    # Box-Cox can still fail on pathological data; fall back.
                    self._logger.warning(
                        "Transform %s failed on %r (%s); falling back to RobustScaler.",
                        treatment,
                        column,
                        exc,
                    )
                    treatment = "robust"
                    transformer = RobustScaler()
                    transformer.fit(filled)
                    skew_after = skew_before

            self.transformers_[column] = transformer
            records.append(
                {
                    "column": column,
                    "test": test_name,
                    "p_value": None if p_value is None else round(p_value, 6),
                    "is_normal": bool(is_normal),
                    "skew_before": round(skew_before, 4),
                    "skew_after": round(skew_after, 4),
                    "treatment": treatment,
                }
            )

        self.decisions_ = pd.DataFrame(records)
        if not self.decisions_.empty:
            counts = self.decisions_["treatment"].value_counts().to_dict()
            self.fit_report_["treatments"] = counts
            self._logger.info("Scaling decisions: %s", counts)

    # ------------------------------------------------------------------ #
    def _normality_p_value(self, values: np.ndarray) -> tuple[Optional[float], str]:
        """Run the configured normality test, returning ``(p_value, test_name)``."""
        cfg = self.config
        sample = safe_sample(values, cfg.max_test_sample, self.random_state)
        n = sample.size
        if n < 8 or np.allclose(sample, sample[0]):
            return None, "skipped"

        test = cfg.normality_test
        if test is NormalityTest.AUTO:
            test = (
                NormalityTest.DAGOSTINO
                if n >= _MIN_SAMPLES_FOR_NORMALTEST
                else NormalityTest.SHAPIRO
            )

        try:
            if test is NormalityTest.SHAPIRO:
                sample = safe_sample(sample, _SHAPIRO_MAX_SAMPLES, self.random_state)
                result = stats.shapiro(sample)
            else:
                result = stats.normaltest(sample)
            return float(result.pvalue), test.value
        except ValueError:
            return None, "failed"

    def _select_transformer(
        self, *, is_normal: bool, is_skewed: bool, positive_only: bool
    ) -> tuple[str, Optional[object]]:
        """Map the diagnostic flags onto a concrete transformer."""
        cfg = self.config

        if is_skewed:
            if positive_only and cfg.allow_box_cox:
                return "box_cox", PowerTransformer(
                    method="box-cox", standardize=cfg.standardize_after_power
                )
            return "yeo_johnson", PowerTransformer(
                method="yeo-johnson", standardize=cfg.standardize_after_power
            )

        kind = cfg.normal_scaler if is_normal else cfg.default_scaler
        return kind.value, self._build_scaler(kind)

    @staticmethod
    def _build_scaler(kind: ScalerKind) -> Optional[object]:
        """Instantiate a plain scaler from its enum."""
        if kind is ScalerKind.STANDARD:
            return StandardScaler()
        if kind is ScalerKind.ROBUST:
            return RobustScaler()
        if kind is ScalerKind.MINMAX:
            return MinMaxScaler()
        return None

    # ------------------------------------------------------------------ #
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enabled:
            return X

        out = X.copy()
        for column, transformer in self.transformers_.items():
            if column not in out.columns or transformer is None:
                continue
            values = pd.to_numeric(out[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            na_mask = values.isna()
            filled = values.fillna(values.median() if values.notna().any() else 0.0)
            transformed = transformer.transform(filled.to_frame()).ravel()
            series = pd.Series(transformed, index=out.index, dtype="float64")
            # Preserve the missingness pattern rather than inventing a value.
            out[column] = series.where(~na_mask, other=np.nan)
        return out

    # ------------------------------------------------------------------ #
    def skew_report(self) -> pd.DataFrame:
        """Return the per-column decision table (empty frame when disabled)."""
        self._check_is_fitted()
        return self.decisions_.copy()
