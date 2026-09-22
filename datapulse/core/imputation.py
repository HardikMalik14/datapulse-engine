"""Module 2 - Context-aware missing value imputation.

Different column types deserve different treatments, and *missingness itself*
is frequently predictive (a blank ``income`` field is not a random event). This
module therefore:

1. Drops columns that are emptier than a configurable threshold - imputing a
   90%-null column manufactures signal that does not exist.
2. Emits ``<col>__was_missing`` binary indicators before filling, so the model
   can learn from the missingness pattern.
3. Imputes numerical columns with mean/median/KNN/MICE-style iterative
   regression, and categorical columns with the mode, a distribution-preserving
   random draw, or an explicit sentinel category.

The KNN and iterative imputers are multivariate: they use the *other* columns
to estimate the missing one, which is why this module runs after schema
inference (so the neighbour distance is computed on real numbers) but before
scaling (so fences and scalers are fitted on complete data).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (side-effect import)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

from datapulse.base import AbstractTransformer
from datapulse.config import (
    CategoricalImputationStrategy,
    ImputationConfig,
    NumericalImputationStrategy,
)

__all__ = ["ContextAwareImputer"]


class ContextAwareImputer(AbstractTransformer):
    """Type-aware imputation with optional missingness indicators.

    Parameters
    ----------
    config:
        An :class:`~datapulse.config.ImputationConfig`.
    random_state:
        Seed for the stochastic strategies (iterative imputation, frequency
        draws).

    Attributes
    ----------
    numerical_columns_, categorical_columns_ : list[str]
        Columns routed to each imputer.
    dropped_columns_ : list[str]
        Columns removed for exceeding the null-rate threshold.
    indicator_columns_ : list[str]
        Names of the generated ``__was_missing`` flags.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from datapulse.config import ImputationConfig, NumericalImputationStrategy
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "x"]})
    >>> cfg = ImputationConfig(numerical_strategy=NumericalImputationStrategy.MEDIAN)
    >>> ContextAwareImputer(cfg).fit_transform(df).isna().sum().sum()
    np.int64(0)

    """

    stage_name = "imputation"

    def __init__(
        self,
        config: Optional[ImputationConfig] = None,
        random_state: int = 42,
    ) -> None:
        self.config = config or ImputationConfig()
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        cfg = self.config

        na_rates = X.isna().mean()
        self.na_rates_ = {c: float(na_rates[c]) for c in X.columns}
        self.dropped_columns_ = [
            c for c in X.columns if na_rates[c] > cfg.drop_columns_above_na_rate
        ]
        working = X.drop(columns=self.dropped_columns_)

        self.numerical_columns_ = [
            c for c in working.columns if pd.api.types.is_numeric_dtype(working[c])
        ]
        self.categorical_columns_ = [
            c for c in working.columns if c not in self.numerical_columns_
        ]

        self.indicator_columns_ = []
        if cfg.add_missing_indicators:
            self.indicator_columns_ = [
                f"{c}__was_missing"
                for c in working.columns
                if na_rates[c] >= cfg.missing_indicator_min_rate
            ]

        self.numerical_imputer_ = self._build_numerical_imputer()
        if self.numerical_imputer_ is not None and self.numerical_columns_:
            # Fit on a writable float array: under pandas copy-on-write the
            # DataFrame's buffer can be read-only, which IterativeImputer's
            # in-place round-robin updates cannot use.
            self.numerical_imputer_.fit(self._numeric_matrix(working))

        self.categorical_fill_: dict[str, object] = {}
        self.categorical_distribution_: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for column in self.categorical_columns_:
            values = working[column].dropna()
            if values.empty:
                self.categorical_fill_[column] = cfg.categorical_fill_value
                continue
            counts = values.value_counts()
            if cfg.categorical_strategy is CategoricalImputationStrategy.CONSTANT:
                self.categorical_fill_[column] = cfg.categorical_fill_value
            else:
                self.categorical_fill_[column] = counts.index[0]
            if cfg.categorical_strategy is CategoricalImputationStrategy.FREQUENCY:
                probabilities = (counts / counts.sum()).to_numpy(dtype=float)
                self.categorical_distribution_[column] = (
                    counts.index.to_numpy(),
                    probabilities,
                )

        self.fit_report_.update(
            {
                "dropped_high_na_columns": self.dropped_columns_,
                "n_numerical_imputed": len(self.numerical_columns_),
                "n_categorical_imputed": len(self.categorical_columns_),
                "n_missing_indicators": len(self.indicator_columns_),
                "total_missing_cells": int(X.isna().sum().sum()),
            }
        )
        if self.dropped_columns_:
            self._logger.warning(
                "Dropping %d column(s) above %.0f%% missing: %s",
                len(self.dropped_columns_),
                cfg.drop_columns_above_na_rate * 100,
                self.dropped_columns_,
            )

    # ------------------------------------------------------------------ #
    def _build_numerical_imputer(self):  # noqa: ANN202 - sklearn estimator
        """Instantiate the numerical imputer described by the config."""
        cfg = self.config
        strategy = cfg.numerical_strategy

        if strategy is NumericalImputationStrategy.KNN:
            return KNNImputer(
                n_neighbors=cfg.knn_neighbors,
                weights=cfg.knn_weights,
                keep_empty_features=True,
            )
        if strategy is NumericalImputationStrategy.ITERATIVE:
            estimators = {
                "bayesian_ridge": BayesianRidge(),
                "random_forest": RandomForestRegressor(
                    n_estimators=50, max_depth=10, random_state=self.random_state, n_jobs=1
                ),
                "extra_trees": ExtraTreesRegressor(
                    n_estimators=50, max_depth=10, random_state=self.random_state, n_jobs=1
                ),
            }
            return IterativeImputer(
                estimator=estimators[cfg.iterative_estimator],
                max_iter=cfg.iterative_max_iter,
                random_state=self.random_state,
                keep_empty_features=True,
            )
        if strategy is NumericalImputationStrategy.CONSTANT:
            return SimpleImputer(
                strategy="constant",
                fill_value=cfg.numerical_fill_value,
                keep_empty_features=True,
            )
        return SimpleImputer(strategy=strategy.value, keep_empty_features=True)

    # ------------------------------------------------------------------ #
    def _numeric_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a writable float matrix of the fitted numerical columns."""
        block = frame.reindex(columns=self.numerical_columns_)
        return np.array(block.to_numpy(dtype="float64"), dtype="float64", copy=True)

    # ------------------------------------------------------------------ #
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        out = X.drop(columns=[c for c in self.dropped_columns_ if c in X.columns])

        indicators: dict[str, pd.Series] = {}
        for name in self.indicator_columns_:
            source = name.removesuffix("__was_missing")
            if source in out.columns:
                indicators[name] = out[source].isna().astype("float64")
            else:  # column vanished upstream; keep the width stable
                indicators[name] = pd.Series(0.0, index=out.index)

        if self.numerical_columns_ and self.numerical_imputer_ is not None:
            imputed = self.numerical_imputer_.transform(self._numeric_matrix(out))
            out[self.numerical_columns_] = pd.DataFrame(
                imputed, columns=self.numerical_columns_, index=out.index
            )
            # keep_empty_features leaves all-NaN columns untouched; zero them.
            out[self.numerical_columns_] = out[self.numerical_columns_].fillna(
                cfg.numerical_fill_value
            )

        rng = np.random.default_rng(self.random_state)
        for column in self.categorical_columns_:
            if column not in out.columns:
                continue
            mask = out[column].isna()
            if not mask.any():
                continue
            if (
                cfg.categorical_strategy is CategoricalImputationStrategy.FREQUENCY
                and column in self.categorical_distribution_
            ):
                categories, probabilities = self.categorical_distribution_[column]
                draws = rng.choice(categories, size=int(mask.sum()), p=probabilities)
                out.loc[mask, column] = draws
            else:
                out.loc[mask, column] = self.categorical_fill_.get(
                    column, cfg.categorical_fill_value
                )

        if indicators:
            out = pd.concat([out, pd.DataFrame(indicators, index=out.index)], axis=1)
        return out
