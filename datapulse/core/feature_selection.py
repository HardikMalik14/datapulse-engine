"""Module 7 - Hybrid feature selection suite.

No single selector is trustworthy on its own, so this module chains four
stages of increasing cost and increasing awareness of the target:

1. **Variance threshold** (unsupervised, O(n)). Removes constant and
   near-constant columns - typically one-hot dummies for levels that survived
   the frequency filter but appear a handful of times.
2. **Correlation pruning** (unsupervised). Greedily drops one member of each
   pair above ``drop_correlated_above``, preferring to keep the column with the
   higher variance. Collinear features destabilise linear coefficients and
   split importance arbitrarily in trees.
3. **Mutual information** (supervised, non-parametric). Captures non-linear and
   non-monotonic dependence that Pearson correlation misses.
4. **L1-based RFE** (supervised, model-driven). Recursive Feature Elimination
   around an L1-penalised linear model: the sparsity penalty zeroes redundant
   coefficients, and RFE removes them in batches, re-fitting each time so the
   remaining features are judged *jointly* rather than marginally.

``protected_features`` bypass every stage - some columns are required for
regulatory or business reasons regardless of what the statistics say.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import (
    RFE,
    RFECV,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LogisticRegression

from datapulse.base import AbstractTransformer
from datapulse.config import FeatureSelectionConfig, MutualInfoMode, TaskType

__all__ = ["HybridFeatureSelector", "l1_logistic_regression"]


def _sklearn_at_least(major: int, minor: int) -> bool:
    """Version gate for Scikit-Learn API changes."""
    parts = sklearn.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (IndexError, ValueError):  # pragma: no cover - dev builds
        return False


def l1_logistic_regression(C: float, random_state: int) -> LogisticRegression:
    """Build an L1-penalised logistic regression across Scikit-Learn versions.

    ``penalty="l1"`` was deprecated in Scikit-Learn 1.8 in favour of
    ``l1_ratio=1.0``; this keeps the framework warning-free on 1.8+ while still
    working on earlier releases.
    """
    common = {"C": C, "solver": "liblinear", "max_iter": 2_000, "random_state": random_state}
    if _sklearn_at_least(1, 8):
        return LogisticRegression(l1_ratio=1.0, **common)
    return LogisticRegression(penalty="l1", **common)


class HybridFeatureSelector(AbstractTransformer):
    """Variance -> correlation -> mutual information -> L1-RFE selection chain.

    Parameters
    ----------
    config:
        A :class:`~datapulse.config.FeatureSelectionConfig`.
    task_type:
        Drives the choice of MI estimator and RFE base model.
    random_state, n_jobs:
        Standard reproducibility / parallelism knobs.

    Attributes
    ----------
    selected_features_ : list[str]
        Surviving columns, in input order.
    selection_report_ : pandas.DataFrame
        Per-feature audit: variance, MI score, RFE rank, stage that dropped it.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from datapulse.config import FeatureSelectionConfig, TaskType
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(400, 8)), columns=[f"f{i}" for i in range(8)])
    >>> X["constant"] = 1.0
    >>> y = pd.Series((X["f0"] + X["f1"] > 0).astype(int))
    >>> sel = HybridFeatureSelector(FeatureSelectionConfig(min_features_to_keep=3),
    ...                             TaskType.CLASSIFICATION)
    >>> "constant" in sel.fit(X, y).selected_features_
    False

    """

    stage_name = "feature_selection"
    requires_y = True

    def __init__(
        self,
        config: Optional[FeatureSelectionConfig] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.config = config or FeatureSelectionConfig()
        self.task_type = task_type
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ #
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        cfg = self.config
        numeric = X.select_dtypes(include=[np.number]).copy()
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        protected = [c for c in cfg.protected_features if c in numeric.columns]
        candidates = list(numeric.columns)
        dropped_by: dict[str, str] = {}

        if not cfg.enabled:
            self.selected_features_ = list(X.columns)
            self.selection_report_ = pd.DataFrame(
                {"feature": self.selected_features_, "dropped_by": ""}
            )
            return

        target = pd.Series(np.asarray(y).ravel(), index=numeric.index)

        # --- stage 1: variance --------------------------------------- #
        variances = numeric.var(axis=0, ddof=0)
        survivors = [
            c
            for c in candidates
            if c in protected or float(variances.get(c, 0.0)) > cfg.variance_threshold
        ]
        for column in candidates:
            if column not in survivors:
                dropped_by[column] = "variance"
        candidates = survivors

        # --- stage 2: correlation ------------------------------------ #
        correlation_dropped: list[str] = []
        if cfg.drop_correlated_above is not None and len(candidates) > 1:
            correlation_dropped = self._correlated_to_drop(
                numeric[candidates], threshold=cfg.drop_correlated_above, protected=protected
            )
            for column in correlation_dropped:
                dropped_by[column] = "correlation"
            candidates = [c for c in candidates if c not in correlation_dropped]

        # --- stage 3: mutual information ----------------------------- #
        mi_scores = pd.Series(dtype="float64")
        if cfg.mutual_info_enabled and candidates:
            mi_scores = self._mutual_information(numeric[candidates], target)
            keep = self._apply_mi_rule(mi_scores, protected)
            for column in candidates:
                if column not in keep:
                    dropped_by[column] = "mutual_information"
            candidates = [c for c in candidates if c in keep]

        # --- stage 4: L1-based RFE ------------------------------------ #
        rfe_ranking = pd.Series(dtype="float64")
        if cfg.rfe_enabled and len(candidates) > cfg.min_features_to_keep:
            keep, rfe_ranking = self._recursive_elimination(
                numeric[candidates], target, protected
            )
            for column in candidates:
                if column not in keep:
                    dropped_by[column] = "rfe"
            candidates = [c for c in candidates if c in keep]

        # --- safety net ---------------------------------------------- #
        if len(candidates) < cfg.min_features_to_keep:
            ranked = (
                mi_scores.sort_values(ascending=False).index.tolist()
                if not mi_scores.empty
                else list(numeric.columns)
            )
            for column in ranked:
                if len(candidates) >= cfg.min_features_to_keep:
                    break
                if column not in candidates:
                    candidates.append(column)
                    dropped_by.pop(column, None)
            self._logger.warning(
                "Selection fell below min_features_to_keep=%d; restored top-MI features.",
                cfg.min_features_to_keep,
            )

        for column in protected:
            if column not in candidates:
                candidates.append(column)
                dropped_by.pop(column, None)

        self.selected_features_ = [c for c in numeric.columns if c in set(candidates)]
        self.selection_report_ = pd.DataFrame(
            {
                "feature": list(numeric.columns),
                "variance": [float(variances.get(c, np.nan)) for c in numeric.columns],
                "mutual_info": [float(mi_scores.get(c, np.nan)) for c in numeric.columns],
                "rfe_rank": [float(rfe_ranking.get(c, np.nan)) for c in numeric.columns],
                "selected": [c in set(self.selected_features_) for c in numeric.columns],
                "dropped_by": [dropped_by.get(c, "") for c in numeric.columns],
            }
        ).sort_values(["selected", "mutual_info"], ascending=[False, False])

        self.fit_report_.update(
            {
                "n_features_before": int(X.shape[1]),
                "n_features_after": len(self.selected_features_),
                "dropped_by_stage": pd.Series(dropped_by).value_counts().to_dict()
                if dropped_by
                else {},
            }
        )
        self._logger.info(
            "Feature selection: %d -> %d features (%s).",
            X.shape[1],
            len(self.selected_features_),
            self.fit_report_["dropped_by_stage"],
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _correlated_to_drop(
        frame: pd.DataFrame, *, threshold: float, protected: list[str]
    ) -> list[str]:
        """Greedily drop one column from each highly-correlated pair."""
        correlation = frame.corr(numeric_only=True).abs()
        if correlation.empty:
            return []
        upper = correlation.where(
            np.triu(np.ones(correlation.shape, dtype=bool), k=1)
        )
        variances = frame.var(axis=0, ddof=0)
        to_drop: set[str] = set()
        for column in upper.columns:
            partners = upper.index[upper[column] > threshold].tolist()
            for partner in partners:
                if column in to_drop or partner in to_drop:
                    continue
                # Keep the more informative (higher variance) of the pair.
                loser = column if variances.get(column, 0) < variances.get(partner, 0) else partner
                if loser not in protected:
                    to_drop.add(loser)
        return sorted(to_drop)

    def _mutual_information(self, frame: pd.DataFrame, target: pd.Series) -> pd.Series:
        """Compute mutual information between each feature and the target."""
        estimator = (
            mutual_info_classif
            if self.task_type is TaskType.CLASSIFICATION
            else mutual_info_regression
        )
        values = estimator(
            frame.to_numpy(dtype="float64"),
            np.asarray(target).ravel(),
            random_state=self.random_state,
        )
        return pd.Series(values, index=frame.columns).sort_values(ascending=False)

    def _apply_mi_rule(self, scores: pd.Series, protected: list[str]) -> set[str]:
        """Translate the configured MI mode into a keep-set."""
        cfg = self.config
        if scores.empty:
            return set(protected)
        if cfg.mutual_info_mode is MutualInfoMode.TOP_K:
            keep = set(scores.nlargest(min(cfg.mutual_info_top_k, len(scores))).index)
        elif cfg.mutual_info_mode is MutualInfoMode.THRESHOLD:
            keep = set(scores[scores > cfg.mutual_info_threshold].index)
        else:
            n_keep = max(
                cfg.min_features_to_keep,
                int(np.ceil(len(scores) * cfg.mutual_info_percentile)),
            )
            keep = set(scores.nlargest(min(n_keep, len(scores))).index)
        return keep | set(protected)

    def _recursive_elimination(
        self, frame: pd.DataFrame, target: pd.Series, protected: list[str]
    ) -> tuple[set[str], pd.Series]:
        """Run L1-penalised RFE (optionally cross-validated)."""
        cfg = self.config
        n_features = frame.shape[1]
        n_select = cfg.rfe_n_features or max(
            cfg.min_features_to_keep, int(np.ceil(n_features * cfg.rfe_fraction))
        )
        n_select = int(min(max(n_select, 1), n_features))

        if self.task_type is TaskType.CLASSIFICATION:
            estimator = l1_logistic_regression(cfg.l1_C, self.random_state)
        else:
            estimator = Lasso(alpha=1.0 / cfg.l1_C, random_state=self.random_state)

        matrix = frame.to_numpy(dtype="float64")
        labels = np.asarray(target).ravel()

        try:
            if cfg.rfe_use_cv:
                selector = RFECV(
                    estimator=estimator,
                    step=cfg.rfe_step,
                    cv=cfg.rfe_cv_folds,
                    min_features_to_select=max(cfg.min_features_to_keep, 1),
                    n_jobs=self.n_jobs,
                )
            else:
                selector = RFE(
                    estimator=estimator, n_features_to_select=n_select, step=cfg.rfe_step
                )
            selector.fit(matrix, labels)
            ranking = pd.Series(selector.ranking_, index=frame.columns, dtype="float64")
            keep = set(frame.columns[selector.support_])
        except (ValueError, RuntimeError) as exc:
            self._logger.warning("RFE failed (%s); keeping all candidates.", exc)
            return set(frame.columns), pd.Series(1.0, index=frame.columns)

        return keep | set(protected), ranking

    # ------------------------------------------------------------------ #
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            self._logger.warning(
                "Selected features absent from input, filling with 0.0: %s", missing[:10]
            )
        return X.reindex(columns=self.selected_features_).fillna(0.0)

    # ------------------------------------------------------------------ #
    def ranking(self) -> pd.DataFrame:
        """Return the per-feature audit table."""
        self._check_is_fitted()
        return self.selection_report_.copy()
