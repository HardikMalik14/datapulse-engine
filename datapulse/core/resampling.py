"""Module 6 - Class imbalance handling with hard leakage safeguards.

SMOTE and friends synthesise new minority-class rows by interpolating between
neighbours. That is legitimate on a training fold and catastrophic anywhere
else:

* Resampling a **test set** invents labelled rows that never existed and
  reports a metric on synthetic data.
* Resampling **before** the train/test split places interpolations of training
  rows into the test set, which is direct label leakage - the classic reason a
  notebook reports 0.98 F1 and the deployed model reports 0.41.

The safeguard implemented here is structural rather than advisory:
:class:`SmartResampler` exposes the sampling behaviour **only** through
:meth:`SmartResampler.fit_resample`. Its ``transform`` is an identity map, so
if the resampler is ever placed inside a standard Scikit-Learn ``Pipeline``,
holdout data simply passes through untouched. Calling
:meth:`SmartResampler.resample` on data after fitting raises
:class:`~datapulse.exceptions.LeakageGuardError`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, BorderlineSMOTE

from datapulse.base import AbstractTransformer
from datapulse.config import ResamplingConfig, ResamplingStrategy, TaskType
from datapulse.exceptions import LeakageGuardError, TransformerError

__all__ = ["SmartResampler"]


class SmartResampler(AbstractTransformer):
    """Synthetic oversampling that can only ever run on training data.

    Parameters
    ----------
    config:
        A :class:`~datapulse.config.ResamplingConfig`.
    task_type:
        Resampling is refused outright for regression targets.
    random_state:
        Seed for the sampler.
    n_jobs:
        Parallelism passed to the underlying imbalanced-learn sampler.

    Attributes
    ----------
    distribution_before_, distribution_after_ : dict
        Class counts before and after resampling (drives the report plot).
    was_applied_ : bool
        ``False`` when a safeguard vetoed resampling; the reason is in
        :attr:`skip_reason_`.
    skip_reason_ : str

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from datapulse.config import ResamplingConfig
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    >>> y = pd.Series([0] * 270 + [1] * 30)
    >>> Xr, yr = SmartResampler(ResamplingConfig()).fit_resample(X, y)
    >>> int(yr.value_counts().min()) == int(yr.value_counts().max())
    True
    >>> SmartResampler(ResamplingConfig()).fit(X, y).transform(X).shape
    (300, 4)

    """

    stage_name = "resampling"

    def __init__(
        self,
        config: Optional[ResamplingConfig] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.config = config or ResamplingConfig()
        self.task_type = task_type
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ #
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        self.distribution_before_: dict = {}
        self.distribution_after_: dict = {}
        self.was_applied_ = False
        self.skip_reason_ = ""
        if y is not None:
            self.distribution_before_ = self._counts(y)
        self.fit_report_["distribution_before"] = self.distribution_before_

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Identity. Resampling never happens outside :meth:`fit_resample`."""
        return X

    # ------------------------------------------------------------------ #
    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit and resample the **training** set.

        Parameters
        ----------
        X:
            Fully numeric training features (run this after encoding).
        y:
            Training labels.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.Series]
            Balanced features and labels. When a safeguard vetoes resampling,
            the inputs are returned unchanged and :attr:`skip_reason_` explains
            why.

        Raises
        ------
        TransformerError
            If the task is regression or the frame is not fully numeric.

        """
        frame = self._as_frame(X)
        target = self._as_series(y, index=frame.index)
        if target is None:
            raise TransformerError("SmartResampler.fit_resample requires `y`.")

        self.fit(frame, target)
        cfg = self.config

        veto = self._veto_reason(frame, target)
        if veto is not None:
            self.skip_reason_ = veto
            self.was_applied_ = False
            self.distribution_after_ = dict(self.distribution_before_)
            self.fit_report_.update(
                {"resampling_applied": False, "skip_reason": veto}
            )
            self._logger.info("Resampling skipped: %s", veto)
            return frame, target

        sampler = self._build_sampler(frame, target)
        try:
            resampled_x, resampled_y = sampler.fit_resample(frame, target)
        except (ValueError, RuntimeError) as exc:
            self.skip_reason_ = f"sampler failed: {exc}"
            self.was_applied_ = False
            self.distribution_after_ = dict(self.distribution_before_)
            self._logger.warning("Resampling failed, returning original data: %s", exc)
            return frame, target

        resampled_x = pd.DataFrame(
            np.asarray(resampled_x), columns=frame.columns
        ).reset_index(drop=True)
        resampled_y = pd.Series(
            np.asarray(resampled_y).ravel(), name=target.name or "target"
        ).reset_index(drop=True)

        self.was_applied_ = True
        self.distribution_after_ = self._counts(resampled_y)
        self.fit_report_.update(
            {
                "resampling_applied": True,
                "strategy": cfg.strategy.value,
                "distribution_after": self.distribution_after_,
                "n_synthetic_rows": int(len(resampled_y) - len(target)),
            }
        )
        self._logger.info(
            "%s: %s -> %s (+%d synthetic rows)",
            cfg.strategy.value,
            self.distribution_before_,
            self.distribution_after_,
            len(resampled_y) - len(target),
        )
        return resampled_x, resampled_y

    # ------------------------------------------------------------------ #
    def resample(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Always raises - resampling a non-training split is never valid.

        Raises
        ------
        LeakageGuardError
            Unconditionally.

        """
        raise LeakageGuardError(
            "Resampling can only be performed inside fit_resample() on a training "
            "split. Applying SMOTE/ADASYN to validation, test or production data "
            "fabricates labelled rows and invalidates every metric computed on them."
        )

    # ------------------------------------------------------------------ #
    def _veto_reason(self, X: pd.DataFrame, y: pd.Series) -> Optional[str]:
        """Return a human-readable veto reason, or ``None`` to proceed."""
        cfg = self.config
        if not cfg.enabled or cfg.strategy is ResamplingStrategy.NONE:
            return "disabled by configuration"
        if self.task_type is not TaskType.CLASSIFICATION:
            return "target is continuous (resampling is classification-only)"

        counts = pd.Series(np.asarray(y).ravel()).value_counts()
        if len(counts) < 2:
            return "only one class present"
        minority, majority = int(counts.min()), int(counts.max())
        if minority < cfg.min_minority_samples:
            return (
                f"minority class has {minority} rows (< min_minority_samples="
                f"{cfg.min_minority_samples}); interpolating so few points "
                "manufactures noise, not signal"
            )
        ratio = majority / max(minority, 1)
        if ratio < cfg.imbalance_ratio_trigger:
            return (
                f"imbalance ratio {ratio:.2f} below trigger "
                f"{cfg.imbalance_ratio_trigger}"
            )

        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric and cfg.strategy is not ResamplingStrategy.SMOTENC:
            raise TransformerError(
                "SMOTE-family samplers require a fully numeric matrix; run the "
                f"resampler after encoding. Offending columns: {non_numeric[:10]}"
            )
        if X.isna().any().any():
            return "features still contain NaN (run imputation first)"
        return None

    def _build_sampler(self, X: pd.DataFrame, y: pd.Series):  # noqa: ANN202
        """Instantiate the configured imbalanced-learn sampler."""
        cfg = self.config
        counts = pd.Series(np.asarray(y).ravel()).value_counts()
        # k must be strictly smaller than the minority class size.
        k_neighbors = max(1, min(cfg.k_neighbors, int(counts.min()) - 1))
        common = {
            "sampling_strategy": cfg.sampling_strategy,
            "random_state": self.random_state,
        }

        if cfg.strategy is ResamplingStrategy.SMOTE:
            return SMOTE(k_neighbors=k_neighbors, **common)
        if cfg.strategy is ResamplingStrategy.BORDERLINE_SMOTE:
            m_neighbors = max(2, min(cfg.m_neighbors, int(counts.min()) - 1))
            return BorderlineSMOTE(
                k_neighbors=k_neighbors, m_neighbors=m_neighbors, kind=cfg.kind, **common
            )
        if cfg.strategy is ResamplingStrategy.ADASYN:
            return ADASYN(n_neighbors=k_neighbors, **common)
        if cfg.strategy is ResamplingStrategy.SMOTENC:
            categorical_idx = [
                i
                for i, c in enumerate(X.columns)
                if not pd.api.types.is_numeric_dtype(X[c])
            ]
            if not categorical_idx:
                return SMOTE(k_neighbors=k_neighbors, **common)
            return SMOTENC(
                categorical_features=categorical_idx, k_neighbors=k_neighbors, **common
            )
        raise TransformerError(f"Unsupported resampling strategy: {cfg.strategy}")

    @staticmethod
    def _counts(y: pd.Series) -> dict:
        """Class counts as a plain, JSON-friendly dict."""
        counts = pd.Series(np.asarray(y).ravel()).value_counts().sort_index()
        return {str(k): int(v) for k, v in counts.items()}
