"""The framework contract: :class:`AbstractTransformer`.

Every DataPulse module is a Scikit-Learn transformer, but with three
guarantees that vanilla ``ColumnTransformer``/``Pipeline`` stacks do not give
you:

* **DataFrames in, DataFrames out.** Column names, dtypes and the index survive
  the whole chain, so a drift report or a SHAP plot downstream can still say
  *which* feature moved.
* **Schema enforcement.** The set of columns seen at ``fit`` time is recorded;
  at ``transform`` time a mismatch raises :class:`~datapulse.exceptions.SchemaError`
  instead of silently mis-aligning positional data.
* **Uniform diagnostics.** Fit duration, row/column deltas and per-transformer
  notes are captured in :attr:`AbstractTransformer.fit_report_`.

Subclasses implement ``_fit`` and ``_transform`` only; validation, timing,
logging and error wrapping are handled by the template methods here.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from datapulse.exceptions import NotFittedError, SchemaError, TransformerError
from datapulse.logger import get_logger

__all__ = ["AbstractTransformer", "FitReport"]

ArrayLike = Union[pd.DataFrame, np.ndarray, Sequence[Sequence[Any]]]
TargetLike = Optional[Union[pd.Series, np.ndarray, Sequence[Any]]]


class FitReport(dict):
    """A small ``dict`` subclass carrying per-transformer fit diagnostics.

    Using a ``dict`` keeps the object trivially picklable and JSON-friendly
    while still allowing attribute-style access for readability.
    """

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - convenience
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        inner = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"FitReport({inner})"


class AbstractTransformer(BaseEstimator, TransformerMixin, ABC):
    """Base class for all DataPulse transformers.

    Attributes
    ----------
    feature_names_in_ : list[str]
        Columns observed during ``fit``.
    feature_names_out_ : list[str]
        Columns produced by ``transform``.
    n_features_in_ : int
        Number of input columns (Scikit-Learn convention).
    fit_report_ : FitReport
        Diagnostics recorded during the last ``fit``.
    is_fitted_ : bool
        Whether ``fit`` has completed successfully.

    Notes
    -----
    ``requires_y`` should be overridden by subclasses that genuinely need the
    target (target encoding, mutual information, RFE). It exists so the
    orchestrator can fail loudly when a supervised transformer is fitted
    without labels, rather than producing a silently degenerate encoding.

    """

    #: Set to ``True`` in subclasses that cannot be fitted without ``y``.
    requires_y: bool = False

    #: Human-friendly stage name used in logs and reports.
    stage_name: str = "abstract"

    # ------------------------------------------------------------------ #
    # Scikit-Learn API
    # ------------------------------------------------------------------ #
    def fit(self, X: ArrayLike, y: TargetLike = None) -> AbstractTransformer:
        """Learn the transformation parameters from ``X`` (and optionally ``y``).

        Parameters
        ----------
        X:
            Feature matrix. Anything convertible to a :class:`pandas.DataFrame`.
        y:
            Target vector, required when :attr:`requires_y` is ``True``.

        Returns
        -------
        AbstractTransformer
            ``self``, per the Scikit-Learn contract.

        Raises
        ------
        TransformerError
            If the underlying ``_fit`` implementation fails.

        """
        frame = self._as_frame(X)
        target = self._as_series(y, index=frame.index)

        if self.requires_y and target is None:
            raise TransformerError(
                f"{type(self).__name__} is a supervised transformer and requires `y` at fit time."
            )

        self.feature_names_in_ = list(map(str, frame.columns))
        self.n_features_in_ = frame.shape[1]
        self.is_fitted_ = False
        self.fit_report_ = FitReport(
            stage=self.stage_name,
            transformer=type(self).__name__,
            n_rows_in=int(frame.shape[0]),
            n_features_in=int(frame.shape[1]),
        )

        started = time.perf_counter()
        try:
            self._fit(frame, target)
        except Exception as exc:  # noqa: BLE001 - re-raised as framework error
            raise TransformerError(
                f"{type(self).__name__}.fit failed: {exc}"
            ) from exc
        elapsed = time.perf_counter() - started

        self.is_fitted_ = True
        self.fit_report_["fit_seconds"] = round(elapsed, 4)
        self._logger.debug(
            "%s fitted on %s rows x %s cols in %.3fs",
            type(self).__name__,
            frame.shape[0],
            frame.shape[1],
            elapsed,
        )
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Apply the learnt transformation and return a :class:`pandas.DataFrame`.

        Parameters
        ----------
        X:
            Feature matrix with the same columns seen at ``fit`` time.

        Returns
        -------
        pandas.DataFrame
            Transformed frame with meaningful column names.

        """
        self._check_is_fitted()
        frame = self._as_frame(X)
        frame = self._align_to_schema(frame)

        try:
            result = self._transform(frame)
        except Exception as exc:  # noqa: BLE001
            raise TransformerError(
                f"{type(self).__name__}.transform failed: {exc}"
            ) from exc

        result = self._as_frame(result)
        self.feature_names_out_ = list(map(str, result.columns))
        return result

    def fit_transform(
        self, X: ArrayLike, y: TargetLike = None, **fit_params: Any
    ) -> pd.DataFrame:
        """Fit then transform.

        Subclasses override this when the *training* output must differ from
        the inference output - most importantly out-of-fold target encoding,
        where re-using the in-fold statistics would leak the label.
        """
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Return output feature names (Scikit-Learn ``>=1.0`` contract)."""
        self._check_is_fitted()
        names = getattr(self, "feature_names_out_", None)
        if names is None:
            names = list(input_features or self.feature_names_in_)
        return np.asarray(names, dtype=object)

    # ------------------------------------------------------------------ #
    # Template methods for subclasses
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """Learn parameters. Implemented by every concrete transformer."""

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learnt parameters to ``X``."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def _logger(self):  # noqa: ANN202 - logging.Logger
        """Lazily-created logger (kept out of ``__init__`` so pickling is clean)."""
        return get_logger(f"{type(self).__module__}.{type(self).__name__}")

    def _check_is_fitted(self) -> None:
        """Raise :class:`NotFittedError` when the transformer is unfitted."""
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(
                f"{type(self).__name__} is not fitted yet. Call `fit` before `transform`."
            )

    @staticmethod
    def _as_frame(X: ArrayLike) -> pd.DataFrame:
        """Coerce any array-like into a DataFrame with string column names."""
        if isinstance(X, pd.DataFrame):
            frame = X
        elif isinstance(X, pd.Series):
            frame = X.to_frame()
        else:
            array = np.asarray(X)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            frame = pd.DataFrame(
                array, columns=[f"feature_{i}" for i in range(array.shape[1])]
            )
        if not all(isinstance(c, str) for c in frame.columns):
            frame = frame.rename(columns=str)
        return frame

    @staticmethod
    def _as_series(y: TargetLike, index: Optional[pd.Index] = None) -> Optional[pd.Series]:
        """Coerce a target-like object into a ``Series`` aligned to ``index``."""
        if y is None:
            return None
        if isinstance(y, pd.Series):
            return y
        array = np.asarray(y).ravel()
        return pd.Series(array, index=index if index is not None else None, name="target")

    def _align_to_schema(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate and reorder incoming columns to match the fitted schema.

        Extra columns are dropped with a warning (common when a serving payload
        carries metadata); missing columns are a hard error because imputing a
        whole feature would produce a confidently wrong prediction.
        """
        expected = list(self.feature_names_in_)
        incoming = list(map(str, frame.columns))
        if incoming == expected:
            return frame

        missing = set(expected) - set(incoming)
        extra = set(incoming) - set(expected)
        if missing:
            raise SchemaError.from_columns(missing, extra)
        if extra:
            self._logger.warning(
                "%s: dropping %d unexpected column(s) not seen at fit time: %s",
                type(self).__name__,
                len(extra),
                sorted(extra)[:10],
            )
        return frame.loc[:, expected]

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #
    def __repr__(self, N_CHAR_MAX: int = 700) -> str:  # noqa: N803 - sklearn signature
        state = "fitted" if getattr(self, "is_fitted_", False) else "unfitted"
        return f"{type(self).__name__}(stage={self.stage_name!r}, {state})"
