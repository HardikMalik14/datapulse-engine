"""Module 5 - Categorical encoding with leakage-safe target encoding.

Cardinality decides the strategy:

* **Low cardinality** (``<= one_hot_max_cardinality``) -> ``OneHotEncoder`` with
  ``min_frequency`` collapsing of rare levels and ``handle_unknown`` safety for
  categories that only appear in production.
* **High cardinality** -> **out-of-fold target encoding**. One-hot encoding a
  10,000-level ``zip_code`` produces a sparse disaster; naive target encoding
  produces something worse - a feature that has already seen the label.

Why out-of-fold matters
-----------------------
Naive target encoding computes ``mean(y | category)`` on the full training set
and applies it to that same training set. Every row therefore contributes to
its own encoded value, and for a rare category the encoding *is* the label.
Validation scores look excellent; production performance collapses.

The fix implemented here is K-fold out-of-fold encoding: to encode fold *i*,
statistics are computed using folds ``!= i`` only. The mapping applied at
inference time is fitted on all of the training data, which is legitimate
because the test rows never contributed to it. Encodings are additionally
shrunk toward the global prior with an m-estimate::

    encoding = (n_c * mean_c + m * prior) / (n_c + m)

so a category seen twice is pulled back toward the base rate rather than
trusted outright.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from datapulse.base import AbstractTransformer
from datapulse.config import EncodingConfig, TaskType
from datapulse.exceptions import TransformerError
from datapulse.utils.validation import ensure_unique_columns, sanitize_column_name

__all__ = ["CategoricalEncoderPipeline", "TargetEncoderOOF"]

_MAX_MULTICLASS_TARGET_ENCODING = 10


class TargetEncoderOOF:
    """Smoothed, out-of-fold target encoder for a set of columns.

    This is a plain helper (not a transformer) used by
    :class:`CategoricalEncoderPipeline`; keeping it separate makes the leakage
    logic unit-testable in isolation.

    Parameters
    ----------
    columns:
        Column names to encode.
    task_type:
        Classification or regression.
    n_folds:
        Number of out-of-fold splits.
    smoothing:
        The ``m`` in the m-estimate; larger values shrink harder toward the prior.
    noise:
        Standard deviation of optional Gaussian noise added to OOF values.
    unseen:
        What to emit for categories not seen at fit time.
    random_state:
        Seed for fold assignment and noise.

    """

    def __init__(
        self,
        columns: list[str],
        *,
        task_type: TaskType,
        n_folds: int = 5,
        smoothing: float = 10.0,
        noise: float = 0.0,
        unseen: str = "prior",
        random_state: int = 42,
    ) -> None:
        self.columns = list(columns)
        self.task_type = task_type
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.noise = noise
        self.unseen = unseen
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> TargetEncoderOOF:
        """Learn full-data mappings (used at inference time)."""
        self.targets_ = self._encode_target(y)
        self.priors_ = {name: float(values.mean()) for name, values in self.targets_.items()}
        self.mappings_: dict[str, dict[str, pd.Series]] = {}
        for column in self.columns:
            self.mappings_[column] = {
                name: self._smoothed_map(X[column], values, self.priors_[name])
                for name, values in self.targets_.items()
            }
        self.output_columns_ = [
            self._name(column, target_name)
            for column in self.columns
            for target_name in self.targets_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the full-data mappings. Safe for holdout and production data."""
        data: dict[str, np.ndarray] = {}
        for column in self.columns:
            keys = X[column].astype("object")
            for target_name, mapping in self.mappings_[column].items():
                prior = self.priors_[target_name]
                fill = {"prior": prior, "zero": 0.0, "nan": np.nan}[self.unseen]
                encoded = keys.map(mapping).astype("float64").fillna(fill)
                data[self._name(column, target_name)] = encoded.to_numpy()
        return pd.DataFrame(data, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit, then return **out-of-fold** encodings for the training rows."""
        self.fit(X, y)
        if not self.columns:
            return pd.DataFrame(index=X.index)

        splitter = self._make_splitter(y)
        data = {name: np.full(len(X), np.nan) for name in self.output_columns_}
        positions = np.arange(len(X))

        for train_idx, valid_idx in splitter.split(positions, self._stratify_labels(y)):
            for column in self.columns:
                in_fold_keys = X[column].to_numpy()[train_idx]
                valid_keys = pd.Series(X[column].to_numpy()[valid_idx]).astype("object")
                for target_name, target_values in self.targets_.items():
                    fold_target = pd.Series(target_values.to_numpy()[train_idx])
                    prior = float(fold_target.mean())
                    mapping = self._smoothed_map(
                        pd.Series(in_fold_keys), fold_target, prior
                    )
                    encoded = valid_keys.map(mapping).astype("float64").fillna(prior)
                    data[self._name(column, target_name)][valid_idx] = encoded.to_numpy()

        frame = pd.DataFrame(data, index=X.index)
        if self.noise > 0:
            rng = np.random.default_rng(self.random_state)
            frame += rng.normal(0.0, self.noise, size=frame.shape)
        # Any row a fold could not reach (degenerate splits) falls back to the
        # full-data mapping rather than staying NaN.
        if frame.isna().any().any():
            fallback = self.transform(X)
            frame = frame.fillna(fallback)
        return frame

    # ------------------------------------------------------------------ #
    def _encode_target(self, y: pd.Series) -> dict[str, pd.Series]:
        """Turn the target into one or more numeric columns to average."""
        series = pd.Series(np.asarray(y).ravel())
        if self.task_type is TaskType.REGRESSION:
            return {"te": pd.to_numeric(series, errors="coerce").fillna(0.0)}

        classes = pd.unique(series.dropna())
        if len(classes) <= 2:
            positive = sorted(classes, key=str)[-1] if len(classes) else 1
            return {"te": (series == positive).astype("float64")}
        if len(classes) > _MAX_MULTICLASS_TARGET_ENCODING:
            raise TransformerError(
                f"Target encoding supports at most {_MAX_MULTICLASS_TARGET_ENCODING} "
                f"classes; got {len(classes)}. Disable target encoding or bucket the target."
            )
        return {
            f"te_{sanitize_column_name(str(cls))}": (series == cls).astype("float64")
            for cls in sorted(classes, key=str)
        }

    def _smoothed_map(
        self, keys: pd.Series, target: pd.Series, prior: float
    ) -> pd.Series:
        """m-estimate smoothed ``mean(target | key)``."""
        frame = pd.DataFrame(
            {"key": keys.astype("object").to_numpy(), "target": target.to_numpy()}
        )
        grouped = frame.groupby("key", dropna=False)["target"].agg(["count", "mean"])
        smoothed = (grouped["count"] * grouped["mean"] + self.smoothing * prior) / (
            grouped["count"] + self.smoothing
        )
        return smoothed

    def _make_splitter(self, y: pd.Series):  # noqa: ANN202 - sklearn splitter
        n_splits = max(2, min(self.n_folds, len(y) // 2))
        if self.task_type is TaskType.CLASSIFICATION:
            labels = self._stratify_labels(y)
            counts = pd.Series(labels).value_counts()
            if counts.min() >= n_splits:
                return StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=self.random_state
                )
        return KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    @staticmethod
    def _stratify_labels(y: pd.Series) -> np.ndarray:
        return np.asarray(y).ravel()

    @staticmethod
    def _name(column: str, target_name: str) -> str:
        return f"{column}__{target_name}"


class CategoricalEncoderPipeline(AbstractTransformer):
    """Route categorical columns to one-hot or out-of-fold target encoding.

    Parameters
    ----------
    config:
        An :class:`~datapulse.config.EncodingConfig`.
    task_type:
        Classification or regression; controls target-encoding semantics.
    random_state:
        Seed for fold assignment.

    Attributes
    ----------
    one_hot_columns_, target_columns_ : list[str]
        The routing decision made at fit time.
    one_hot_encoder_ : sklearn.preprocessing.OneHotEncoder | None
    target_encoder_ : TargetEncoderOOF | None

    Examples
    --------
    >>> import pandas as pd
    >>> from datapulse.config import EncodingConfig, TaskType
    >>> X = pd.DataFrame({"city": [f"c{i%40}" for i in range(400)],
    ...                   "tier": ["a", "b"] * 200})
    >>> y = pd.Series([0, 1] * 200)
    >>> enc = CategoricalEncoderPipeline(EncodingConfig(), TaskType.CLASSIFICATION)
    >>> out = enc.fit_transform(X, y)
    >>> "city__te" in out.columns and any(c.startswith("tier_") for c in out.columns)
    True

    """

    stage_name = "encoding"

    def __init__(
        self,
        config: Optional[EncodingConfig] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        random_state: int = 42,
    ) -> None:
        self.config = config or EncodingConfig()
        self.task_type = task_type
        self.random_state = random_state

    @property
    def requires_y(self) -> bool:  # type: ignore[override]
        """Target encoding cannot be fitted without labels."""
        return bool(self.config.enabled and self.config.target_encoding_enabled)

    # ------------------------------------------------------------------ #
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        cfg = self.config
        categorical = [
            c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
        ]
        self.numeric_passthrough_ = [c for c in X.columns if c not in categorical]

        cardinality = {c: int(X[c].nunique(dropna=True)) for c in categorical}
        self.cardinality_ = cardinality
        self.one_hot_columns_ = [
            c for c in categorical if cardinality[c] <= cfg.one_hot_max_cardinality
        ]
        self.target_columns_ = [
            c for c in categorical if c not in self.one_hot_columns_
        ]

        if not cfg.target_encoding_enabled and self.target_columns_:
            self._logger.warning(
                "Target encoding disabled; %d high-cardinality column(s) will be "
                "frequency-encoded only.",
                len(self.target_columns_),
            )

        # --- one-hot ------------------------------------------------- #
        self.one_hot_encoder_ = None
        self.one_hot_names_: list[str] = []
        if self.one_hot_columns_:
            self.one_hot_encoder_ = OneHotEncoder(
                handle_unknown=cfg.handle_unknown,
                min_frequency=cfg.min_frequency,
                drop="first" if cfg.drop_first else None,
                sparse_output=False,
                dtype=np.float64,
            )
            block = X[self.one_hot_columns_].astype("object").fillna("__NA__")
            self.one_hot_encoder_.fit(block)
            raw_names = self.one_hot_encoder_.get_feature_names_out(self.one_hot_columns_)
            self.one_hot_names_ = ensure_unique_columns(
                [sanitize_column_name(n) for n in raw_names]
            )

        # --- target encoding ----------------------------------------- #
        self.target_encoder_ = None
        if self.target_columns_ and cfg.target_encoding_enabled and y is not None:
            self.target_encoder_ = TargetEncoderOOF(
                self.target_columns_,
                task_type=self.task_type,
                n_folds=cfg.target_encoding_folds,
                smoothing=cfg.target_encoding_smoothing,
                noise=cfg.target_encoding_noise,
                unseen=cfg.unseen_category_value,
                random_state=self.random_state,
            )
            block = X[self.target_columns_].astype("object").fillna("__NA__")
            self.target_encoder_.fit(block, y)

        # --- frequency encoding -------------------------------------- #
        self.frequency_maps_: dict[str, pd.Series] = {}
        if cfg.add_frequency_encoding:
            for column in self.target_columns_:
                counts = X[column].astype("object").fillna("__NA__").value_counts(
                    normalize=True
                )
                self.frequency_maps_[column] = counts

        self.fit_report_.update(
            {
                "n_one_hot_columns": len(self.one_hot_columns_),
                "n_one_hot_features": len(self.one_hot_names_),
                "n_target_encoded_columns": len(self.target_columns_),
                "max_cardinality": max(cardinality.values()) if cardinality else 0,
            }
        )

    # ------------------------------------------------------------------ #
    def _encode_blocks(
        self, X: pd.DataFrame, *, target_frame: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Assemble numeric passthrough + one-hot + target/frequency blocks."""
        pieces: list[pd.DataFrame] = []

        passthrough = [c for c in self.numeric_passthrough_ if c in X.columns]
        if passthrough:
            pieces.append(X[passthrough].astype("float64"))

        if self.one_hot_encoder_ is not None:
            block = X.reindex(columns=self.one_hot_columns_).astype("object").fillna("__NA__")
            encoded = self.one_hot_encoder_.transform(block)
            pieces.append(
                pd.DataFrame(encoded, columns=self.one_hot_names_, index=X.index)
            )

        if target_frame is not None and not target_frame.empty:
            pieces.append(target_frame)

        if self.frequency_maps_:
            freq: dict[str, np.ndarray] = {}
            for column, mapping in self.frequency_maps_.items():
                if column not in X.columns:
                    continue
                keys = X[column].astype("object").fillna("__NA__")
                freq[f"{column}__freq"] = (
                    keys.map(mapping).astype("float64").fillna(0.0).to_numpy()
                )
            if freq:
                pieces.append(pd.DataFrame(freq, index=X.index))

        if not pieces:
            return pd.DataFrame(index=X.index)
        out = pd.concat(pieces, axis=1)
        out.columns = ensure_unique_columns([str(c) for c in out.columns])
        return out

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        target_frame = None
        if self.target_encoder_ is not None:
            block = X.reindex(columns=self.target_columns_).astype("object").fillna("__NA__")
            target_frame = self.target_encoder_.transform(block)
        return self._encode_blocks(X, target_frame=target_frame)

    def fit_transform(
        self, X: Any, y: Any = None, **fit_params: Any
    ) -> pd.DataFrame:
        """Fit and return **out-of-fold** encodings for the training rows.

        This override is the leakage guard: calling ``fit`` then ``transform``
        on the same frame would apply full-data target statistics to the rows
        that produced them.
        """
        frame = self._as_frame(X)
        target = self._as_series(y, index=frame.index)
        self.fit(frame, target)

        target_frame = None
        if self.target_encoder_ is not None and target is not None:
            block = frame.reindex(columns=self.target_columns_).astype("object").fillna("__NA__")
            target_frame = self.target_encoder_.fit_transform(block, target)
            self._logger.info(
                "Applied %d-fold out-of-fold target encoding to %d column(s).",
                self.config.target_encoding_folds,
                len(self.target_columns_),
            )

        out = self._encode_blocks(frame, target_frame=target_frame)
        self.feature_names_out_ = list(out.columns)
        return out
