"""Module 1 - Automatic schema and type inference.

Real tabular data arrives with the wrong dtypes: numbers stored as strings,
timestamps stored as objects, ZIP codes stored as integers, and a primary key
that looks like a perfectly good feature until it gives you 0.99 AUC in
training and 0.51 in production.

:class:`SchemaInferencer` classifies every column into one of

``numerical`` | ``categorical_low`` | ``categorical_high`` | ``datetime`` | ``boolean`` | ``dropped``

using cardinality, parseability and null-rate heuristics, then *coerces* the
frame to those types so that all downstream modules can trust their inputs.
:class:`DateTimeFeaturizer` then explodes datetime columns into modelling
features (calendar parts, cyclical sine/cosine encodings, and recency).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from datapulse.base import AbstractTransformer
from datapulse.config import SchemaConfig
from datapulse.utils.validation import (
    coerce_numeric,
    is_datetime_like,
    parse_datetime_series,
)

__all__ = ["FeatureKind", "ColumnProfile", "DataSchema", "SchemaInferencer", "DateTimeFeaturizer"]


class FeatureKind(str, Enum):
    """Semantic role assigned to a column by the inferencer."""

    NUMERICAL = "numerical"
    CATEGORICAL_LOW = "categorical_low"
    CATEGORICAL_HIGH = "categorical_high"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    DROPPED = "dropped"


class ColumnProfile(BaseModel):
    """Per-column profile produced during inference."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: FeatureKind
    dtype: str
    n_unique: int
    unique_ratio: float
    null_rate: float
    reason: str = ""
    is_positive_only: bool = False
    skewness: Optional[float] = None


class DataSchema(BaseModel):
    """Immutable description of a dataset, learnt at ``fit`` time.

    This object is serialized alongside the pipeline artifact so that a serving
    process can answer "what did this model expect?" without loading the
    training data.
    """

    model_config = ConfigDict(extra="forbid")

    n_rows: int = 0
    n_columns: int = 0
    profiles: dict[str, ColumnProfile] = Field(default_factory=dict)

    # ---- convenience accessors -------------------------------------- #
    def _of_kind(self, *kinds: FeatureKind) -> list[str]:
        return [n for n, p in self.profiles.items() if p.kind in kinds]

    @property
    def numerical(self) -> list[str]:
        """Continuous / discrete numeric columns."""
        return self._of_kind(FeatureKind.NUMERICAL)

    @property
    def categorical_low(self) -> list[str]:
        """Low-cardinality categoricals (one-hot candidates)."""
        return self._of_kind(FeatureKind.CATEGORICAL_LOW, FeatureKind.BOOLEAN)

    @property
    def categorical_high(self) -> list[str]:
        """High-cardinality categoricals (target-encoding candidates)."""
        return self._of_kind(FeatureKind.CATEGORICAL_HIGH)

    @property
    def categorical(self) -> list[str]:
        """All categorical columns."""
        return self.categorical_low + self.categorical_high

    @property
    def datetime(self) -> list[str]:
        """Datetime columns."""
        return self._of_kind(FeatureKind.DATETIME)

    @property
    def dropped(self) -> list[str]:
        """Columns excluded from modelling (constants, IDs, mostly-null)."""
        return self._of_kind(FeatureKind.DROPPED)

    @property
    def retained(self) -> list[str]:
        """All columns that survive inference, in original order."""
        return [n for n, p in self.profiles.items() if p.kind is not FeatureKind.DROPPED]

    def to_frame(self) -> pd.DataFrame:
        """Render the schema as a tidy DataFrame for reports and notebooks."""
        if not self.profiles:
            return pd.DataFrame(
                columns=["name", "kind", "dtype", "n_unique", "unique_ratio", "null_rate", "reason"]
            )
        records = [p.model_dump() for p in self.profiles.values()]
        frame = pd.DataFrame(records)
        frame["kind"] = frame["kind"].map(
            lambda k: k.value if isinstance(k, FeatureKind) else str(k)
        )
        return frame

    def summary(self) -> str:
        """Single-line counts by kind, for logging."""
        return (
            f"numerical={len(self.numerical)}, "
            f"cat_low={len(self.categorical_low)}, "
            f"cat_high={len(self.categorical_high)}, "
            f"datetime={len(self.datetime)}, "
            f"dropped={len(self.dropped)}"
        )


class SchemaInferencer(AbstractTransformer):
    """Classify and coerce columns into modelling-ready dtypes.

    Parameters
    ----------
    config:
        A :class:`~datapulse.config.SchemaConfig`.

    Attributes
    ----------
    schema_ : DataSchema
        The inferred schema, available after ``fit``.

    Examples
    --------
    >>> import pandas as pd
    >>> from datapulse.config import SchemaConfig
    >>> df = pd.DataFrame({"a": [f"{i / 3:.2f}" for i in range(60)],
    ...                    "b": ["x", "y", "z"] * 20,
    ...                    "id": range(1000, 1060)})
    >>> inf = SchemaInferencer(SchemaConfig()).fit(df)
    >>> inf.schema_.numerical, inf.schema_.categorical_low, inf.schema_.dropped
    (['a'], ['b'], ['id'])

    """

    stage_name = "schema_inference"

    def __init__(self, config: Optional[SchemaConfig] = None) -> None:
        self.config = config or SchemaConfig()

    # ------------------------------------------------------------------ #
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        cfg = self.config
        n_rows = max(len(X), 1)
        profiles: dict[str, ColumnProfile] = {}

        for column in X.columns:
            series = X[column]
            null_rate = float(series.isna().mean())
            n_unique = int(series.nunique(dropna=True))
            unique_ratio = n_unique / n_rows

            kind, reason, coerced = self._classify(
                column=column,
                series=series,
                n_unique=n_unique,
                unique_ratio=unique_ratio,
                null_rate=null_rate,
            )

            skewness: Optional[float] = None
            positive_only = False
            if kind is FeatureKind.NUMERICAL and coerced is not None:
                numeric = coerced.dropna()
                if len(numeric) > 2:
                    skew_value = float(numeric.skew())
                    skewness = None if np.isnan(skew_value) else skew_value
                    positive_only = bool((numeric > 0).all())

            profiles[str(column)] = ColumnProfile(
                name=str(column),
                kind=kind,
                dtype=str(series.dtype),
                n_unique=n_unique,
                unique_ratio=round(unique_ratio, 6),
                null_rate=round(null_rate, 6),
                reason=reason,
                is_positive_only=positive_only,
                skewness=skewness,
            )

        self.schema_ = DataSchema(
            n_rows=int(len(X)), n_columns=int(X.shape[1]), profiles=profiles
        )
        self.fit_report_["schema"] = self.schema_.summary()
        self.fit_report_["dropped_columns"] = self.schema_.dropped
        self._logger.info("Inferred schema -> %s", self.schema_.summary())

    # ------------------------------------------------------------------ #
    def _classify(
        self,
        *,
        column: str,
        series: pd.Series,
        n_unique: int,
        unique_ratio: float,
        null_rate: float,
    ) -> tuple[FeatureKind, str, Optional[pd.Series]]:
        """Return ``(kind, reason, coerced_series)`` for a single column."""
        cfg = self.config

        # 1. Explicit user overrides always win.
        if column in cfg.force_drop:
            return FeatureKind.DROPPED, "forced by config", None
        if column in cfg.force_numerical:
            return FeatureKind.NUMERICAL, "forced by config", coerce_numeric(series)
        if column in cfg.force_datetime:
            return FeatureKind.DATETIME, "forced by config", parse_datetime_series(series)
        if column in cfg.force_categorical:
            kind = (
                FeatureKind.CATEGORICAL_HIGH
                if n_unique > cfg.high_cardinality_threshold
                else FeatureKind.CATEGORICAL_LOW
            )
            return kind, "forced by config", None

        # 2. Degenerate columns.
        if n_unique <= 1 and cfg.drop_constant_columns:
            return FeatureKind.DROPPED, "constant or all-null column", None

        # 3. Datetime detection (before numeric, after constants).
        if pd.api.types.is_datetime64_any_dtype(series) or is_datetime_like(
            series, cfg.datetime_parse_threshold
        ):
            return FeatureKind.DATETIME, "parses as datetime", parse_datetime_series(series)

        # 4. Booleans.
        if pd.api.types.is_bool_dtype(series):
            return FeatureKind.BOOLEAN, "boolean dtype", None

        # 5. Numeric (native or recoverable from strings).
        coerced = coerce_numeric(series)
        non_null = series.notna().sum()
        recovered = float(coerced.notna().sum() / non_null) if non_null else 0.0
        looks_numeric = pd.api.types.is_numeric_dtype(series) or recovered >= 0.9

        if looks_numeric:
            # Identifier-like: a near-unique integer column is a key, not a feature.
            is_integral = bool(
                pd.api.types.is_integer_dtype(series)
                or (coerced.dropna() % 1 == 0).all()
            )
            if (
                cfg.drop_identifier_columns
                and is_integral
                and unique_ratio > cfg.cardinality_ratio_threshold
            ):
                return FeatureKind.DROPPED, "identifier-like (near-unique integers)", None
            # Low-cardinality integers are really categories (ratings, flags).
            if is_integral and n_unique <= cfg.numeric_as_categorical_max_unique:
                return (
                    FeatureKind.CATEGORICAL_LOW,
                    f"integer with only {n_unique} levels",
                    None,
                )
            return FeatureKind.NUMERICAL, "numeric dtype or recoverable", coerced

        # 6. Everything else is categorical; split by cardinality.
        if cfg.drop_identifier_columns and unique_ratio > cfg.cardinality_ratio_threshold:
            return FeatureKind.DROPPED, "free-text / identifier-like string", None
        if n_unique > cfg.high_cardinality_threshold:
            return (
                FeatureKind.CATEGORICAL_HIGH,
                f"cardinality {n_unique} > {cfg.high_cardinality_threshold}",
                None,
            )
        return FeatureKind.CATEGORICAL_LOW, f"cardinality {n_unique}", None

    # ------------------------------------------------------------------ #
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop excluded columns and cast survivors to their inferred dtype."""
        out = X.copy()
        schema = self.schema_

        dropped = [c for c in schema.dropped if c in out.columns]
        if dropped:
            out = out.drop(columns=dropped)

        for column in out.columns:
            profile = schema.profiles.get(column)
            if profile is None:
                continue
            if profile.kind is FeatureKind.NUMERICAL:
                out[column] = coerce_numeric(out[column]).astype("float64")
            elif profile.kind is FeatureKind.DATETIME:
                out[column] = parse_datetime_series(out[column])
            elif profile.kind is FeatureKind.BOOLEAN:
                out[column] = out[column].astype("object")
            else:  # categorical
                out[column] = (
                    out[column].astype("object").where(out[column].notna(), other=np.nan)
                )
        return out


class DateTimeFeaturizer(AbstractTransformer):
    """Explode datetime columns into calendar, cyclical and recency features.

    For each datetime column ``t`` the following are emitted (subject to
    config): ``t__year``, ``t__month``, ``t__day``, ``t__dayofweek``,
    ``t__hour``, ``t__quarter``, ``t__is_weekend``, ``t__days_since_ref`` and,
    when :attr:`SchemaConfig.cyclical_encoding` is on, ``t__month_sin`` /
    ``t__month_cos`` / ``t__dow_sin`` / ``t__dow_cos``.

    Cyclical encodings matter: raw ``month`` tells a linear model that December
    (12) is maximally far from January (1), which is exactly backwards.
    """

    stage_name = "datetime_features"

    def __init__(self, config: Optional[SchemaConfig] = None) -> None:
        self.config = config or SchemaConfig()

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        self.datetime_columns_ = [
            c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])
        ]
        reference: dict[str, pd.Timestamp] = {}
        for column in self.datetime_columns_:
            if self.config.datetime_reference:
                reference[column] = pd.Timestamp(self.config.datetime_reference)
            else:
                observed_max = X[column].max()
                reference[column] = (
                    pd.Timestamp(observed_max)
                    if pd.notna(observed_max)
                    else pd.Timestamp("1970-01-01")
                )
        self.reference_dates_ = reference

        # Decide the emitted column list ONCE, at fit time. Deciding it at
        # transform time would let a constant-valued batch (e.g. a single-day
        # scoring request) silently change the feature matrix width.
        engineered = self._build_features(X)
        self.generated_columns_ = [
            c for c in engineered.columns if engineered[c].nunique(dropna=True) > 1
        ]
        self.fit_report_["datetime_columns"] = self.datetime_columns_
        self.fit_report_["generated_datetime_features"] = len(self.generated_columns_)

    def _build_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute every candidate datetime feature for ``X``."""
        generated: dict[str, pd.Series] = {}
        for column in self.datetime_columns_:
            if column not in X.columns:
                continue
            values = pd.to_datetime(X[column], errors="coerce")
            prefix = column
            generated[f"{prefix}__year"] = values.dt.year.astype("float64")
            generated[f"{prefix}__month"] = values.dt.month.astype("float64")
            generated[f"{prefix}__day"] = values.dt.day.astype("float64")
            generated[f"{prefix}__dayofweek"] = values.dt.dayofweek.astype("float64")
            generated[f"{prefix}__quarter"] = values.dt.quarter.astype("float64")
            generated[f"{prefix}__hour"] = values.dt.hour.astype("float64")
            generated[f"{prefix}__is_weekend"] = (
                (values.dt.dayofweek >= 5).astype("float64").where(values.notna())
            )
            reference = self.reference_dates_.get(column, pd.Timestamp("1970-01-01"))
            generated[f"{prefix}__days_since_ref"] = (
                (reference - values).dt.total_seconds() / 86_400.0
            ).astype("float64")

            if self.config.cyclical_encoding:
                month = values.dt.month.astype("float64")
                dow = values.dt.dayofweek.astype("float64")
                generated[f"{prefix}__month_sin"] = np.sin(2 * np.pi * month / 12.0)
                generated[f"{prefix}__month_cos"] = np.cos(2 * np.pi * month / 12.0)
                generated[f"{prefix}__dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
                generated[f"{prefix}__dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        if not generated:
            return pd.DataFrame(index=X.index)
        return pd.DataFrame(generated, index=X.index)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns=[c for c in self.datetime_columns_ if c in X.columns])
        if not self.config.expand_datetime_features or not self.generated_columns_:
            return out
        engineered = self._build_features(X)
        engineered = engineered.reindex(columns=self.generated_columns_)
        return pd.concat([out, engineered], axis=1)
