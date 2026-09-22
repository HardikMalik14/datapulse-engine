"""Low-level, side-effect-free helpers shared by the transformers."""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd

from datapulse.exceptions import SchemaError

__all__ = [
    "coerce_numeric",
    "ensure_unique_columns",
    "frame_fingerprint",
    "is_datetime_like",
    "parse_datetime_series",
    "safe_sample",
    "sanitize_column_name",
    "split_features_target",
]

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z_]+")


def sanitize_column_name(name: str) -> str:
    """Return a model-safe column name.

    LightGBM, XGBoost and several serving layers reject names containing
    ``[]<>`` or spaces, so generated feature names are normalised here.

    Examples
    --------
    >>> sanitize_column_name("Annual Income ($)")
    'Annual_Income'
    >>> sanitize_column_name("plan tier=pro")
    'plan_tier_pro'

    """
    cleaned = _NON_ALNUM.sub("_", str(name)).strip("_")
    return cleaned or "feature"


def ensure_unique_columns(columns: Sequence[str]) -> list[str]:
    """De-duplicate column names by appending ``__1``, ``__2``, ...

    Examples
    --------
    >>> ensure_unique_columns(["a", "a", "b"])
    ['a', 'a__1', 'b']

    """
    seen: dict[str, int] = {}
    output: list[str] = []
    for column in columns:
        if column not in seen:
            seen[column] = 0
            output.append(column)
        else:
            seen[column] += 1
            output.append(f"{column}__{seen[column]}")
    return output


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion that never raises.

    Strings such as ``"1,234"``, ``"$45.2"`` and ``" 7 "`` are recovered;
    anything else becomes ``NaN``.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = (
        series.astype("string")
        .str.replace(r"[,$%\s]", "", regex=True)
        .replace({"": None, "nan": None, "None": None, "null": None, "NA": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse a series to datetime, returning all-``NaT`` on failure."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            try:
                return pd.to_datetime(series, errors="coerce")
            except (ValueError, TypeError):
                return pd.Series(pd.NaT, index=series.index)


def is_datetime_like(series: pd.Series, threshold: float = 0.8) -> bool:
    """Heuristically decide whether an object column holds dates.

    Parameters
    ----------
    series:
        Candidate column.
    threshold:
        Minimum fraction of *non-null* values that must parse successfully.

    Notes
    -----
    Pure-integer columns are deliberately rejected: a column of years or IDs
    parses happily as nanoseconds-since-epoch and would otherwise be silently
    destroyed.

    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.sample(
        min(len(non_null), 500), random_state=0
    ) if len(non_null) > 500 else non_null
    parsed = parse_datetime_series(sample.astype("string"))
    return float(parsed.notna().mean()) >= threshold


def safe_sample(
    values: np.ndarray, max_size: int, random_state: int = 0
) -> np.ndarray:
    """Return at most ``max_size`` elements, sampled without replacement."""
    array = np.asarray(values)
    if array.size <= max_size:
        return array
    rng = np.random.default_rng(random_state)
    return rng.choice(array, size=max_size, replace=False)


def split_features_target(
    frame: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into ``(X, y)``.

    Raises
    ------
    SchemaError
        If ``target_column`` is absent.

    """
    if target_column not in frame.columns:
        raise SchemaError(
            f"Target column {target_column!r} not found. Available: {list(frame.columns)[:20]}"
        )
    features = frame.drop(columns=[target_column])
    target = frame[target_column]
    return features, target


def frame_fingerprint(frame: pd.DataFrame, n_rows: int = 1000) -> str:
    """Return a short, stable hash of a frame's schema and head.

    Used to stamp serialized artifacts so a served model can detect that it is
    being fed a structurally different table from the one it was trained on.
    """
    hasher = hashlib.sha256()
    hasher.update("|".join(f"{c}:{frame[c].dtype}" for c in frame.columns).encode())
    head = frame.head(n_rows)
    try:
        hasher.update(pd.util.hash_pandas_object(head, index=False).values.tobytes())
    except TypeError:  # pragma: no cover - exotic dtypes
        hasher.update(head.astype("string").to_csv(index=False).encode())
    return hasher.hexdigest()[:16]


def summarise_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missingness summary sorted by severity."""
    na_counts = frame.isna().sum()
    summary = pd.DataFrame(
        {
            "column": na_counts.index,
            "n_missing": na_counts.to_numpy(),
            "pct_missing": (na_counts / max(len(frame), 1)).to_numpy(),
            "dtype": [str(frame[c].dtype) for c in na_counts.index],
        }
    )
    return summary.sort_values("pct_missing", ascending=False).reset_index(drop=True)


def infer_positive_only(series: pd.Series) -> bool:
    """Whether every non-null value is strictly positive (Box-Cox eligibility)."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return False
    return bool((values > 0).all())


def resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """Normalise a Scikit-Learn style ``n_jobs`` value."""
    if n_jobs is None:
        return 1
    return int(n_jobs)
