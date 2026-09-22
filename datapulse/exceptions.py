"""Exception hierarchy for the DataPulse Engine.

Every failure raised by the framework derives from :class:`DataPulseError`, so
callers can trap framework problems without swallowing unrelated exceptions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__ = [
    "DataPulseError",
    "ConfigurationError",
    "SchemaError",
    "NotFittedError",
    "TransformerError",
    "LeakageGuardError",
    "SerializationError",
    "ReportingError",
]


class DataPulseError(Exception):
    """Base class for all DataPulse Engine errors."""


class ConfigurationError(DataPulseError):
    """Raised when a :class:`~datapulse.config.PipelineConfig` is invalid."""


class SchemaError(DataPulseError):
    """Raised when the incoming frame does not match the fitted schema."""

    @classmethod
    def from_columns(
        cls,
        missing: Iterable[str],
        unexpected: Iterable[str],
    ) -> SchemaError:
        """Build a descriptive error from column set differences."""
        missing_cols: Sequence[str] = sorted(missing)
        unexpected_cols: Sequence[str] = sorted(unexpected)
        parts: list[str] = []
        if missing_cols:
            parts.append(f"missing columns: {missing_cols}")
        if unexpected_cols:
            parts.append(f"unexpected columns: {unexpected_cols}")
        detail = "; ".join(parts) or "column layout differs from fit time"
        return cls(f"Input schema mismatch -> {detail}")


class NotFittedError(DataPulseError):
    """Raised when ``transform`` is called before ``fit``."""


class TransformerError(DataPulseError):
    """Raised when a transformer fails during fit or transform."""


class LeakageGuardError(DataPulseError):
    """Raised when an operation would leak target information into inference.

    The canonical example is attempting to resample (SMOTE/ADASYN) a holdout
    set: resampling is a *training-only* operation and the engine refuses to
    perform it outside a ``fit``/``fit_resample`` context.
    """


class SerializationError(DataPulseError):
    """Raised when an artifact cannot be written to or read from disk."""


class ReportingError(DataPulseError):
    """Raised when a visual or tabular report cannot be produced."""
