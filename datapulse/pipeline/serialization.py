"""Module 10 - Pipeline serialization and export.

A model artifact that ships without its preprocessing is not deployable: the
fences, fold statistics, encoder vocabularies and scaler centres *are* part of
the model. This module packages the entire fitted transformer chain into one
file, together with metadata that lets a serving process answer the questions
that matter during an incident:

* Which library versions produced this? (``scikit-learn`` pickles are not
  guaranteed compatible across versions - the loader warns on mismatch.)
* What columns and dtypes did it expect?
* What configuration produced it?
* What did the training data look like (fingerprint, row count, class prior)?

A JSON sidecar mirrors the metadata so it can be indexed by a model registry
without unpickling anything.
"""

from __future__ import annotations

import getpass
import json
import pickle
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from datapulse.config import PipelineConfig, SerializationConfig
from datapulse.exceptions import SerializationError
from datapulse.logger import get_logger

__all__ = ["ArtifactMetadata", "LoadedArtifact", "PipelineSerializer"]

_logger = get_logger(__name__)

try:  # pragma: no cover - trivial
    from datapulse import __version__ as _DATAPULSE_VERSION
except ImportError:  # pragma: no cover
    _DATAPULSE_VERSION = "0.0.0"


class ArtifactMetadata(BaseModel):
    """Provenance and contract information stored beside the pipeline."""

    model_config = ConfigDict(extra="allow")

    artifact_name: str
    datapulse_version: str = _DATAPULSE_VERSION
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    created_by: str = "unknown"
    hostname: str = "unknown"
    python_version: str = platform.python_version()
    library_versions: dict[str, str] = Field(default_factory=dict)

    task_type: str = "classification"
    target_column: str = ""
    target_classes: Optional[list[str]] = None
    class_prior: Optional[dict[str, float]] = None

    n_training_rows: int = 0
    raw_feature_names: list[str] = Field(default_factory=list)
    raw_dtypes: dict[str, str] = Field(default_factory=dict)
    output_feature_names: list[str] = Field(default_factory=list)
    n_features_out: int = 0
    training_fingerprint: str = ""

    stages: list[str] = Field(default_factory=list)
    config: Optional[dict] = None
    schema_table: Optional[list[dict]] = None
    drift_summary: Optional[dict] = None

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the metadata to JSON."""
        return self.model_dump_json(indent=indent)


class LoadedArtifact:
    """A pipeline restored from disk, plus its metadata.

    Attributes
    ----------
    engine : object
        The fitted engine / pipeline object.
    metadata : ArtifactMetadata
    path : pathlib.Path

    """

    def __init__(self, engine: Any, metadata: ArtifactMetadata, path: Path) -> None:
        self.engine = engine
        self.metadata = metadata
        self.path = path

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convenience passthrough to the wrapped engine's ``transform``."""
        return self.engine.transform(X)

    def check_environment(self) -> list[str]:
        """Return a list of version-mismatch warnings (empty when clean)."""
        problems: list[str] = []
        current = _library_versions()
        for library, version in self.metadata.library_versions.items():
            running = current.get(library)
            if running and running != version:
                problems.append(
                    f"{library}: artifact built with {version}, running {running}"
                )
        return problems

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<LoadedArtifact {self.metadata.artifact_name!r} "
            f"built {self.metadata.created_at} "
            f"({self.metadata.n_features_out} output features)>"
        )


def _library_versions() -> dict[str, str]:
    """Snapshot the versions of libraries whose pickles are version-sensitive."""
    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    try:  # imbalanced-learn is optional at inference time
        import imblearn

        versions["imbalanced-learn"] = imblearn.__version__
    except ImportError:  # pragma: no cover
        pass
    return versions


class PipelineSerializer:
    """Package, write and restore end-to-end pipeline artifacts.

    Parameters
    ----------
    config:
        A :class:`~datapulse.config.SerializationConfig`.

    Examples
    --------
    >>> import tempfile, pandas as pd
    >>> from datapulse.config import SerializationConfig
    >>> from sklearn.preprocessing import StandardScaler
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     ser = PipelineSerializer(SerializationConfig(output_dir=tmp))
    ...     scaler = StandardScaler().fit([[1.0], [2.0], [3.0]])
    ...     path = ser.save(scaler, metadata_extra={"artifact_name": "demo"})
    ...     restored = ser.load(path)
    ...     restored.metadata.artifact_name
    'demo'

    """

    def __init__(self, config: Optional[SerializationConfig] = None) -> None:
        self.config = config or SerializationConfig()

    # ------------------------------------------------------------------ #
    def build_metadata(
        self,
        *,
        pipeline_config: Optional[PipelineConfig] = None,
        raw_frame: Optional[pd.DataFrame] = None,
        target: Optional[pd.Series] = None,
        output_features: Optional[list[str]] = None,
        stages: Optional[list[str]] = None,
        schema_table: Optional[pd.DataFrame] = None,
        drift_summary: Optional[dict] = None,
        fingerprint: str = "",
        extra: Optional[dict] = None,
    ) -> ArtifactMetadata:
        """Assemble an :class:`ArtifactMetadata` from a fitted run."""
        payload: dict[str, Any] = {
            "artifact_name": self.config.artifact_name,
            "library_versions": _library_versions(),
            "stages": stages or [],
            "output_feature_names": output_features or [],
            "n_features_out": len(output_features or []),
            "training_fingerprint": fingerprint,
            "created_by": _safe_user(),
            "hostname": _safe_hostname(),
        }

        if pipeline_config is not None:
            payload["task_type"] = pipeline_config.task_type.value
            payload["target_column"] = pipeline_config.target_column
            if self.config.include_config:
                payload["config"] = json.loads(pipeline_config.to_json())

        if raw_frame is not None:
            payload["n_training_rows"] = int(len(raw_frame))
            payload["raw_feature_names"] = [str(c) for c in raw_frame.columns]
            payload["raw_dtypes"] = {str(c): str(raw_frame[c].dtype) for c in raw_frame.columns}

        if target is not None:
            values = pd.Series(np.asarray(target).ravel())
            if values.nunique() <= 50:
                counts = values.value_counts(normalize=True)
                payload["target_classes"] = [str(c) for c in counts.index]
                payload["class_prior"] = {str(k): round(float(v), 6) for k, v in counts.items()}

        if self.config.include_schema and schema_table is not None:
            payload["schema_table"] = schema_table.to_dict(orient="records")

        if drift_summary is not None:
            payload["drift_summary"] = drift_summary

        payload.update(extra or {})
        return ArtifactMetadata(**payload)

    # ------------------------------------------------------------------ #
    def save(
        self,
        engine: Any,
        *,
        metadata: Optional[ArtifactMetadata] = None,
        metadata_extra: Optional[dict] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Write the pipeline and its metadata to a single artifact file.

        Parameters
        ----------
        engine:
            The fitted engine or Scikit-Learn pipeline to package.
        metadata:
            Pre-built metadata; when omitted a minimal record is generated.
        metadata_extra:
            Fields merged into the generated metadata.
        filename:
            Override the configured artifact name.

        Returns
        -------
        pathlib.Path
            Path to the written artifact.

        Raises
        ------
        SerializationError
            If the artifact cannot be written.

        """
        cfg = self.config
        if metadata is None:
            fields: dict[str, Any] = {
                "artifact_name": cfg.artifact_name,
                "library_versions": _library_versions(),
                "created_by": _safe_user(),
                "hostname": _safe_hostname(),
            }
            fields.update(metadata_extra or {})
            metadata = ArtifactMetadata(**fields)
        elif metadata_extra:
            metadata = metadata.model_copy(update=metadata_extra)

        directory = Path(cfg.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        extension = "joblib" if cfg.format == "joblib" else "pkl"
        path = directory / f"{filename or cfg.artifact_name}.{extension}"

        bundle = {
            "engine": engine,
            "metadata": metadata.model_dump(),
            "format_version": 1,
        }
        try:
            if cfg.format == "joblib":
                joblib.dump(bundle, path, compress=cfg.compress)
            else:
                with path.open("wb") as handle:
                    pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except (OSError, pickle.PicklingError, TypeError) as exc:
            raise SerializationError(f"Failed to write artifact {path}: {exc}") from exc

        if cfg.write_metadata_sidecar:
            sidecar = path.with_suffix(".meta.json")
            sidecar.write_text(metadata.to_json(), encoding="utf-8")
            _logger.info("Metadata sidecar written: %s", sidecar)

        size_mb = path.stat().st_size / 1_048_576
        _logger.info("Artifact written: %s (%.2f MB)", path, size_mb)
        return path

    # ------------------------------------------------------------------ #
    @staticmethod
    def load(path: Union[str, Path], *, strict: bool = False) -> LoadedArtifact:
        """Restore an artifact written by :meth:`save`.

        Parameters
        ----------
        path:
            Artifact location.
        strict:
            When ``True``, raise on library-version mismatch instead of warning.

        Returns
        -------
        LoadedArtifact

        Raises
        ------
        SerializationError
            If the file is missing, corrupt, or (with ``strict``) built against
            different library versions.

        """
        location = Path(path)
        if not location.exists():
            raise SerializationError(f"Artifact not found: {location}")

        try:
            if location.suffix == ".joblib":
                bundle = joblib.load(location)
            else:
                with location.open("rb") as handle:
                    bundle = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError) as exc:
            raise SerializationError(f"Failed to load artifact {location}: {exc}") from exc

        if not isinstance(bundle, dict) or "engine" not in bundle:
            raise SerializationError(
                f"{location} is not a DataPulse artifact (missing 'engine' key)."
            )

        metadata = ArtifactMetadata(**bundle.get("metadata", {"artifact_name": location.stem}))
        artifact = LoadedArtifact(bundle["engine"], metadata, location)

        problems = artifact.check_environment()
        if problems:
            message = "Library version mismatch: " + "; ".join(problems)
            if strict:
                raise SerializationError(message)
            _logger.warning("%s. Unpickled estimators may behave differently.", message)

        _logger.info("Artifact loaded: %s", artifact)
        return artifact


def _safe_user() -> str:
    """Best-effort username; containers often have no passwd entry."""
    try:
        return getpass.getuser()
    except (KeyError, OSError):  # pragma: no cover
        return "unknown"


def _safe_hostname() -> str:
    """Best-effort hostname."""
    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover
        return "unknown"
