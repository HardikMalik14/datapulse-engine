"""The orchestrator: :class:`DataPulseEngine`.

This module wires modules 1-10 into one end-to-end workflow and - critically -
decides *where* each one is allowed to run.

Stage order and the reasoning behind it
---------------------------------------
======  ===========================  =========================================
Order   Stage                        Why here
======  ===========================  =========================================
1       Schema inference             Everything downstream needs real dtypes.
2       Datetime featurisation       Turns timestamps into modellable numbers.
3       Imputation                   Outlier detectors and scalers cannot see
                                     ``NaN``; imputing first keeps them honest.
4       Outlier treatment            Fences must be learnt on complete data, and
                                     before scalers so extreme values do not
                                     define the scale.
5       NaN guard                    Catches ``NaN`` reintroduced by ``mask``.
6       Skew correction + scaling    On raw numerics only - never on dummies.
7       Categorical encoding         Out-of-fold on train, full-map on holdout.
8       Feature selection            Needs the final numeric matrix.
9       Resampling                   **Training only**, after everything else,
                                     so synthetic rows live in the same space
                                     the model will see.
======  ===========================  =========================================

Steps 1-8 form a genuine Scikit-Learn :class:`~sklearn.pipeline.Pipeline`
available as :attr:`DataPulseEngine.pipeline_`; that object is what gets
serialized, so inference needs nothing from this class.

Step 9 is deliberately *outside* the pipeline. A resampler inside a pipeline is
the single most common leakage bug in applied ML - :class:`SmartResampler` is
built so that even if it is placed in one, holdout data passes through
untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from datapulse.base import AbstractTransformer
from datapulse.config import OutlierAction, PipelineConfig, TaskType
from datapulse.core.drift import DriftAuditor, DriftReport
from datapulse.core.encoding import CategoricalEncoderPipeline
from datapulse.core.feature_selection import HybridFeatureSelector
from datapulse.core.imputation import ContextAwareImputer
from datapulse.core.outliers import OutlierEngine
from datapulse.core.resampling import SmartResampler
from datapulse.core.scaling import SkewAwareScaler
from datapulse.core.schema_inference import DateTimeFeaturizer, SchemaInferencer
from datapulse.exceptions import NotFittedError
from datapulse.logger import get_logger
from datapulse.pipeline.serialization import LoadedArtifact, PipelineSerializer
from datapulse.reporting.visuals import VisualReporter
from datapulse.utils.validation import frame_fingerprint, split_features_target

__all__ = ["DataPulseEngine", "ResidualNaNImputer"]

_logger = get_logger(__name__)


class ResidualNaNImputer(AbstractTransformer):
    """Final numeric safety net before encoding.

    Earlier stages can legitimately reintroduce ``NaN`` - ``OutlierAction.MASK``
    does so by design - and a single ``NaN`` reaching an estimator is a hard
    failure at serving time. This transformer fills any residual numeric gap
    with the training median and reports how much it had to do, so the gap is
    visible rather than silently patched.
    """

    stage_name = "nan_guard"

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        numeric = X.select_dtypes(include=[np.number])
        self.fill_values_ = {
            column: float(numeric[column].median())
            if numeric[column].notna().any()
            else 0.0
            for column in numeric.columns
        }
        self.fit_report_["residual_nan_cells"] = int(numeric.isna().sum().sum())

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for column, value in self.fill_values_.items():
            if column in out.columns:
                filled = pd.to_numeric(out[column], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                out[column] = filled.fillna(value)
        return out


class DataPulseEngine:
    """End-to-end preprocessing, feature engineering and pipeline framework.

    Parameters
    ----------
    config:
        A :class:`~datapulse.config.PipelineConfig` driving every module.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The fitted, inference-safe transformer chain (stages 1-8).
    resampler_ : SmartResampler
        Training-only stage 9.
    feature_names_out_ : list[str]
    drift_report_ : DriftReport | None
    reporter_ : VisualReporter

    Examples
    --------
    >>> from datapulse.config import PipelineConfig
    >>> from datapulse.data.synthetic import generate_messy_dataset
    >>> train, test = generate_messy_dataset(n_rows=600, random_state=0, split=True)
    >>> engine = DataPulseEngine(PipelineConfig(target_column="churned"))
    >>> X_train, y_train = engine.fit_transform(train)
    >>> X_test, y_test = engine.transform(test, with_target=True)
    >>> X_train.isna().sum().sum() == 0 and list(X_train.columns) == list(X_test.columns)
    True

    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.is_fitted_ = False
        self.reporter_ = VisualReporter(config.reporting)
        self.serializer_ = PipelineSerializer(config.serialization)
        self.drift_auditor_ = DriftAuditor(config.drift)
        self.drift_report_: Optional[DriftReport] = None
        self.pipeline_: Optional[Pipeline] = None
        self.resampler_: Optional[SmartResampler] = None
        self.feature_names_out_: list[str] = []
        self._raw_reference_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_stages(self) -> list[tuple[str, AbstractTransformer]]:
        """Instantiate the (unfitted) inference-safe stages in order."""
        cfg = self.config
        return [
            ("schema", SchemaInferencer(cfg.schema_inference)),
            ("datetime", DateTimeFeaturizer(cfg.schema_inference)),
            ("imputer", ContextAwareImputer(cfg.imputation, random_state=cfg.random_state)),
            (
                "outliers",
                OutlierEngine(cfg.outliers, random_state=cfg.random_state, n_jobs=cfg.n_jobs),
            ),
            ("nan_guard", ResidualNaNImputer()),
            ("scaler", SkewAwareScaler(cfg.scaling, random_state=cfg.random_state)),
            (
                "encoder",
                CategoricalEncoderPipeline(
                    cfg.encoding, task_type=cfg.task_type, random_state=cfg.random_state
                ),
            ),
            (
                "selector",
                HybridFeatureSelector(
                    cfg.feature_selection,
                    task_type=cfg.task_type,
                    random_state=cfg.random_state,
                    n_jobs=cfg.n_jobs,
                ),
            ),
        ]

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit_transform(
        self, train: pd.DataFrame, *, resample: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit every stage on ``train`` and return the model-ready training set.

        Parameters
        ----------
        train:
            Raw training frame **including** the target column.
        resample:
            Whether to run stage 9. Set ``False`` to inspect the un-balanced
            matrix (e.g. when the downstream estimator handles imbalance with
            class weights instead).

        Returns
        -------
        tuple[pandas.DataFrame, pandas.Series]
            ``(X_train, y_train)`` - fully numeric, NaN-free, and resampled
            when stage 9 ran.

        """
        cfg = self.config
        features, target = split_features_target(train, cfg.target_column)
        self._raw_reference_ = features.copy()
        self._training_fingerprint_ = frame_fingerprint(features)
        _logger.info(
            "Fitting DataPulse engine | %s | %d rows x %d raw columns",
            cfg.summary(),
            len(features),
            features.shape[1],
        )

        stages = self._build_stages()
        fitted: list[tuple[str, AbstractTransformer]] = []
        current, current_target = features, target

        for name, transformer in stages:
            if name == "outliers" and cfg.outliers.action is OutlierAction.DROP:
                # Row-count-changing stage: use the sampler-style API so X and y
                # stay aligned. This is legal only because we are on train data.
                current, current_target = transformer.fit_resample(current, current_target)
            else:
                current = transformer.fit_transform(current, current_target)
            fitted.append((name, transformer))
            _logger.debug(
                "stage %-9s -> %d rows x %d cols", name, current.shape[0], current.shape[1]
            )

        self.pipeline_ = Pipeline(steps=fitted)
        self.stage_names_ = [name for name, _ in fitted]
        self.feature_names_out_ = list(current.columns)

        self.resampler_ = SmartResampler(
            cfg.resampling,
            task_type=cfg.task_type,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        if resample and cfg.task_type is TaskType.CLASSIFICATION:
            current, current_target = self.resampler_.fit_resample(current, current_target)
        else:
            self.resampler_.fit(current, current_target)

        self.is_fitted_ = True
        _logger.info(
            "Engine fitted: %d raw columns -> %d engineered features; training matrix %s",
            features.shape[1],
            len(self.feature_names_out_),
            tuple(current.shape),
        )
        return current, current_target

    def fit(self, train: pd.DataFrame) -> DataPulseEngine:
        """Fit the engine and discard the transformed training matrix."""
        self.fit_transform(train)
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def transform(
        self, frame: pd.DataFrame, *, with_target: bool = False
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, Optional[pd.Series]]]:
        """Apply the fitted chain to new data. Never resamples, never refits.

        Parameters
        ----------
        frame:
            New data, with or without the target column.
        with_target:
            When ``True``, return ``(X, y)``; ``y`` is ``None`` if the target
            column is absent.

        Returns
        -------
        pandas.DataFrame or tuple

        """
        self._check_fitted()
        target: Optional[pd.Series] = None
        features = frame
        if self.config.target_column in frame.columns:
            features, target = split_features_target(frame, self.config.target_column)

        transformed = self.pipeline_.transform(features)
        transformed = transformed.reindex(columns=self.feature_names_out_).fillna(0.0)
        return (transformed, target) if with_target else transformed

    # ------------------------------------------------------------------ #
    # Auditing and reporting
    # ------------------------------------------------------------------ #
    def audit_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        processed: bool = True,
    ) -> DriftReport:
        """Compare two datasets and store the resulting report.

        Parameters
        ----------
        reference, current:
            Raw frames (target column optional).
        processed:
            Audit the engineered feature space (``True``, the default) rather
            than the raw columns. Engineered space is usually what you want:
            it is what the model actually consumes.

        Returns
        -------
        DriftReport

        """
        if processed:
            self._check_fitted()
            left = self.transform(reference)
            right = self.transform(current)
        else:
            left = reference.drop(columns=[self.config.target_column], errors="ignore")
            right = current.drop(columns=[self.config.target_column], errors="ignore")

        self.drift_report_ = self.drift_auditor_.audit(left, right)
        return self.drift_report_

    def generate_reports(
        self,
        *,
        raw_frame: Optional[pd.DataFrame] = None,
        processed_frame: Optional[pd.DataFrame] = None,
    ) -> dict[str, Path]:
        """Render the full visual suite for the current fitted state.

        Parameters
        ----------
        raw_frame:
            Frame used for the nullity matrix; defaults to the training data.
        processed_frame:
            Frame used for the correlation heatmap; defaults to the transformed
            training data.

        Returns
        -------
        dict[str, pathlib.Path]
            Report name -> file path.

        """
        self._check_fitted()
        cfg = self.config
        if not cfg.reporting.enabled:
            return {}

        raw = raw_frame if raw_frame is not None else self._raw_reference_
        if raw is not None:
            self.reporter_.nullity_matrix(raw)

        processed = processed_frame
        if processed is None and raw is not None:
            processed = self.transform(raw)
        if processed is not None:
            self.reporter_.correlation_heatmap(processed)

        scaler: SkewAwareScaler = self.pipeline_.named_steps["scaler"]
        if getattr(scaler, "decisions_", None) is not None:
            self.reporter_.skew_report(
                scaler.decisions_, threshold=cfg.scaling.skew_threshold
            )

        selector: HybridFeatureSelector = self.pipeline_.named_steps["selector"]
        if getattr(selector, "selection_report_", None) is not None:
            self.reporter_.feature_relevance(selector.selection_report_)

        if self.resampler_ is not None and self.resampler_.distribution_before_:
            self.reporter_.class_balance(
                self.resampler_.distribution_before_,
                self.resampler_.distribution_after_,
                strategy=cfg.resampling.strategy.value,
                applied=self.resampler_.was_applied_,
            )

        if self.drift_report_ is not None:
            self.reporter_.drift_report(
                self.drift_report_.table,
                warn_threshold=cfg.drift.psi_warn_threshold,
                alert_threshold=cfg.drift.psi_alert_threshold,
            )

        return dict(self.reporter_.generated_)

    # ------------------------------------------------------------------ #
    def summary(self, *, detail_chars: int = 96) -> pd.DataFrame:
        """Return a tidy per-stage diagnostics table.

        Each stage contributes different diagnostics, so the stage-specific
        keys are folded into a single ``details`` column rather than producing
        a wide, mostly-empty frame.

        Parameters
        ----------
        detail_chars:
            Truncation width for the ``details`` column.

        """
        self._check_fitted()
        common = {"stage", "transformer", "n_rows_in", "n_features_in", "fit_seconds"}
        stages: list[tuple[str, Any]] = list(self.pipeline_.steps)
        if self.resampler_ is not None:
            stages.append(("resampler", self.resampler_))

        records: list[dict[str, Any]] = []
        for name, transformer in stages:
            report = dict(getattr(transformer, "fit_report_", {}))
            details = ", ".join(
                f"{key}={value}" for key, value in report.items() if key not in common
            )
            records.append(
                {
                    "stage": name,
                    "transformer": report.get("transformer", type(transformer).__name__),
                    "rows_in": report.get("n_rows_in"),
                    "features_in": report.get("n_features_in"),
                    "fit_seconds": report.get("fit_seconds"),
                    "details": (
                        details
                        if len(details) <= detail_chars
                        else details[: detail_chars - 3] + "..."
                    ),
                }
            )
        return pd.DataFrame(records)

    def stage_reports(self) -> dict[str, dict]:
        """Return the raw, untruncated diagnostics for every stage."""
        self._check_fitted()
        reports = {
            name: dict(getattr(transformer, "fit_report_", {}))
            for name, transformer in self.pipeline_.steps
        }
        if self.resampler_ is not None:
            reports["resampler"] = dict(self.resampler_.fit_report_)
        return reports

    @property
    def schema(self):  # noqa: ANN201 - DataSchema
        """The inferred :class:`~datapulse.core.schema_inference.DataSchema`."""
        self._check_fitted()
        return self.pipeline_.named_steps["schema"].schema_

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def save(self, *, filename: Optional[str] = None) -> Path:
        """Package the fitted inference chain into a deployable artifact.

        Only stages 1-8 are serialized: the resampler is training-only and has
        no business travelling to a serving environment.

        Returns
        -------
        pathlib.Path

        """
        self._check_fitted()
        metadata = self.serializer_.build_metadata(
            pipeline_config=self.config,
            raw_frame=self._raw_reference_,
            output_features=self.feature_names_out_,
            stages=self.stage_names_,
            schema_table=self.schema.to_frame(),
            drift_summary=self.drift_report_.to_dict() if self.drift_report_ else None,
            fingerprint=getattr(self, "_training_fingerprint_", ""),
        )
        return self.serializer_.save(self.pipeline_, metadata=metadata, filename=filename)

    @classmethod
    def load(cls, path: Union[str, Path], *, strict: bool = False) -> LoadedArtifact:
        """Restore a serialized pipeline.

        The returned :class:`~datapulse.pipeline.serialization.LoadedArtifact`
        exposes ``transform`` directly - a serving process never needs to
        construct a :class:`DataPulseEngine`.
        """
        return PipelineSerializer.load(path, strict=strict)

    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not self.is_fitted_ or self.pipeline_ is None:
            raise NotFittedError(
                "DataPulseEngine is not fitted. Call fit_transform(train_df) first."
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = (
            f"fitted, {len(self.feature_names_out_)} output features"
            if self.is_fitted_
            else "unfitted"
        )
        return f"<DataPulseEngine target={self.config.target_column!r} ({state})>"
