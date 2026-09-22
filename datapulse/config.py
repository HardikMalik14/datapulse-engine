"""Pydantic-driven configuration for every DataPulse Engine transformer.

The whole framework is *configuration first*: no transformer accepts loose
keyword arguments that are not mirrored in a validated Pydantic model. This
gives three properties that matter in production:

1. **Fail fast** - an impossible configuration (e.g. a contamination rate of
   ``1.4``) is rejected at construction time, not three hours into a training
   job.
2. **Serialisable** - the exact configuration that produced an artifact is
   round-trippable to JSON/YAML and shipped alongside the ``.joblib`` bundle.
3. **Self-documenting** - ``PipelineConfig.model_json_schema()`` yields a full
   JSON-Schema description of the pipeline surface area.

Examples
--------
>>> from datapulse.config import PipelineConfig
>>> cfg = PipelineConfig(target_column="churn", task_type="classification")
>>> cfg.imputation.numerical_strategy
<NumericalImputationStrategy.KNN: 'knn'>

"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

__all__ = [
    "TaskType",
    "NumericalImputationStrategy",
    "CategoricalImputationStrategy",
    "OutlierMethod",
    "OutlierAction",
    "ScalerKind",
    "ResamplingStrategy",
    "NormalityTest",
    "SchemaConfig",
    "ImputationConfig",
    "OutlierConfig",
    "ScalingConfig",
    "EncodingConfig",
    "ResamplingConfig",
    "FeatureSelectionConfig",
    "DriftConfig",
    "ReportingConfig",
    "SerializationConfig",
    "PipelineConfig",
]

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class TaskType(str, Enum):
    """Supervised learning task flavour."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class NumericalImputationStrategy(str, Enum):
    """Strategies available for numerical missing-value imputation."""

    MEAN = "mean"
    MEDIAN = "median"
    KNN = "knn"
    ITERATIVE = "iterative"
    CONSTANT = "constant"


class CategoricalImputationStrategy(str, Enum):
    """Strategies available for categorical missing-value imputation."""

    MODE = "mode"
    FREQUENCY = "frequency"
    CONSTANT = "constant"


class OutlierMethod(str, Enum):
    """Detection back-ends for the outlier engine."""

    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    HYBRID = "hybrid"
    NONE = "none"


class OutlierAction(str, Enum):
    """What to do with a detected outlier."""

    CLIP = "clip"
    """Winsorise to the fitted lower/upper fence (inference safe)."""

    MASK = "mask"
    """Replace with ``NaN`` so a downstream imputer handles it."""

    DROP = "drop"
    """Remove the row - *training only*, handled via ``fit_resample``."""

    FLAG = "flag"
    """Keep the value but append a binary ``__is_outlier`` indicator."""


class ScalerKind(str, Enum):
    """Scaler applied once skewness has been addressed."""

    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


class NormalityTest(str, Enum):
    """Statistical test used to decide whether a column is 'normal enough'."""

    DAGOSTINO = "dagostino"
    SHAPIRO = "shapiro"
    AUTO = "auto"
    """D'Agostino-Pearson for n >= 20, Shapiro-Wilk below that."""


class ResamplingStrategy(str, Enum):
    """Synthetic oversampling algorithms."""

    SMOTE = "smote"
    BORDERLINE_SMOTE = "borderline_smote"
    ADASYN = "adasyn"
    SMOTENC = "smotenc"
    NONE = "none"


class MutualInfoMode(str, Enum):
    """How mutual-information filtering selects survivors."""

    TOP_K = "top_k"
    PERCENTILE = "percentile"
    THRESHOLD = "threshold"


# --------------------------------------------------------------------------- #
# Section configs
# --------------------------------------------------------------------------- #
class _StrictModel(BaseModel):
    """Base model: strict, immutable-ish, enum-aware."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        frozen=False,
        arbitrary_types_allowed=False,
    )


class SchemaConfig(_StrictModel):
    """Module 1 - automatic schema and dtype inference."""

    enabled: bool = True
    high_cardinality_threshold: PositiveInt = Field(
        default=15,
        description="Nunique strictly above this becomes high-cardinality categorical.",
    )
    cardinality_ratio_threshold: UnitFloat = Field(
        default=0.5,
        description=(
            "nunique / n_rows above this marks a column as an identifier-like "
            "column and excludes it from modelling."
        ),
    )
    numeric_as_categorical_max_unique: NonNegativeInt = Field(
        default=10,
        description=(
            "Integer columns with at most this many unique values are treated "
            "as categorical (e.g. 0/1 flags, 1-5 ratings)."
        ),
    )
    datetime_parse_threshold: UnitFloat = Field(
        default=0.8,
        description="Fraction of object values that must parse as dates.",
    )
    drop_constant_columns: bool = True
    drop_identifier_columns: bool = True
    expand_datetime_features: bool = Field(
        default=True,
        description="Derive year/month/day/dow/hour plus cyclical encodings.",
    )
    cyclical_encoding: bool = True
    datetime_reference: Optional[str] = Field(
        default=None,
        description="ISO date used to compute 'days since' features; defaults to max seen.",
    )
    force_numerical: list[str] = Field(default_factory=list)
    force_categorical: list[str] = Field(default_factory=list)
    force_datetime: list[str] = Field(default_factory=list)
    force_drop: list[str] = Field(default_factory=list)


class ImputationConfig(_StrictModel):
    """Module 2 - context-aware missing value handling."""

    enabled: bool = True
    numerical_strategy: NumericalImputationStrategy = NumericalImputationStrategy.KNN
    categorical_strategy: CategoricalImputationStrategy = (
        CategoricalImputationStrategy.MODE
    )
    knn_neighbors: PositiveInt = 5
    knn_weights: Literal["uniform", "distance"] = "distance"
    iterative_max_iter: PositiveInt = 10
    iterative_estimator: Literal["bayesian_ridge", "random_forest", "extra_trees"] = (
        "bayesian_ridge"
    )
    numerical_fill_value: float = 0.0
    categorical_fill_value: str = "__MISSING__"
    add_missing_indicators: bool = Field(
        default=True,
        description="Append `<col>__was_missing` flags; missingness is often signal.",
    )
    missing_indicator_min_rate: UnitFloat = Field(
        default=0.01,
        description="Only add an indicator when the column's NA rate exceeds this.",
    )
    drop_columns_above_na_rate: UnitFloat = Field(
        default=0.6,
        description="Columns more empty than this are dropped outright.",
    )


class OutlierConfig(_StrictModel):
    """Module 3 - outlier detection and treatment."""

    enabled: bool = True
    method: OutlierMethod = OutlierMethod.HYBRID
    action: OutlierAction = OutlierAction.CLIP
    iqr_multiplier: float = Field(default=1.5, gt=0.0)
    contamination: Union[UnitFloat, Literal["auto"]] = "auto"
    n_estimators: PositiveInt = 200
    max_samples: Union[Literal["auto"], PositiveInt] = "auto"
    winsorize_quantiles: tuple[UnitFloat, UnitFloat] = (0.01, 0.99)
    max_drop_fraction: UnitFloat = Field(
        default=0.05,
        description="Safety valve: never drop more than this share of training rows.",
    )
    exclude_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_quantiles(self) -> OutlierConfig:
        low, high = self.winsorize_quantiles
        if low >= high:
            raise ValueError(
                f"winsorize_quantiles must be increasing, got ({low}, {high})"
            )
        return self


class ScalingConfig(_StrictModel):
    """Module 4 - skew detection plus adaptive scaling."""

    enabled: bool = True
    normality_test: NormalityTest = NormalityTest.AUTO
    alpha: UnitFloat = Field(default=0.05, description="Normality test significance.")
    skew_threshold: float = Field(
        default=0.75,
        ge=0.0,
        description="|skew| above this triggers a power transform.",
    )
    allow_box_cox: bool = Field(
        default=True,
        description="Box-Cox needs strictly positive data; Yeo-Johnson is the fallback.",
    )
    default_scaler: ScalerKind = ScalerKind.ROBUST
    normal_scaler: ScalerKind = Field(
        default=ScalerKind.STANDARD,
        description="Scaler for columns that pass the normality test.",
    )
    standardize_after_power: bool = True
    max_test_sample: PositiveInt = Field(
        default=5000,
        description="Sub-sample size for normality testing on large frames.",
    )
    exclude_columns: list[str] = Field(default_factory=list)


class EncodingConfig(_StrictModel):
    """Module 5 - categorical encoding with leakage-safe target encoding."""

    enabled: bool = True
    one_hot_max_cardinality: PositiveInt = Field(
        default=15,
        description="At or below this cardinality a column is one-hot encoded.",
    )
    drop_first: bool = False
    handle_unknown: Literal["ignore", "infrequent_if_exist"] = "infrequent_if_exist"
    min_frequency: Optional[UnitFloat] = Field(
        default=0.01,
        description="One-hot categories rarer than this collapse into `infrequent`.",
    )
    target_encoding_enabled: bool = True
    target_encoding_folds: PositiveInt = Field(
        default=5,
        description="Out-of-fold splits used to build leakage-free encodings.",
    )
    target_encoding_smoothing: float = Field(
        default=10.0,
        ge=0.0,
        description="Bayesian shrinkage toward the global prior (m-estimate).",
    )
    target_encoding_noise: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional Gaussian noise added to OOF encodings for regularisation.",
    )
    add_frequency_encoding: bool = Field(
        default=True,
        description="Also emit `<col>__freq` counts for high-cardinality columns.",
    )
    unseen_category_value: Literal["prior", "nan", "zero"] = "prior"


class ResamplingConfig(_StrictModel):
    """Module 6 - class imbalance handling (training-time only)."""

    enabled: bool = True
    strategy: ResamplingStrategy = ResamplingStrategy.SMOTE
    sampling_strategy: Union[Literal["auto", "minority", "not majority"], float, dict] = (
        "auto"
    )
    k_neighbors: PositiveInt = 5
    m_neighbors: PositiveInt = Field(default=10, description="Borderline-SMOTE only.")
    kind: Literal["borderline-1", "borderline-2"] = "borderline-1"
    min_minority_samples: PositiveInt = Field(
        default=6,
        description="Refuse to synthesise when the minority class is tinier than this.",
    )
    imbalance_ratio_trigger: float = Field(
        default=1.5,
        ge=1.0,
        description="Skip resampling when majority/minority is below this ratio.",
    )


class FeatureSelectionConfig(_StrictModel):
    """Module 7 - hybrid variance / mutual-information / L1-RFE selector."""

    enabled: bool = True
    variance_threshold: float = Field(default=0.0, ge=0.0)
    drop_correlated_above: Optional[UnitFloat] = Field(
        default=0.95,
        description="Greedy removal of one of each highly-correlated pair.",
    )
    mutual_info_enabled: bool = True
    mutual_info_mode: MutualInfoMode = MutualInfoMode.PERCENTILE
    mutual_info_percentile: UnitFloat = 0.75
    mutual_info_top_k: PositiveInt = 50
    mutual_info_threshold: float = Field(default=0.0, ge=0.0)
    rfe_enabled: bool = True
    rfe_n_features: Optional[PositiveInt] = Field(
        default=None,
        description="Absolute target count; when None, `rfe_fraction` is used.",
    )
    rfe_fraction: UnitFloat = Field(default=0.6, gt=0.0)
    rfe_step: Union[PositiveInt, UnitFloat] = 0.1
    rfe_use_cv: bool = False
    rfe_cv_folds: PositiveInt = 3
    l1_C: float = Field(default=0.1, gt=0.0, description="Inverse L1 strength.")
    min_features_to_keep: PositiveInt = 5
    protected_features: list[str] = Field(
        default_factory=list,
        description="Never dropped, regardless of statistics (domain must-haves).",
    )

    @model_validator(mode="after")
    def _check_percentile(self) -> FeatureSelectionConfig:
        if self.mutual_info_percentile <= 0:
            raise ValueError("mutual_info_percentile must be > 0")
        return self


class DriftConfig(_StrictModel):
    """Module 8 - data drift and quality auditing."""

    enabled: bool = True
    psi_bins: PositiveInt = 10
    psi_binning: Literal["quantile", "uniform"] = "quantile"
    psi_warn_threshold: float = Field(default=0.10, gt=0.0)
    psi_alert_threshold: float = Field(default=0.25, gt=0.0)
    wasserstein_warn_threshold: float = Field(
        default=0.10,
        gt=0.0,
        description="On standardised (z-scored) scale.",
    )
    wasserstein_alert_threshold: float = Field(default=0.25, gt=0.0)
    ks_test_enabled: bool = True
    epsilon: float = Field(
        default=1e-6,
        gt=0.0,
        description="Additive smoothing so empty PSI bins do not explode to infinity.",
    )
    max_columns_reported: PositiveInt = 200

    @model_validator(mode="after")
    def _ordered_thresholds(self) -> DriftConfig:
        if self.psi_warn_threshold >= self.psi_alert_threshold:
            raise ValueError("psi_warn_threshold must be < psi_alert_threshold")
        if self.wasserstein_warn_threshold >= self.wasserstein_alert_threshold:
            raise ValueError(
                "wasserstein_warn_threshold must be < wasserstein_alert_threshold"
            )
        return self


class ReportingConfig(_StrictModel):
    """Module 9 - visual reporting suite."""

    enabled: bool = True
    output_dir: Path = Path("reports")
    dpi: PositiveInt = 130
    figure_format: Literal["png", "svg", "pdf"] = "png"
    style: str = "whitegrid"
    palette: str = "mako"
    max_heatmap_features: PositiveInt = 40
    generate_correlation_heatmap: bool = True
    generate_nullity_matrix: bool = True
    generate_class_balance: bool = True
    generate_skew_report: bool = True
    generate_drift_report: bool = True
    generate_feature_importance: bool = True


class SerializationConfig(_StrictModel):
    """Module 10 - artifact packaging."""

    enabled: bool = True
    output_dir: Path = Path("artifacts")
    artifact_name: str = "datapulse_pipeline"
    format: Literal["joblib", "pickle"] = "joblib"
    compress: int = Field(default=3, ge=0, le=9)
    include_config: bool = True
    include_schema: bool = True
    write_metadata_sidecar: bool = True


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #
class PipelineConfig(_StrictModel):
    """Root configuration object driving the whole engine.

    Parameters
    ----------
    target_column:
        Name of the supervised target inside the training frame.
    task_type:
        ``classification`` or ``regression``; controls encoders, selectors and
        whether resampling is legal at all.
    random_state:
        Global seed propagated to every stochastic component.

    Examples
    --------
    >>> cfg = PipelineConfig(target_column="y", task_type="classification")
    >>> cfg.outliers.action = OutlierAction.CLIP
    >>> _ = cfg.to_json()

    """

    target_column: str = Field(..., min_length=1)
    task_type: TaskType = TaskType.CLASSIFICATION
    random_state: int = 42
    n_jobs: int = Field(default=-1, description="-1 uses all cores.")
    verbose: bool = True

    schema_inference: SchemaConfig = Field(default_factory=SchemaConfig)
    imputation: ImputationConfig = Field(default_factory=ImputationConfig)
    outliers: OutlierConfig = Field(default_factory=OutlierConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    resampling: ResamplingConfig = Field(default_factory=ResamplingConfig)
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig
    )
    drift: DriftConfig = Field(default_factory=DriftConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)

    @field_validator("target_column")
    @classmethod
    def _strip_target(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("target_column cannot be blank")
        return cleaned

    @model_validator(mode="after")
    def _cross_section_rules(self) -> PipelineConfig:
        """Enforce invariants that span more than one section."""
        if (
            self.task_type is TaskType.REGRESSION
            and self.resampling.enabled
            and self.resampling.strategy is not ResamplingStrategy.NONE
        ):
            # Synthetic minority oversampling is undefined for a continuous
            # target: silently disable rather than fail a long training run.
            self.resampling.enabled = False

        if (
            self.schema_inference.high_cardinality_threshold
            < self.encoding.one_hot_max_cardinality
        ):
            raise ValueError(
                "encoding.one_hot_max_cardinality must not exceed "
                "schema_inference.high_cardinality_threshold, otherwise columns "
                "would be routed to target encoding and one-hot encoding at once."
            )
        return self

    # ------------------------------------------------------------------ #
    # (De)serialisation helpers
    # ------------------------------------------------------------------ #
    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the configuration to a JSON string."""
        return self.model_dump_json(indent=indent)

    def save(self, path: Union[str, Path]) -> Path:
        """Write the configuration to ``path`` as JSON and return the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: Union[str, Path]) -> PipelineConfig:
        """Load a configuration previously written by :meth:`save`."""
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)

    def summary(self) -> str:
        """Return a compact human-readable summary for logs."""
        return (
            f"PipelineConfig(target={self.target_column!r}, "
            f"task={self.task_type.value}, "
            f"impute={self.imputation.numerical_strategy.value}/"
            f"{self.imputation.categorical_strategy.value}, "
            f"outliers={self.outliers.method.value}:{self.outliers.action.value}, "
            f"resample={self.resampling.strategy.value if self.resampling.enabled else 'off'}, "
            f"seed={self.random_state})"
        )
