"""DataPulse Engine - modular data preprocessing and feature engineering.

A configuration-driven, Scikit-Learn-compatible framework that turns raw,
messy tabular data into a model-ready feature matrix, and packages the whole
transformation chain into a single deployable artifact.

Quick start
-----------
>>> from datapulse import DataPulseEngine, PipelineConfig
>>> from datapulse.data.synthetic import generate_messy_dataset
>>> train, test = generate_messy_dataset(n_rows=800, random_state=0, split=True)
>>> engine = DataPulseEngine(PipelineConfig(target_column="churned"))
>>> X_train, y_train = engine.fit_transform(train)
>>> X_test, y_test = engine.transform(test, with_target=True)
>>> artifact = engine.save()
"""

from __future__ import annotations

__version__ = "1.0.0"

from datapulse.base import AbstractTransformer
from datapulse.config import (
    CategoricalImputationStrategy,
    NormalityTest,
    NumericalImputationStrategy,
    OutlierAction,
    OutlierMethod,
    PipelineConfig,
    ResamplingStrategy,
    ScalerKind,
    TaskType,
)
from datapulse.core import (
    CategoricalEncoderPipeline,
    ContextAwareImputer,
    DataSchema,
    DateTimeFeaturizer,
    DriftAuditor,
    DriftReport,
    HybridFeatureSelector,
    OutlierEngine,
    SchemaInferencer,
    SkewAwareScaler,
    SmartResampler,
)
from datapulse.exceptions import (
    ConfigurationError,
    DataPulseError,
    LeakageGuardError,
    NotFittedError,
    SchemaError,
    SerializationError,
    TransformerError,
)
from datapulse.logger import configure_logging, get_logger
from datapulse.pipeline import (
    ArtifactMetadata,
    DataPulseEngine,
    LoadedArtifact,
    PipelineSerializer,
)
from datapulse.reporting import VisualReporter

__all__ = [
    "__version__",
    # framework
    "AbstractTransformer",
    "DataPulseEngine",
    "PipelineConfig",
    # config enums
    "CategoricalImputationStrategy",
    "NormalityTest",
    "NumericalImputationStrategy",
    "OutlierAction",
    "OutlierMethod",
    "ResamplingStrategy",
    "ScalerKind",
    "TaskType",
    # modules
    "CategoricalEncoderPipeline",
    "ContextAwareImputer",
    "DataSchema",
    "DateTimeFeaturizer",
    "DriftAuditor",
    "DriftReport",
    "HybridFeatureSelector",
    "OutlierEngine",
    "SchemaInferencer",
    "SkewAwareScaler",
    "SmartResampler",
    "VisualReporter",
    # serialization
    "ArtifactMetadata",
    "LoadedArtifact",
    "PipelineSerializer",
    # errors + logging
    "ConfigurationError",
    "DataPulseError",
    "LeakageGuardError",
    "NotFittedError",
    "SchemaError",
    "SerializationError",
    "TransformerError",
    "configure_logging",
    "get_logger",
]
