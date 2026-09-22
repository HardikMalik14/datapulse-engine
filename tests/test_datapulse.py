"""Test-suite for the DataPulse Engine.

The tests are grouped by the property they defend. The most important group is
:class:`TestLeakageGuards` - those are the invariants that separate a pipeline
you can deploy from one that only looks good in a notebook.

Run with::

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datapulse import DataPulseEngine, PipelineConfig
from datapulse.config import (
    CategoricalImputationStrategy,
    EncodingConfig,
    FeatureSelectionConfig,
    ImputationConfig,
    NumericalImputationStrategy,
    OutlierAction,
    OutlierConfig,
    OutlierMethod,
    ResamplingConfig,
    ResamplingStrategy,
    ScalingConfig,
    SchemaConfig,
    TaskType,
)
from datapulse.core import (
    CategoricalEncoderPipeline,
    ContextAwareImputer,
    DriftAuditor,
    FeatureKind,
    HybridFeatureSelector,
    OutlierEngine,
    SchemaInferencer,
    SkewAwareScaler,
    SmartResampler,
)
from datapulse.data.synthetic import generate_messy_dataset
from datapulse.exceptions import (
    LeakageGuardError,
    NotFittedError,
    SchemaError,
    TransformerError,
)

SEED = 7


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def messy_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small train/test split of the synthetic messy dataset."""
    return generate_messy_dataset(n_rows=1_200, random_state=SEED, split=True)


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> PipelineConfig:
    """A fast configuration suitable for tests."""
    root = tmp_path_factory.mktemp("datapulse")
    cfg = PipelineConfig(target_column="churned", random_state=SEED, n_jobs=1)
    cfg.imputation.numerical_strategy = NumericalImputationStrategy.MEDIAN
    cfg.outliers.n_estimators = 50
    cfg.reporting.output_dir = root / "reports"
    cfg.serialization.output_dir = root / "artifacts"
    return cfg


@pytest.fixture(scope="module")
def fitted(messy_split, config):
    """A fitted engine plus its training and holdout matrices."""
    train, test = messy_split
    engine = DataPulseEngine(config)
    X_train, y_train = engine.fit_transform(train)
    X_test, y_test = engine.transform(test, with_target=True)
    return engine, X_train, y_train, X_test, y_test


# --------------------------------------------------------------------------- #
class TestConfiguration:
    """Pydantic validation is the first line of defence."""

    def test_rejects_blank_target(self):
        with pytest.raises(ValueError):
            PipelineConfig(target_column="   ")

    def test_rejects_unknown_field(self):
        with pytest.raises(ValueError):
            PipelineConfig(target_column="y", nonexistent_option=True)

    def test_rejects_inverted_winsorize_quantiles(self):
        with pytest.raises(ValueError):
            OutlierConfig(winsorize_quantiles=(0.9, 0.1))

    def test_rejects_inverted_drift_thresholds(self):
        from datapulse.config import DriftConfig

        with pytest.raises(ValueError):
            DriftConfig(psi_warn_threshold=0.5, psi_alert_threshold=0.2)

    def test_regression_disables_resampling(self):
        cfg = PipelineConfig(target_column="y", task_type=TaskType.REGRESSION)
        assert cfg.resampling.enabled is False

    def test_round_trips_through_json(self, tmp_path):
        cfg = PipelineConfig(target_column="y")
        path = cfg.save(tmp_path / "cfg.json")
        assert PipelineConfig.load(path).target_column == "y"


# --------------------------------------------------------------------------- #
class TestSchemaInference:
    """Module 1."""

    def test_classifies_every_kind(self):
        frame = pd.DataFrame(
            {
                "amount": ["1,200.50", "980.10", "2,340.00", "150.75"] * 30,
                "when": pd.date_range("2024-01-01", periods=120).astype(str),
                "tier": ["a", "b", "c", "a"] * 30,
                "city": [f"c{i % 40}" for i in range(120)],
                "row_id": range(1000, 1120),
                "constant": ["x"] * 120,
            }
        )
        inferencer = SchemaInferencer(SchemaConfig(high_cardinality_threshold=10))
        inferencer.fit(frame)
        schema = inferencer.schema_

        assert "amount" in schema.numerical
        assert "when" in schema.datetime
        assert "tier" in schema.categorical_low
        assert "city" in schema.categorical_high
        assert schema.profiles["row_id"].kind is FeatureKind.DROPPED
        assert schema.profiles["constant"].kind is FeatureKind.DROPPED

    def test_transform_coerces_dtypes(self):
        frame = pd.DataFrame({"amount": ["$1.5", "$2.5", "$3.5", "$9.5"] * 10})
        out = SchemaInferencer().fit_transform(frame)
        assert pd.api.types.is_numeric_dtype(out["amount"])

    def test_datetime_features_are_stable_across_batches(self, messy_split):
        train, test = messy_split
        engine_config = SchemaConfig()
        inferencer = SchemaInferencer(engine_config).fit(train)
        from datapulse.core import DateTimeFeaturizer

        featurizer = DateTimeFeaturizer(engine_config)
        train_out = featurizer.fit_transform(inferencer.transform(train))
        # A single-day batch must not change the feature width.
        one_day = test.head(5).copy()
        one_day["signup_date"] = "2022-05-05"
        test_out = featurizer.transform(inferencer.transform(one_day))
        assert list(train_out.columns) == list(test_out.columns)


# --------------------------------------------------------------------------- #
class TestImputation:
    """Module 2."""

    @pytest.mark.parametrize(
        "strategy",
        [
            NumericalImputationStrategy.MEDIAN,
            NumericalImputationStrategy.KNN,
            NumericalImputationStrategy.ITERATIVE,
        ],
    )
    def test_no_nans_survive(self, strategy):
        rng = np.random.default_rng(SEED)
        frame = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
        frame.loc[frame.sample(40, random_state=SEED).index, "a"] = np.nan
        cfg = ImputationConfig(numerical_strategy=strategy, iterative_max_iter=3)
        out = ContextAwareImputer(cfg).fit_transform(frame)
        assert out.isna().sum().sum() == 0

    def test_missing_indicator_is_created(self):
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0] * 10})
        out = ContextAwareImputer(ImputationConfig()).fit_transform(frame)
        assert "a__was_missing" in out.columns
        assert out["a__was_missing"].sum() == 20

    def test_drops_mostly_empty_columns(self):
        frame = pd.DataFrame({"a": [1.0] * 100, "sparse": [np.nan] * 95 + [1.0] * 5})
        imputer = ContextAwareImputer(ImputationConfig(drop_columns_above_na_rate=0.6))
        out = imputer.fit_transform(frame)
        assert "sparse" not in out.columns

    def test_frequency_strategy_preserves_distribution(self):
        values = ["a"] * 70 + ["b"] * 30
        frame = pd.DataFrame({"c": values + [None] * 100})
        cfg = ImputationConfig(
            categorical_strategy=CategoricalImputationStrategy.FREQUENCY,
            add_missing_indicators=False,
        )
        out = ContextAwareImputer(cfg, random_state=SEED).fit_transform(frame)
        share_a = (out["c"] == "a").mean()
        assert 0.55 < share_a < 0.85  # roughly the observed 70/30 split


# --------------------------------------------------------------------------- #
class TestOutliers:
    """Module 3."""

    def test_clip_bounds_extremes(self):
        frame = pd.DataFrame({"x": list(np.arange(100.0)) + [50_000.0]})
        cfg = OutlierConfig(method=OutlierMethod.IQR, action=OutlierAction.CLIP)
        out = OutlierEngine(cfg, random_state=SEED).fit_transform(frame)
        assert out["x"].max() < 1_000.0
        assert len(out) == len(frame)  # clipping never changes row count

    def test_mask_produces_nans(self):
        frame = pd.DataFrame({"x": list(np.arange(100.0)) + [50_000.0]})
        cfg = OutlierConfig(method=OutlierMethod.IQR, action=OutlierAction.MASK)
        out = OutlierEngine(cfg, random_state=SEED).fit_transform(frame)
        assert out["x"].isna().sum() >= 1

    def test_drop_only_via_fit_resample(self):
        rng = np.random.default_rng(SEED)
        frame = pd.DataFrame({"x": rng.normal(size=500)})
        frame.loc[:4, "x"] = 500.0
        target = pd.Series(rng.integers(0, 2, size=500))
        cfg = OutlierConfig(method=OutlierMethod.HYBRID, action=OutlierAction.DROP)
        engine = OutlierEngine(cfg, random_state=SEED, n_jobs=1)

        filtered_x, filtered_y = engine.fit_resample(frame, target)
        assert len(filtered_x) < len(frame)
        assert len(filtered_x) == len(filtered_y)  # X and y stay aligned

        # transform() must never remove rows, even with action='drop'.
        assert len(engine.transform(frame)) == len(frame)

    def test_drop_respects_budget(self):
        rng = np.random.default_rng(SEED)
        frame = pd.DataFrame({"x": rng.normal(size=400)})
        target = pd.Series(rng.integers(0, 2, size=400))
        cfg = OutlierConfig(
            method=OutlierMethod.ISOLATION_FOREST,
            action=OutlierAction.DROP,
            contamination=0.4,
            max_drop_fraction=0.05,
            n_estimators=50,
        )
        filtered_x, _ = OutlierEngine(cfg, random_state=SEED, n_jobs=1).fit_resample(
            frame, target
        )
        assert len(filtered_x) >= int(400 * 0.94)


# --------------------------------------------------------------------------- #
class TestScaling:
    """Module 4."""

    def test_lognormal_gets_a_power_transform(self):
        rng = np.random.default_rng(SEED)
        frame = pd.DataFrame({"income": rng.lognormal(3, 1, 800)})
        scaler = SkewAwareScaler(ScalingConfig(), random_state=SEED).fit(frame)
        decision = scaler.decisions_.iloc[0]
        assert decision["treatment"] in {"box_cox", "yeo_johnson"}
        assert abs(decision["skew_after"]) < abs(decision["skew_before"])

    def test_normal_column_is_standardised_not_transformed(self):
        rng = np.random.default_rng(SEED)
        frame = pd.DataFrame({"z": rng.normal(0, 1, 2_000)})
        scaler = SkewAwareScaler(ScalingConfig(), random_state=SEED).fit(frame)
        assert scaler.decisions_.iloc[0]["treatment"] in {"standard", "robust"}

    def test_negative_values_fall_back_to_yeo_johnson(self):
        rng = np.random.default_rng(SEED)
        values = rng.lognormal(3, 1, 800) - 30.0  # skewed but not positive-only
        scaler = SkewAwareScaler(ScalingConfig(), random_state=SEED).fit(
            pd.DataFrame({"v": values})
        )
        assert scaler.decisions_.iloc[0]["treatment"] == "yeo_johnson"


# --------------------------------------------------------------------------- #
class TestEncoding:
    """Module 5 - including the out-of-fold leakage guard."""

    @staticmethod
    def _leaky_frame(n: int = 900) -> tuple[pd.DataFrame, pd.Series]:
        """A high-cardinality column that *is* the label for rare levels."""
        rng = np.random.default_rng(SEED)
        labels = rng.integers(0, 2, size=n)
        # Each category appears ~3 times, so in-fold means memorise the label.
        categories = [f"k_{i // 3}" for i in range(n)]
        return pd.DataFrame({"key": categories}), pd.Series(labels)

    def test_low_cardinality_gets_one_hot(self):
        frame = pd.DataFrame({"tier": ["a", "b", "c"] * 100})
        target = pd.Series([0, 1, 0] * 100)
        out = CategoricalEncoderPipeline(
            EncodingConfig(), TaskType.CLASSIFICATION
        ).fit_transform(frame, target)
        assert any(c.startswith("tier_") for c in out.columns)

    def test_high_cardinality_gets_target_encoding(self):
        frame = pd.DataFrame({"city": [f"c{i % 60}" for i in range(600)]})
        target = pd.Series(np.resize([0, 1], 600))
        encoder = CategoricalEncoderPipeline(EncodingConfig(), TaskType.CLASSIFICATION)
        out = encoder.fit_transform(frame, target)
        assert encoder.target_columns_ == ["city"]
        assert "city__te" in out.columns

    def test_out_of_fold_encoding_suppresses_leakage(self):
        frame, target = self._leaky_frame()
        encoder = CategoricalEncoderPipeline(EncodingConfig(), TaskType.CLASSIFICATION)
        out_of_fold = encoder.fit_transform(frame, target)["key__te"]
        in_fold = encoder.transform(frame)["key__te"]

        oof_corr = abs(np.corrcoef(out_of_fold, target)[0, 1])
        naive_corr = abs(np.corrcoef(in_fold, target)[0, 1])
        assert naive_corr > 0.25, "the naive encoding should visibly leak"
        assert oof_corr < naive_corr / 2, "OOF encoding must suppress the leak"

    def test_fit_transform_differs_from_transform_on_train(self):
        frame, target = self._leaky_frame()
        encoder = CategoricalEncoderPipeline(EncodingConfig(), TaskType.CLASSIFICATION)
        train_view = encoder.fit_transform(frame, target)["key__te"].to_numpy()
        infer_view = encoder.transform(frame)["key__te"].to_numpy()
        assert not np.allclose(train_view, infer_view)

    def test_unseen_categories_fall_back_to_prior(self):
        frame = pd.DataFrame({"city": [f"c{i % 60}" for i in range(600)]})
        target = pd.Series(np.resize([0, 1], 600))
        encoder = CategoricalEncoderPipeline(EncodingConfig(), TaskType.CLASSIFICATION)
        encoder.fit_transform(frame, target)
        out = encoder.transform(pd.DataFrame({"city": ["never_seen"] * 5}))
        assert out["city__te"].notna().all()
        assert np.allclose(out["city__te"], target.mean(), atol=0.05)


# --------------------------------------------------------------------------- #
class TestResampling:
    """Module 6."""

    @staticmethod
    def _imbalanced(n: int = 600):
        rng = np.random.default_rng(SEED)
        X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
        y = pd.Series([0] * int(n * 0.92) + [1] * (n - int(n * 0.92)))
        return X, y

    @pytest.mark.parametrize(
        "strategy",
        [
            ResamplingStrategy.SMOTE,
            ResamplingStrategy.BORDERLINE_SMOTE,
            ResamplingStrategy.ADASYN,
        ],
    )
    def test_each_strategy_balances(self, strategy):
        X, y = self._imbalanced()
        sampler = SmartResampler(
            ResamplingConfig(strategy=strategy), random_state=SEED, n_jobs=1
        )
        _, resampled_y = sampler.fit_resample(X, y)
        counts = resampled_y.value_counts()
        assert counts.min() / counts.max() > 0.7
        assert sampler.was_applied_

    def test_transform_is_identity(self):
        X, y = self._imbalanced()
        sampler = SmartResampler(ResamplingConfig(), random_state=SEED, n_jobs=1).fit(X, y)
        out = sampler.transform(X)
        assert out.shape == X.shape
        assert np.allclose(out.to_numpy(), X.to_numpy())

    def test_resample_method_always_refuses(self):
        X, y = self._imbalanced()
        with pytest.raises(LeakageGuardError):
            SmartResampler().resample(X, y)

    def test_tiny_minority_is_vetoed(self):
        rng = np.random.default_rng(SEED)
        X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
        y = pd.Series([0] * 197 + [1] * 3)
        sampler = SmartResampler(ResamplingConfig(), random_state=SEED, n_jobs=1)
        _, out_y = sampler.fit_resample(X, y)
        assert not sampler.was_applied_
        assert "minority" in sampler.skip_reason_
        assert len(out_y) == len(y)

    def test_balanced_data_is_left_alone(self):
        rng = np.random.default_rng(SEED)
        X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
        y = pd.Series([0, 1] * 200)
        sampler = SmartResampler(ResamplingConfig(), random_state=SEED, n_jobs=1)
        sampler.fit_resample(X, y)
        assert not sampler.was_applied_

    def test_non_numeric_input_is_rejected(self):
        X, y = self._imbalanced()
        X["text"] = "abc"
        with pytest.raises(TransformerError):
            SmartResampler(ResamplingConfig(), random_state=SEED, n_jobs=1).fit_resample(X, y)


# --------------------------------------------------------------------------- #
class TestFeatureSelection:
    """Module 7."""

    @staticmethod
    def _signal_and_noise(n: int = 600):
        rng = np.random.default_rng(SEED)
        X = pd.DataFrame(rng.normal(size=(n, 10)), columns=[f"noise_{i}" for i in range(10)])
        X["signal_a"] = rng.normal(size=n)
        X["signal_b"] = rng.normal(size=n)
        X["constant"] = 3.0
        y = pd.Series((X["signal_a"] + X["signal_b"] + rng.normal(0, 0.3, n) > 0).astype(int))
        return X, y

    def test_drops_constant_and_keeps_signal(self):
        X, y = self._signal_and_noise()
        cfg = FeatureSelectionConfig(min_features_to_keep=2, rfe_fraction=0.3)
        selector = HybridFeatureSelector(cfg, TaskType.CLASSIFICATION, SEED, n_jobs=1).fit(X, y)
        assert "constant" not in selector.selected_features_
        assert {"signal_a", "signal_b"} <= set(selector.selected_features_)
        assert len(selector.selected_features_) < X.shape[1]

    def test_correlated_duplicate_is_pruned(self):
        X, y = self._signal_and_noise()
        X["signal_a_copy"] = X["signal_a"] * 1.001
        cfg = FeatureSelectionConfig(drop_correlated_above=0.95, min_features_to_keep=2)
        selector = HybridFeatureSelector(cfg, TaskType.CLASSIFICATION, SEED, n_jobs=1).fit(X, y)
        kept = set(selector.selected_features_)
        assert not {"signal_a", "signal_a_copy"} <= kept

    def test_protected_features_survive(self):
        X, y = self._signal_and_noise()
        cfg = FeatureSelectionConfig(
            protected_features=["noise_0"], min_features_to_keep=2, rfe_fraction=0.2
        )
        selector = HybridFeatureSelector(cfg, TaskType.CLASSIFICATION, SEED, n_jobs=1).fit(X, y)
        assert "noise_0" in selector.selected_features_

    def test_min_features_floor_is_respected(self):
        X, y = self._signal_and_noise()
        cfg = FeatureSelectionConfig(min_features_to_keep=6, rfe_fraction=0.05)
        selector = HybridFeatureSelector(cfg, TaskType.CLASSIFICATION, SEED, n_jobs=1).fit(X, y)
        assert len(selector.selected_features_) >= 6


# --------------------------------------------------------------------------- #
class TestDrift:
    """Module 8."""

    def test_identical_frames_are_stable(self):
        rng = np.random.default_rng(SEED)
        frame = pd.DataFrame({"x": rng.normal(size=2_000), "c": rng.choice(list("abc"), 2_000)})
        report = DriftAuditor().audit(frame, frame.copy())
        assert report.is_stable
        assert report.table["psi"].max() < 0.01

    def test_shifted_distribution_alerts(self):
        rng = np.random.default_rng(SEED)
        reference = pd.DataFrame({"x": rng.normal(0, 1, 3_000)})
        current = pd.DataFrame({"x": rng.normal(2.5, 1, 3_000)})
        report = DriftAuditor().audit(reference, current)
        assert not report.is_stable
        assert report.table.loc[0, "psi"] > 0.25

    def test_binary_columns_get_a_real_psi(self):
        """Discrete columns must not collapse to a single bin (PSI == 0)."""
        reference = pd.DataFrame({"flag": [0.0] * 900 + [1.0] * 100})
        current = pd.DataFrame({"flag": [0.0] * 500 + [1.0] * 500})
        report = DriftAuditor().audit(reference, current)
        assert report.table.loc[0, "psi"] > 0.25

    def test_new_categories_are_counted(self):
        reference = pd.DataFrame({"c": ["a", "b"] * 500})
        current = pd.DataFrame({"c": ["a", "b", "zzz"] * 300})
        report = DriftAuditor().audit(reference, current)
        assert int(report.table.loc[0, "new_categories"]) == 1

    def test_requires_shared_columns(self):
        from datapulse.exceptions import DataPulseError

        with pytest.raises(DataPulseError):
            DriftAuditor().audit(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))


# --------------------------------------------------------------------------- #
class TestEngine:
    """The orchestrator wiring stages 1-9 together."""

    def test_output_is_model_ready(self, fitted):
        _, X_train, y_train, X_test, y_test = fitted
        for matrix in (X_train, X_test):
            assert matrix.isna().sum().sum() == 0
            assert matrix.select_dtypes(exclude=[np.number]).empty
            assert np.isfinite(matrix.to_numpy(dtype=float)).all()
        assert list(X_train.columns) == list(X_test.columns)
        assert len(X_train) == len(y_train)

    def test_planted_traps_are_removed(self, fitted):
        engine = fitted[0]
        schema = engine.schema
        assert "customer_id" in schema.dropped
        assert "data_version" in schema.dropped
        assert "monthly_spend" in schema.numerical  # recovered from currency strings
        assert "signup_date" in schema.datetime
        assert "city" in schema.categorical_high

    def test_transform_is_deterministic(self, fitted, messy_split):
        engine = fitted[0]
        _, test = messy_split
        first = engine.transform(test)
        second = engine.transform(test)
        pd.testing.assert_frame_equal(first, second)

    def test_transform_before_fit_raises(self, config, messy_split):
        engine = DataPulseEngine(config)
        with pytest.raises(NotFittedError):
            engine.transform(messy_split[1])

    def test_missing_column_raises_schema_error(self, fitted, messy_split):
        engine = fitted[0]
        _, test = messy_split
        broken = test.drop(columns=["tenure_months"])
        with pytest.raises((SchemaError, TransformerError)):
            engine.transform(broken)

    def test_extra_columns_are_tolerated(self, fitted, messy_split):
        engine = fitted[0]
        _, test = messy_split
        padded = test.copy()
        padded["unexpected_payload_field"] = "ignore me"
        assert len(engine.transform(padded)) == len(test)

    def test_summary_lists_every_stage(self, fitted):
        summary = fitted[0].summary()
        assert set(summary["stage"]) >= {"schema", "imputer", "encoder", "selector", "resampler"}

    def test_downstream_model_learns_the_signal(self, fitted):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        _, X_train, y_train, X_test, y_test = fitted
        model = LogisticRegression(max_iter=2_000).fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        assert auc > 0.70, f"engineered features should be learnable (AUC={auc:.3f})"


# --------------------------------------------------------------------------- #
class TestLeakageGuards:
    """The invariants that make this framework safe to deploy."""

    def test_holdout_row_count_is_preserved(self, fitted, messy_split):
        engine = fitted[0]
        _, test = messy_split
        assert len(engine.transform(test)) == len(test)

    def test_training_matrix_was_resampled_but_holdout_was_not(self, fitted, messy_split):
        engine, X_train, _, X_test, _ = fitted
        train, test = messy_split
        assert len(X_train) > len(train), "SMOTE should have grown the training set"
        assert len(X_test) == len(test), "the holdout must never be resampled"

    def test_serialized_artifact_contains_no_resampler(self, fitted):
        engine = fitted[0]
        assert "resampler" not in dict(engine.pipeline_.named_steps)

    def test_fitting_twice_gives_identical_output(self, config, messy_split):
        train, test = messy_split
        first = DataPulseEngine(config).fit(train).transform(test)
        second = DataPulseEngine(config).fit(train).transform(test)
        pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- #
class TestSerialization:
    """Module 10."""

    def test_round_trip_reproduces_output(self, fitted, messy_split):
        engine, _, _, X_test, _ = fitted
        _, test = messy_split
        path = engine.save(filename="roundtrip")
        assert path.exists()
        assert path.with_suffix(".meta.json").exists()

        restored = DataPulseEngine.load(path)
        features = test.drop(columns=[engine.config.target_column])
        replayed = restored.transform(features).reindex(columns=engine.feature_names_out_)
        np.testing.assert_allclose(
            replayed.to_numpy(dtype=float), X_test.to_numpy(dtype=float)
        )

    def test_metadata_captures_the_contract(self, fitted):
        engine = fitted[0]
        restored = DataPulseEngine.load(engine.save(filename="metadata_check"))
        meta = restored.metadata
        assert meta.target_column == "churned"
        assert meta.n_features_out == len(engine.feature_names_out_)
        assert "scikit-learn" in meta.library_versions
        assert meta.raw_feature_names
        assert restored.check_environment() == []

    def test_missing_artifact_raises(self, tmp_path):
        from datapulse.exceptions import SerializationError

        with pytest.raises(SerializationError):
            DataPulseEngine.load(tmp_path / "nope.joblib")


# --------------------------------------------------------------------------- #
class TestReporting:
    """Module 9."""

    def test_every_report_is_written(self, fitted, messy_split):
        engine, X_train, _, _, _ = fitted
        train, test = messy_split
        engine.audit_drift(train, test)
        generated = engine.generate_reports(
            raw_frame=train.drop(columns=[engine.config.target_column]),
            processed_frame=X_train,
        )
        expected = {
            "nullity_matrix",
            "correlation_heatmap",
            "skew_transformations",
            "feature_relevance",
            "class_balance_smote",
            "drift_psi",
        }
        assert expected <= set(generated)
        assert all(path.exists() and path.stat().st_size > 0 for path in generated.values())
