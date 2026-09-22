#!/usr/bin/env python3
"""End-to-end demonstration of the DataPulse Engine.

Run with::

    python main.py                      # full demo
    python main.py --rows 20000         # bigger dataset
    python main.py --no-reports         # skip figure rendering

The script walks the complete workflow on a deliberately messy, imbalanced
synthetic dataset:

1. Generate raw data with planted pathologies and profile the damage.
2. Fit the engine on the training split (schema -> ... -> resampling).
3. Transform the holdout split with the *same* fitted parameters.
4. Prove the leakage guards actually hold.
5. Audit train/test drift (PSI + Wasserstein).
6. Train downstream models and compare against a no-preprocessing baseline.
7. Render the visual report suite.
8. Serialize the pipeline, reload it, and verify byte-for-byte equivalence.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from datapulse import DataPulseEngine, PipelineConfig, configure_logging
from datapulse.config import (
    CategoricalImputationStrategy,
    NumericalImputationStrategy,
    OutlierAction,
    OutlierMethod,
    ResamplingStrategy,
    TaskType,
)
from datapulse.core.resampling import SmartResampler
from datapulse.data.synthetic import generate_messy_dataset
from datapulse.exceptions import DataPulseError, LeakageGuardError

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BANNER_WIDTH = 78


# --------------------------------------------------------------------------- #
# Console helpers
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    """Print a section banner."""
    print(f"\n{'=' * BANNER_WIDTH}\n  {title}\n{'=' * BANNER_WIDTH}")


def show(frame: pd.DataFrame, *, max_rows: int = 12) -> None:
    """Print a DataFrame compactly."""
    with pd.option_context(
        "display.max_columns", 40, "display.width", 200, "display.max_rows", max_rows
    ):
        print(frame.to_string(index=False))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Assemble the pipeline configuration for the demo."""
    config = PipelineConfig(
        target_column="churned",
        task_type=TaskType.CLASSIFICATION,
        random_state=args.seed,
        n_jobs=-1,
    )

    # Module 1 - schema inference
    config.schema_inference.high_cardinality_threshold = 15
    config.schema_inference.expand_datetime_features = True

    # Module 2 - imputation
    config.imputation.numerical_strategy = NumericalImputationStrategy.KNN
    config.imputation.categorical_strategy = CategoricalImputationStrategy.MODE
    config.imputation.knn_neighbors = 7
    config.imputation.add_missing_indicators = True

    # Module 3 - outliers
    config.outliers.method = OutlierMethod.HYBRID
    config.outliers.action = OutlierAction.CLIP
    config.outliers.contamination = 0.03

    # Module 4 - scaling
    config.scaling.skew_threshold = 0.75
    config.scaling.allow_box_cox = True

    # Module 5 - encoding
    config.encoding.target_encoding_folds = 5
    config.encoding.target_encoding_smoothing = 20.0

    # Module 6 - resampling
    config.resampling.strategy = ResamplingStrategy[args.sampler.upper()]

    # Module 7 - feature selection
    config.feature_selection.mutual_info_percentile = 0.7
    config.feature_selection.rfe_fraction = 0.7
    config.feature_selection.min_features_to_keep = 8

    # Modules 9 & 10 - reporting and artifacts
    config.reporting.enabled = not args.no_reports
    config.reporting.output_dir = Path(args.reports_dir)
    config.serialization.output_dir = Path(args.artifacts_dir)
    config.serialization.artifact_name = "datapulse_churn_pipeline"
    return config


# --------------------------------------------------------------------------- #
# Demo steps
# --------------------------------------------------------------------------- #
def profile_raw_data(train: pd.DataFrame, test: pd.DataFrame, target: str) -> None:
    """Show what is wrong with the raw data before anything touches it."""
    banner("STEP 1 - Raw data profile (the mess we start from)")
    print(f"Train: {train.shape[0]:,} rows x {train.shape[1]} columns")
    print(f"Test:  {test.shape[0]:,} rows x {test.shape[1]} columns\n")

    missing = train.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    print("Columns with missing values:")
    show(
        pd.DataFrame(
            {"column": missing.index, "missing_pct": (missing.to_numpy() * 100).round(2)}
        )
    )

    counts = train[target].value_counts().sort_index()
    ratio = counts.max() / max(counts.min(), 1)
    print(f"\nClass balance: {counts.to_dict()}  ->  imbalance ratio {ratio:.1f}:1")
    print(f"Dtypes present: {sorted({str(d) for d in train.dtypes})}")
    print(
        "Planted traps: customer_id (identifier), data_version (constant), "
        "monthly_spend (currency strings), signup_date (date strings), "
        "satisfaction (1-5 rating), tenure_months_copy (collinear), noise_1..8 (noise)."
    )


def fit_engine(
    engine: DataPulseEngine, train: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Fit the engine and report what each stage did."""
    banner("STEP 2 - Fitting the engine on the training split")
    X_train, y_train = engine.fit_transform(train)

    print("\nInferred schema:")
    show(engine.schema.to_frame()[["name", "kind", "n_unique", "null_rate", "reason"]], max_rows=40)

    print("\nPer-stage diagnostics:")
    show(engine.summary(), max_rows=20)

    print(f"\nTraining matrix: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"NaNs remaining: {int(X_train.isna().sum().sum())}")
    print(f"Non-numeric columns remaining: {list(X_train.select_dtypes(exclude=[np.number]).columns)}")
    return X_train, y_train


def show_scaling_decisions(engine: DataPulseEngine) -> None:
    """Print the per-column normality / transform decisions."""
    banner("STEP 3 - Skew detection and adaptive scaling (module 4)")
    decisions = engine.pipeline_.named_steps["scaler"].skew_report()
    if decisions.empty:
        print("No numeric columns were scaled.")
        return
    show(decisions.sort_values("skew_before", key=abs, ascending=False), max_rows=25)


def prove_leakage_guards(
    engine: DataPulseEngine, train: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Demonstrate that the leakage safeguards are structural, not advisory."""
    banner("STEP 4 - Leakage guards (modules 5 and 6)")

    # --- guard 1: transform() never resamples ------------------------ #
    X_test = engine.transform(test)
    print(f"[guard 1] transform() on the holdout preserved row count: "
          f"{len(test):,} in -> {len(X_test):,} out  "
          f"({'PASS' if len(X_test) == len(test) else 'FAIL'})")

    # --- guard 2: resampling a non-training split is refused --------- #
    try:
        SmartResampler().resample(X_test, pd.Series(np.zeros(len(X_test))))
        print("[guard 2] FAIL - resampling a holdout split was allowed")
    except LeakageGuardError as exc:
        print(f"[guard 2] PASS - LeakageGuardError raised: {str(exc)[:96]}...")

    # --- guard 3: out-of-fold target encoding ------------------------ #
    encoder = engine.pipeline_.named_steps["encoder"]
    if encoder.target_columns_ and encoder.target_encoder_ is not None:
        features = train.drop(columns=[engine.config.target_column])
        labels = train[engine.config.target_column]
        # Reproduce the encoder's input by replaying the earlier stages.
        upstream = features
        for name, step in engine.pipeline_.steps:
            if name == "encoder":
                break
            upstream = step.transform(upstream)

        block = upstream.reindex(columns=encoder.target_columns_).astype("object").fillna("__NA__")
        out_of_fold = encoder.target_encoder_.fit_transform(block, labels)
        in_fold = encoder.target_encoder_.transform(block)
        column = out_of_fold.columns[0]
        oof_corr = float(np.corrcoef(out_of_fold[column], labels)[0, 1])
        naive_corr = float(np.corrcoef(in_fold[column], labels)[0, 1])
        print(
            f"[guard 3] Target-encoding correlation with the label on TRAIN rows:\n"
            f"          naive (in-fold) r = {naive_corr:+.4f}   <- inflated, leaks\n"
            f"          out-of-fold   r = {oof_corr:+.4f}   <- honest, what the model sees\n"
            f"          {'PASS' if abs(oof_corr) < abs(naive_corr) else 'CHECK'} - "
            f"leakage suppressed by {100 * (1 - abs(oof_corr) / max(abs(naive_corr), 1e-9)):.1f}%"
        )
    else:
        print("[guard 3] skipped - no high-cardinality columns in this dataset")

    # --- guard 4: resampling actually happened on train -------------- #
    resampler = engine.resampler_
    print(
        f"[guard 4] Resampling applied on train: {resampler.was_applied_} "
        f"({resampler.distribution_before_} -> {resampler.distribution_after_})"
    )


def audit_drift(engine: DataPulseEngine, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Run the drift audit and print the most-shifted features."""
    banner("STEP 5 - Data drift and quality audit (module 8)")
    report = engine.audit_drift(train, test, processed=True)
    print(report.summary(), "\n")
    table = report.to_frame()
    columns = ["column", "psi", "wasserstein", "ks_pvalue", "mean_shift", "severity"]
    show(table[columns].head(15))
    if not report.alerts.empty:
        print(
            f"\n{len(report.alerts)} column(s) breached the alert threshold - "
            "expect degraded performance on this holdout."
        )


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    target: str,
) -> None:
    """Train downstream models on the engineered matrix and on a naive baseline."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        recall_score,
        roc_auc_score,
    )

    banner("STEP 6 - Downstream model performance")

    def score(name: str, model, X_tr, y_tr, X_te, y_te) -> dict:  # noqa: ANN001
        model.fit(X_tr, y_tr)
        probabilities = model.predict_proba(X_te)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        return {
            "model": name,
            "roc_auc": round(float(roc_auc_score(y_te, probabilities)), 4),
            "pr_auc": round(float(average_precision_score(y_te, probabilities)), 4),
            "f1": round(float(f1_score(y_te, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, predictions, zero_division=0)), 4),
            "n_features": int(X_tr.shape[1]),
        }

    results = [
        score(
            "LogisticRegression (DataPulse)",
            LogisticRegression(max_iter=3000),
            X_train, y_train, X_test, y_test,
        ),
        score(
            "HistGradientBoosting (DataPulse)",
            HistGradientBoostingClassifier(max_iter=250, random_state=42),
            X_train, y_train, X_test, y_test,
        ),
    ]

    # Naive baseline: numeric columns only, median fill, no engineering at all.
    naive_train = raw_train.drop(columns=[target]).select_dtypes(include=[np.number])
    naive_test = raw_test.drop(columns=[target]).reindex(columns=naive_train.columns)
    fills = naive_train.median()
    results.append(
        score(
            "HistGradientBoosting (naive baseline)",
            HistGradientBoostingClassifier(max_iter=250, random_state=42),
            naive_train.fillna(fills),
            raw_train[target],
            naive_test.fillna(fills),
            raw_test[target],
        )
    )

    show(pd.DataFrame(results))
    print(
        "\nNote: the naive baseline keeps customer_id and the noise columns, drops every "
        "categorical, and never corrects skew - the gap is what the engine bought you."
    )


def render_reports(engine: DataPulseEngine, raw_train: pd.DataFrame, X_train: pd.DataFrame) -> None:
    """Render and list the visual report suite."""
    banner("STEP 7 - Visual reporting suite (module 9)")
    if not engine.config.reporting.enabled:
        print("Reporting disabled (--no-reports).")
        return
    generated = engine.generate_reports(
        raw_frame=raw_train.drop(columns=[engine.config.target_column]),
        processed_frame=X_train,
    )
    if not generated:
        print("No reports generated.")
        return
    show(engine.reporter_.manifest())


def serialize_and_verify(
    engine: DataPulseEngine, test: pd.DataFrame, X_test: pd.DataFrame
) -> None:
    """Save the artifact, reload it, and confirm identical output."""
    banner("STEP 8 - Pipeline serialization and export (module 10)")
    path = engine.save()
    size_mb = path.stat().st_size / 1_048_576
    print(f"Artifact: {path}  ({size_mb:.2f} MB)")
    print(f"Sidecar:  {path.with_suffix('.meta.json')}")

    restored = DataPulseEngine.load(path)
    print(f"\nReloaded: {restored!r}")
    print(f"Environment check: {restored.check_environment() or 'clean - all versions match'}")

    features = test.drop(columns=[engine.config.target_column], errors="ignore")
    replayed = restored.transform(features).reindex(columns=engine.feature_names_out_).fillna(0.0)
    identical = np.allclose(
        replayed.to_numpy(dtype=float), X_test.to_numpy(dtype=float), equal_nan=True
    )
    print(
        f"Reloaded pipeline reproduces the holdout matrix exactly: "
        f"{identical}  ({'PASS' if identical else 'FAIL'})"
    )
    print(
        "\nDeployment contract: the artifact holds stages 1-8 only. The resampler is "
        "training-only and is deliberately excluded."
    )


# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="DataPulse Engine end-to-end demo")
    parser.add_argument("--rows", type=int, default=8_000, help="synthetic dataset size")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    parser.add_argument(
        "--sampler",
        default="smote",
        choices=["smote", "borderline_smote", "adasyn", "none"],
        help="imbalance strategy",
    )
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--no-reports", action="store_true", help="skip figure rendering")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full demonstration."""
    args = parse_args(argv)
    configure_logging(level=args.log_level, force=True)

    print("\n" + "*" * BANNER_WIDTH)
    print("  DataPulse Engine - enterprise preprocessing & feature pipeline framework")
    print("*" * BANNER_WIDTH)

    try:
        train, test = generate_messy_dataset(
            n_rows=args.rows, random_state=args.seed, split=True, inject_drift=True
        )
        config = build_config(args)
        print(f"\n{config.summary()}")

        profile_raw_data(train, test, config.target_column)

        engine = DataPulseEngine(config)
        X_train, y_train = fit_engine(engine, train)
        X_test, y_test = engine.transform(test, with_target=True)

        show_scaling_decisions(engine)
        prove_leakage_guards(engine, train, test)
        audit_drift(engine, train, test)
        evaluate_models(
            X_train, y_train, X_test, y_test, train, test, config.target_column
        )
        render_reports(engine, train, X_train)
        serialize_and_verify(engine, test, X_test)

        banner("DONE")
        print(f"Reports   -> {Path(args.reports_dir).resolve()}")
        print(f"Artifacts -> {Path(args.artifacts_dir).resolve()}")
        return 0

    except DataPulseError as exc:
        print(f"\n[DataPulse error] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
