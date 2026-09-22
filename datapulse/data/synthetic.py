"""A deliberately awful synthetic dataset for exercising the whole engine.

Every pathology the framework claims to handle is planted here on purpose, so
the demo is a genuine test rather than a happy path:

==========================================  ===================================
Planted problem                             Module that must handle it
==========================================  ===================================
``customer_id`` - near-unique integers      1 (dropped as an identifier)
``data_version`` - constant column          1 (dropped as degenerate)
``monthly_spend`` - numbers stored as text  1 (coerced to numeric)
``signup_date`` - ISO strings               1 (parsed, exploded to features)
``satisfaction`` - 1-5 integer ratings      1 (treated as categorical)
Missing values, MCAR *and* structured       2 (imputation + indicators)
``annual_income`` - log-normal, heavy tail  3, 4 (winsorize, then Box-Cox)
Injected sensor spikes in ``session_count`` 3 (outlier clipping)
``city`` - 120 levels                       5 (out-of-fold target encoding)
``plan_tier`` - 4 levels                    5 (one-hot encoding)
``noise_1..8`` - pure noise                 7 (feature selection must drop)
``tenure_months_copy`` - collinear duplicate 7 (correlation pruning)
~7% positive class                          6 (SMOTE / ADASYN)
Train/test shift in two columns             8 (PSI / Wasserstein must flag)
==========================================  ===================================

The target is generated from a known logistic model over a handful of the
"real" features, so a downstream classifier *should* be able to learn it -
which is what makes the end-to-end demo meaningful.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

__all__ = ["generate_messy_dataset"]

_PLAN_TIERS = ("free", "basic", "pro", "enterprise")
_CHANNELS = ("organic", "paid_search", "referral", "social", "email")
_DEVICES = ("ios", "android", "web")


def generate_messy_dataset(
    n_rows: int = 6_000,
    *,
    random_state: int = 42,
    positive_rate: float = 0.07,
    missing_rate: float = 0.12,
    n_cities: int = 120,
    n_noise_features: int = 8,
    split: bool = False,
    test_size: float = 0.25,
    inject_drift: bool = True,
    target_column: str = "churned",
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate a messy, imbalanced tabular dataset.

    Parameters
    ----------
    n_rows:
        Total rows to generate.
    random_state:
        Seed for full reproducibility.
    positive_rate:
        Approximate share of the positive class.
    missing_rate:
        Base missingness rate for the columns that carry ``NaN``.
    n_cities:
        Cardinality of the high-cardinality categorical column.
    n_noise_features:
        Number of pure-noise columns the feature selector should discard.
    split:
        When ``True``, return ``(train, test)`` with an optional injected
        covariate shift in the test half.
    test_size:
        Holdout fraction used when ``split`` is ``True``.
    inject_drift:
        Shift ``annual_income`` upward and re-weight ``acquisition_channel`` in
        the test half so the drift auditor has something real to find.
    target_column:
        Name of the generated binary label.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, pandas.DataFrame]

    Examples
    --------
    >>> train, test = generate_messy_dataset(n_rows=500, split=True, random_state=0)
    >>> train.shape[0] + test.shape[0]
    500
    >>> "churned" in train.columns
    True

    """
    rng = np.random.default_rng(random_state)

    # ---------------------------------------------------------------- #
    # Genuine signal-bearing features
    # ---------------------------------------------------------------- #
    tenure_months = rng.gamma(shape=2.2, scale=9.0, size=n_rows).round(1)
    annual_income = rng.lognormal(mean=10.6, sigma=0.75, size=n_rows)  # heavy right tail
    session_count = rng.poisson(lam=18, size=n_rows).astype(float)
    support_tickets = rng.poisson(lam=1.3, size=n_rows).astype(float)
    discount_pct = np.clip(rng.beta(2.0, 8.0, size=n_rows) * 100, 0, 90)
    satisfaction = rng.integers(1, 6, size=n_rows)  # 1-5 rating: categorical, not numeric

    plan_tier = rng.choice(_PLAN_TIERS, size=n_rows, p=[0.34, 0.31, 0.25, 0.10])
    channel = rng.choice(_CHANNELS, size=n_rows, p=[0.30, 0.25, 0.15, 0.18, 0.12])
    device = rng.choice(_DEVICES, size=n_rows, p=[0.40, 0.42, 0.18])
    city = np.array([f"city_{i:03d}" for i in rng.integers(0, n_cities, size=n_rows)])

    signup_date = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 1_400, size=n_rows), unit="D"
    )

    # ---------------------------------------------------------------- #
    # Target: a known logistic model over standardised drivers
    # ---------------------------------------------------------------- #
    def _z(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return (values - values.mean()) / (values.std() or 1.0)

    tier_effect = pd.Series(plan_tier).map(
        {"free": 0.85, "basic": 0.25, "pro": -0.35, "enterprise": -0.80}
    ).to_numpy()
    city_effect = (
        pd.Series(city).astype("category").cat.codes.to_numpy() % 7 - 3
    ) * 0.09  # mild, learnable high-cardinality signal

    logit = (
        -1.25 * _z(tenure_months)
        + 0.95 * _z(support_tickets)
        - 0.70 * _z(np.log1p(annual_income))
        - 0.55 * _z(session_count)
        + 0.45 * _z(discount_pct)
        - 0.60 * _z(satisfaction)
        + tier_effect
        + city_effect
        + rng.normal(0, 0.45, size=n_rows)
    )
    # Shift the intercept so the realised positive rate matches the request.
    intercept = np.quantile(logit, 1 - positive_rate)
    probability = 1.0 / (1.0 + np.exp(-(logit - intercept)))
    target = (rng.uniform(size=n_rows) < probability).astype(int)

    # ---------------------------------------------------------------- #
    # Assemble, then vandalise
    # ---------------------------------------------------------------- #
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(100_000, 100_000 + n_rows),
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "tenure_months": tenure_months,
            "annual_income": annual_income.round(2),
            "monthly_spend": np.round(annual_income / 12.0 * rng.uniform(0.02, 0.12, n_rows), 2),
            "session_count": session_count,
            "support_tickets": support_tickets,
            "discount_pct": discount_pct.round(2),
            "satisfaction": satisfaction,
            "plan_tier": plan_tier,
            "acquisition_channel": channel,
            "device": device,
            "city": city,
            "data_version": "v3",  # constant -> must be dropped
            target_column: target,
        }
    )

    # Collinear duplicate: correlation pruning should remove one of the pair.
    frame["tenure_months_copy"] = frame["tenure_months"] * 1.02 + rng.normal(
        0, 0.05, n_rows
    )

    # Pure noise: the selector must discard these.
    for index in range(n_noise_features):
        frame[f"noise_{index + 1}"] = rng.normal(0, 1, n_rows)

    # Numbers stored as formatted strings.
    frame["monthly_spend"] = frame["monthly_spend"].map(lambda v: f"${v:,.2f}")

    # Sensor spikes: 1.5% of session counts multiplied 20-60x.
    spike_idx = rng.choice(n_rows, size=max(1, int(0.015 * n_rows)), replace=False)
    frame.loc[spike_idx, "session_count"] *= rng.integers(20, 60, size=spike_idx.size)

    # Missingness - MCAR on two columns...
    for column in ("annual_income", "discount_pct"):
        mask = rng.uniform(size=n_rows) < missing_rate
        frame.loc[mask, column] = np.nan

    # ...and structured (MAR) on another: enterprise accounts skip the survey.
    survey_missing = (frame["plan_tier"] == "enterprise") & (
        rng.uniform(size=n_rows) < 0.65
    )
    frame.loc[survey_missing | (rng.uniform(size=n_rows) < missing_rate / 2), "satisfaction"] = (
        np.nan
    )

    # A column so empty it must be dropped rather than imputed.
    beta_mask = rng.uniform(size=n_rows) < 0.82
    frame["beta_feature_score"] = rng.normal(0, 1, n_rows)
    frame.loc[beta_mask, "beta_feature_score"] = np.nan

    # Missing categoricals.
    device_missing = rng.uniform(size=n_rows) < missing_rate / 2
    frame.loc[device_missing, "device"] = None

    frame = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if not split:
        return frame

    # ---------------------------------------------------------------- #
    # Train / test split with optional covariate shift
    # ---------------------------------------------------------------- #
    n_test = int(round(n_rows * test_size))
    test = frame.iloc[:n_test].copy().reset_index(drop=True)
    train = frame.iloc[n_test:].copy().reset_index(drop=True)

    if inject_drift:
        shift_rng = np.random.default_rng(random_state + 1)
        # Incomes rise ~35% in the holdout period.
        test["annual_income"] = test["annual_income"] * shift_rng.uniform(
            1.25, 1.45, size=len(test)
        )
        # Marketing mix moves hard toward paid search.
        reassign = shift_rng.uniform(size=len(test)) < 0.45
        test.loc[reassign, "acquisition_channel"] = "paid_search"
        # A city that only exists in production.
        unseen = shift_rng.uniform(size=len(test)) < 0.05
        test.loc[unseen, "city"] = "city_999"

    return train, test
