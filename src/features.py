"""Feature preparation for the credit-default benchmark.

The original UCL group project used manual outlier removal, SMOTE and a
recursive engineered-expense chain.  The public rebuild keeps the original
notebook/report for provenance but uses a cleaner modelling layer:

* target and identifier are never features;
* direct demographic attributes are excluded from the deployed model and
  retained only for a separate governance audit;
* repayment-status codes are converted into months of delinquency (values
  <= 0 mean no positive delay for this purpose);
* bills and payments are scaled by the customer's credit limit rather than
  arbitrarily deleting financially unusual customers.

This makes the units explicit and avoids treating category codes as ordered
continuous variables.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYS = [f"PAY_AMT{i}" for i in range(1, 7)]
PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]
TARGET_ALIASES = ("DEF", "default payment next month", "default",
                  "default.payment.next.month")


def read_credit_csv(path: str | Path) -> pd.DataFrame:
    """Read either the coursework two-row CSV or a normal one-row CSV."""
    first = pd.read_csv(path, nrows=0)
    if any(str(c).startswith("X1") for c in first.columns):
        return pd.read_csv(path, header=1)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names/codes without dropping customers."""
    out = df.copy()
    out = out.rename(columns={
        "default payment next month": "DEF",
        "default": "DEF",
        "default.payment.next.month": "DEF",
        "PAY_0": "PAY_1",
    })
    required = {"LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                *BILLS, *PAYS, *PAY_STATUS}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "DEF" not in out.columns:
        raise ValueError("Missing default target column")

    # The source uses 0/5/6 as undefined/other education and 0 for undefined
    # marital status.  Keep them as one explicit 'other' bucket for auditing.
    out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    out["MARRIAGE"] = out["MARRIAGE"].replace({0: 3})

    numeric = ["LIMIT_BAL", "AGE", *BILLS, *PAYS, *PAY_STATUS, "DEF"]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="raise")
    if (out["LIMIT_BAL"] <= 0).any():
        raise ValueError("LIMIT_BAL must be positive")
    if not set(out["DEF"].dropna().unique()).issubset({0, 1}):
        raise ValueError("DEF must be binary")
    return out


def behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the final model matrix from credit behaviour only.

    SEX, MARRIAGE, AGE and EDUCATION are deliberately omitted from the
    deployed feature matrix.  The feature set is intentionally compact:
    recent delinquency plus the number of delinquent months capture payment
    history without stacking several highly collinear summaries whose signs
    become hard to defend in a linear scorecard.
    """
    d = clean(df)
    lim = d["LIMIT_BAL"].astype(float)
    delays = d[PAY_STATUS].clip(lower=0).astype(float)

    X = pd.DataFrame(index=d.index)
    X["LOG_LIMIT_BAL"] = np.log1p(lim)
    X["RECENT_UTIL"] = d["BILL_AMT1"] / lim
    X["MEAN_UTIL"] = d[BILLS].mean(axis=1) / lim
    X["BILL_CHANGE_TO_LIMIT"] = (d["BILL_AMT1"] - d["BILL_AMT6"]) / lim
    X["MEAN_PAYMENT_TO_LIMIT"] = d[PAYS].mean(axis=1) / lim
    # Stabilised payment-to-bill measure: the credit limit keeps the
    # denominator defined for zero/negative statement balances.
    X["PAYMENT_TO_EXPOSURE"] = d[PAYS].sum(axis=1) / (
        d[BILLS].abs().sum(axis=1) + lim
    )
    X["RECENT_DELAY"] = delays["PAY_1"]
    X["DELINQUENT_MONTHS"] = (delays > 0).sum(axis=1).astype(float)

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite engineered feature encountered")
    return X.astype(float)


def target(df: pd.DataFrame) -> pd.Series:
    return clean(df)["DEF"].astype(int)


def demographic_audit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attributes retained for audit only, never the deployed model."""
    d = clean(df)
    return d[["SEX", "EDUCATION", "MARRIAGE", "AGE"]].copy()


def demographic_enriched_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behaviour matrix + one-hot demographics, for the governance audit.

    This is a comparison model only; it is not the deployed specification.
    """
    X = behavioural_features(df)
    demo = demographic_audit_frame(df)
    demo = pd.get_dummies(demo, columns=["SEX", "EDUCATION", "MARRIAGE"],
                          dtype=float)
    return pd.concat([X, demo.astype(float)], axis=1)
