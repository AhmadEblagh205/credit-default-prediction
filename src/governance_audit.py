"""Governance diagnostics, not a claim of legal or ethical fairness.

The deployed specification deliberately excludes SEX, MARRIAGE, AGE and
EDUCATION.  This module quantifies (a) the discrimination trade-off from that
choice and (b) held-out error/calibration differences across source-reported
sex and age groups under one global threshold.  Group-specific thresholds are
not proposed or deployed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decision import cost_derived_threshold
from features import (behavioural_features, clean, demographic_audit_frame,
                      demographic_enriched_features, read_credit_csv, target)
from modeling import calibrate_frozen, development_splits

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def _fit_lr(X, y, fit_idx):
    m = Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(C=0.1, max_iter=3000,
                                             random_state=11))])
    m.fit(X.loc[fit_idx], y.loc[fit_idx])
    return m


def _group_rows(group_name, group, y, p, threshold):
    rows = []
    for value in pd.Series(group).dropna().unique():
        m = np.asarray(group == value)
        yy, pp = np.asarray(y)[m], np.asarray(p)[m]
        pred = pp >= threshold
        tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
        rows.append({
            "group": group_name, "value": str(value), "n": int(m.sum()),
            "default_rate": float(yy.mean()),
            "recall": float(tp / max(tp + fn, 1)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "flag_rate": float(pred.mean()),
            "brier": float(brier_score_loss(yy, pp)),
        })
    return rows


def main(seed=42):
    dev = clean(read_credit_csv(RAW / "CreditCard_train.csv"))
    test = clean(read_credit_csv(RAW / "CreditCard_test.csv"))
    Xb, Xd, y = (behavioural_features(dev), demographic_enriched_features(dev),
                 target(dev))
    Xbt, Xdt, yt = (behavioural_features(test), demographic_enriched_features(test),
                    target(test))
    s = development_splits(y, seed)

    mb, md = _fit_lr(Xb, y, s.fit), _fit_lr(Xd, y, s.fit)
    p_b_policy = mb.predict_proba(Xb.loc[s.policy])[:, 1]
    p_d_policy = md.predict_proba(Xd.loc[s.policy])[:, 1]
    p_b_test = mb.predict_proba(Xbt)[:, 1]
    p_d_test = md.predict_proba(Xdt)[:, 1]

    cal = calibrate_frozen(mb, Xb.loc[s.calibration], y.loc[s.calibration])
    p_test = cal.predict_proba(Xbt)[:, 1]
    threshold = cost_derived_threshold(5.0, 1.0)

    demo = demographic_audit_frame(test).reset_index(drop=True)
    sex_label = demo["SEX"].map({1: "source category 1", 2: "source category 2"})
    age_band = pd.cut(demo["AGE"], bins=[20, 29, 39, 49, 59, 100],
                      labels=["21-29", "30-39", "40-49", "50-59", "60+"],
                      include_lowest=True)
    groups = (_group_rows("SEX", sex_label, yt.reset_index(drop=True), p_test, threshold) +
              _group_rows("AGE_BAND", age_band, yt.reset_index(drop=True), p_test, threshold))
    group_df = pd.DataFrame(groups)
    group_df.to_csv(REPORTS / "group_audit.csv", index=False)

    summary = {
        "deployed_features": "credit behaviour only; direct demographics excluded",
        "policy_auc_behavior_only": float(roc_auc_score(y.loc[s.policy], p_b_policy)),
        "policy_auc_with_demographics": float(roc_auc_score(y.loc[s.policy], p_d_policy)),
        "policy_auc_gain_from_demographics": float(
            roc_auc_score(y.loc[s.policy], p_d_policy) - roc_auc_score(y.loc[s.policy], p_b_policy)),
        "test_auc_behavior_only": float(roc_auc_score(yt, p_b_test)),
        "test_auc_with_demographics": float(roc_auc_score(yt, p_d_test)),
        "test_auc_gain_from_demographics": float(roc_auc_score(yt, p_d_test) - roc_auc_score(yt, p_b_test)),
        "global_threshold": threshold,
        "group_diagnostics": groups,
        "warning": "Diagnostic only. Group differences do not establish fairness/unfairness or legal compliance; small groups have high uncertainty.",
    }
    (REPORTS / "governance_audit.json").write_text(json.dumps(summary, indent=2))

    plot = group_df[group_df["group"] == "AGE_BAND"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(plot)); w = 0.34
    ax.bar(x - w/2, plot["recall"], width=w, label="Recall")
    ax.bar(x + w/2, plot["false_positive_rate"], width=w, label="False-positive rate")
    ax.set_xticks(x, plot["value"])
    ax.set(ylabel="Rate", xlabel="Age band (audit only)",
           title="Held-out group diagnostics under one global threshold")
    ax.legend(); ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(FIGURES / "group_audit.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
