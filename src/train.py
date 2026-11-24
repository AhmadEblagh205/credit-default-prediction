"""Train the defensible v2 credit-default decision model.

Protocol
--------
The coursework 24k development file is split once into:
  * 60% model fitting / cross-validation
  * 20% probability calibration
  * 20% policy diagnostics
The original 6k coursework test file remains untouched until final scoring.

The deployed model is a behaviour-only regularised logistic regression.
Direct demographic variables are excluded from the model and retained for a
separate governance audit.  The model is calibrated on a disjoint split and
the 5:1 decision threshold is derived from the stated costs rather than tuned
on the final test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, confusion_matrix

from decision import (bootstrap_empirical_threshold, cost_derived_threshold,
                      sweep_threshold)
from features import behavioural_features, clean, read_credit_csv, target
from modeling import (calibrate_frozen, decision_metrics, development_splits,
                      probability_metrics, tune_logistic)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def _calibration_bins(y, p, n=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n - 1)
    out = []
    for b in range(n):
        m = idx == b
        if m.sum() >= 20:
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def _save_figures(y_test, p_raw, p_cal, y_policy, p_policy,
                  bootstrap, model, X_test, threshold, fn_cost, fp_cost):
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for p, label, marker in [
        (p_raw, f"raw, Brier {brier_score_loss(y_test, p_raw):.3f}", "o-"),
        (p_cal, f"calibrated, Brier {brier_score_loss(y_test, p_cal):.3f}", "s-"),
    ]:
        pts = _calibration_bins(y_test, p)
        ax.plot([x for x, _, _ in pts], [z for _, z, _ in pts], marker,
                lw=1.6, label=label)
    ax.set(xlabel="Predicted default probability", ylabel="Observed default rate",
           title="Held-out reliability")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "calibration_test.png", dpi=160)
    plt.close(fig)

    rows, best = sweep_threshold(y_policy, p_policy, fn_cost, fp_cost)
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    ax.plot([r["threshold"] for r in rows],
            [r["cost_per_client"] for r in rows], lw=1.8,
            label="policy-split expected cost")
    ax.axvline(threshold, ls="--", lw=1.4,
               label=f"cost-derived threshold {threshold:.3f}")
    ax.axvline(best["threshold"], ls=":", lw=1.4,
               label=f"empirical policy optimum {best['threshold']:.3f}")
    ax.set(xlabel="Decision threshold", ylabel="Illustrative cost / client",
           title=f"Threshold diagnostics at FN:FP = {fn_cost:g}:{fp_cost:g}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "threshold_cost.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.hist(bootstrap["values"], bins=np.arange(0.02, 0.602, 0.01), alpha=0.8)
    ax.axvline(threshold, ls="--", lw=1.5, label="cost-derived threshold")
    ax.axvspan(bootstrap["p2_5"], bootstrap["p97_5"], alpha=0.15,
               label="95% bootstrap range")
    ax.set(xlabel="Empirical cost-minimising threshold",
           ylabel="Bootstrap count", title="Policy-threshold stability")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(FIGURES / "threshold_bootstrap.png", dpi=160)
    plt.close(fig)

    # Exact raw-logit coefficients of the fitted logistic model.
    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=X_test.columns)
    order = coefs.abs().sort_values().index
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.hlines(range(len(order)), 0, coefs[order], lw=2)
    ax.plot(coefs[order], range(len(order)), "o")
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set(xlabel="Standardised logistic coefficient",
           title="Interpretable behaviour-only scorecard")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout(); fig.savefig(FIGURES / "coefficients.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=RAW / "CreditCard_train.csv")
    ap.add_argument("--test-csv", default=RAW / "CreditCard_test.csv")
    ap.add_argument("--fn-cost", type=float, default=5.0)
    ap.add_argument("--fp-cost", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev_raw = clean(read_credit_csv(args.train_csv))
    test_raw = clean(read_credit_csv(args.test_csv))
    X, y = behavioural_features(dev_raw), target(dev_raw)
    X_test, y_test = behavioural_features(test_raw), target(test_raw)
    splits = development_splits(y, args.seed)

    search = tune_logistic(X, y, splits.fit)
    model = search.best_estimator_
    calibrator = calibrate_frozen(model, X.loc[splits.calibration],
                                  y.loc[splits.calibration])

    p_cal_raw = model.predict_proba(X.loc[splits.calibration])[:, 1]
    p_policy = calibrator.predict_proba(X.loc[splits.policy])[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]
    p_test = calibrator.predict_proba(X_test)[:, 1]

    theoretical = cost_derived_threshold(args.fn_cost, args.fp_cost)
    policy_rows, policy_best = sweep_threshold(
        y.loc[splits.policy], p_policy, args.fn_cost, args.fp_cost)
    boot = bootstrap_empirical_threshold(
        y.loc[splits.policy], p_policy, args.fn_cost, args.fp_cost,
        n_boot=300, seed=args.seed + 100)

    _save_figures(y_test, p_test_raw, p_test,
                  y.loc[splits.policy], p_policy, boot, model, X_test,
                  theoretical, args.fn_cost, args.fp_cost)

    cv_i = int(search.best_index_)
    summary = {
        "protocol": {
            "development_rows": int(len(y)),
            "fit_rows": int(len(splits.fit)),
            "calibration_rows": int(len(splits.calibration)),
            "policy_rows": int(len(splits.policy)),
            "test_rows": int(len(y_test)),
            "demographics_in_model": False,
            "smote_in_deployed_model": False,
        },
        "model_selection": {
            "best_C": float(search.best_params_["clf__C"]),
            "cv_roc_auc_mean": float(search.cv_results_["mean_test_score"][cv_i]),
            "cv_roc_auc_fold_sd": float(search.cv_results_["std_test_score"][cv_i]),
        },
        "calibration": {
            "calibration_split_raw_brier": float(
                brier_score_loss(y.loc[splits.calibration], p_cal_raw)),
            "test_raw_brier": float(brier_score_loss(y_test, p_test_raw)),
            "test_calibrated_brier": float(brier_score_loss(y_test, p_test)),
        },
        "policy": {
            "fn_cost": float(args.fn_cost), "fp_cost": float(args.fp_cost),
            "cost_derived_threshold": theoretical,
            "empirical_policy_threshold": float(policy_best["threshold"]),
            "empirical_threshold_bootstrap_95": [boot["p2_5"], boot["p97_5"]],
            "policy_cost_at_cost_derived": decision_metrics(
                y.loc[splits.policy], p_policy, theoretical,
                args.fn_cost, args.fp_cost)["cost_per_client"],
            "policy_cost_at_empirical": policy_best["cost_per_client"],
        },
        "held_out_test": {
            **probability_metrics(y_test, p_test),
            "raw_roc_auc": probability_metrics(y_test, p_test_raw)["roc_auc"],
            "decision_at_cost_derived_threshold": decision_metrics(
                y_test, p_test, theoretical, args.fn_cost, args.fp_cost),
        },
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
