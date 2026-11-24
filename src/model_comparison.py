"""Fair interpretability/performance comparison on identical data splits."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from features import behavioural_features, clean, read_credit_csv, target
from modeling import (calibrate_frozen, development_splits, probability_metrics,
                      tune_logistic, decision_metrics)
from decision import cost_derived_threshold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main(seed=42, fn_cost=5.0, fp_cost=1.0):
    dev = clean(read_credit_csv(RAW / "CreditCard_train.csv"))
    test = clean(read_credit_csv(RAW / "CreditCard_test.csv"))
    X, y = behavioural_features(dev), target(dev)
    Xt, yt = behavioural_features(test), target(test)
    s = development_splits(y, seed)

    lr_search = tune_logistic(X, y, s.fit)
    lr = lr_search.best_estimator_
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    hgb_search = GridSearchCV(
        HistGradientBoostingClassifier(max_iter=250, random_state=11),
        {"learning_rate": [0.05], "max_leaf_nodes": [7, 15],
         "l2_regularization": [0.0, 1.0]},
        scoring="roc_auc", cv=cv, n_jobs=1)
    hgb_search.fit(X.loc[s.fit], y.loc[s.fit])
    hgb = hgb_search.best_estimator_

    lr_cal = calibrate_frozen(lr, X.loc[s.calibration], y.loc[s.calibration])
    hgb_cal = calibrate_frozen(hgb, X.loc[s.calibration], y.loc[s.calibration])
    p_lr_raw = lr.predict_proba(Xt)[:, 1]
    p_hgb_raw = hgb.predict_proba(Xt)[:, 1]
    p_lr = lr_cal.predict_proba(Xt)[:, 1]
    p_hgb = hgb_cal.predict_proba(Xt)[:, 1]
    threshold = cost_derived_threshold(fn_cost, fp_cost)

    summary = {
        "same_fit_calibration_test_splits": True,
        "logistic": {
            "best_C": float(lr_search.best_params_["clf__C"]),
            "cv_auc": float(lr_search.best_score_),
            "raw_test": probability_metrics(yt, p_lr_raw),
            "calibrated_test": probability_metrics(yt, p_lr),
            "decision": decision_metrics(yt, p_lr, threshold, fn_cost, fp_cost),
            "explanation": "exact additive contributions to the raw logistic log-odds",
        },
        "hist_gradient_boosting": {
            "best_params": hgb_search.best_params_,
            "cv_auc": float(hgb_search.best_score_),
            "raw_test": probability_metrics(yt, p_hgb_raw),
            "calibrated_test": probability_metrics(yt, p_hgb),
            "decision": decision_metrics(yt, p_hgb, threshold, fn_cost, fp_cost),
        },
    }
    summary["held_out_auc_gap_hgb_minus_logistic"] = (
        summary["hist_gradient_boosting"]["raw_test"]["roc_auc"] -
        summary["logistic"]["raw_test"]["roc_auc"])

    # Paired bootstrap on the same held-out clients: quantify the uncertainty
    # around the discrimination/interpretability trade-off instead of quoting
    # a single AUC gap as if it were exact.
    rng = np.random.default_rng(123)
    yy = np.asarray(yt)
    gaps = []
    for _ in range(1000):
        idx = rng.integers(0, len(yy), len(yy))
        if np.unique(yy[idx]).size < 2:
            continue
        gaps.append(roc_auc_score(yy[idx], p_hgb_raw[idx]) -
                    roc_auc_score(yy[idx], p_lr_raw[idx]))
    q = np.quantile(gaps, [0.025, 0.5, 0.975])
    summary["auc_gap_paired_bootstrap_95"] = [float(q[0]), float(q[2])]
    summary["auc_gap_bootstrap_median"] = float(q[1])

    FIGURES.mkdir(parents=True, exist_ok=True)
    names = ["Logistic", "HistGradientBoosting"]
    aucs = [summary["logistic"]["raw_test"]["roc_auc"],
            summary["hist_gradient_boosting"]["raw_test"]["roc_auc"]]
    briers = [summary["logistic"]["calibrated_test"]["brier"],
              summary["hist_gradient_boosting"]["calibrated_test"]["brier"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(names, aucs)
    ax.set_ylim(0.72, 0.83); ax.set_ylabel("Held-out ROC-AUC")
    ax.set_title("Interpretability / discrimination trade-off")
    for i, (a, b) in enumerate(zip(aucs, briers)):
        ax.text(i, a + 0.003, f"AUC {a:.3f}\ncal Brier {b:.3f}",
                ha="center", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(FIGURES / "model_tradeoff.png", dpi=160)
    plt.close(fig)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "model_comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
